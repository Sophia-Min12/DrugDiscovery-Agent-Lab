"""The agent: a LangGraph state machine whose last word belongs to the verifier.

    plan -> act -> draft -> verify --(all claims supported)--> report
                              ^        |
                              +-revise-+  (at most `max_revisions` times)

The LLM plans tool calls and writes prose. It never decides whether a number is
right: `verify` recomputes every claim with a tool, `revise` hands the LLM the
recomputed values, and `report` marks inline whatever is still wrong after the last
revision. A wrong number can survive the loop; it cannot leave unmarked.
"""
from __future__ import annotations

import json
import re
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from . import drugs, tools
from .claims import Check, Verdict
from .extract import extract, mentions
from .llm import LLM
from .verifier import Verifier

FAILING = {Verdict.CONTRADICTED, Verdict.AMBIGUOUS, Verdict.UNVERIFIABLE, Verdict.UNQUALIFIED}

PLAN_SYSTEM = """You plan tool calls for a drug-discovery assistant.
Available tools:
{catalog}

Reply with JSON only: {{"calls": [{{"tool": "<tool name>", "args": {{"<param>": "<value>"}}}}]}}
Call every tool needed to answer the question, once per compound. Use only the tools listed."""

STYLE = ("Write 4-8 sentences of plain prose for a medicinal chemist; no tables. When a property has more "
         "than one common calculation method, name the method of every value you quote, e.g. 'logP 3.07 (Crippen)'.")

DRAFT_TOOLS_SYSTEM = ("You are a careful medicinal-chemistry assistant. Answer using ONLY the tool results "
                      "provided. Every number you state must come from the tool results; if the question asks "
                      "for something the tools do not provide, say it is not available. " + STYLE)
DRAFT_CLOSED_SYSTEM = "You are a medicinal-chemistry assistant. Answer from your own knowledge. " + STYLE


class AgentState(TypedDict, total=False):
    question: str
    subjects: list[str]
    plan: list[dict]
    plan_source: str              # "llm" | "fallback" | "none"
    observations: list[dict]
    draft: str
    drafts: list[str]
    checks: list[Check]
    unattributed: list[tuple[int, int, str]]
    revisions: int
    llm_calls: int
    report: str


def find_subjects(question: str) -> list[str]:
    names = sorted(set(drugs.names()) | set(drugs.SYNONYMS), key=len, reverse=True)
    found = [name for _, name in mentions(question, names)]
    for token in re.findall(r"[A-Za-z0-9@+\-\[\]()=#/\\.%]{3,}", question):
        if re.search(r"[=()\[\]#@\d]", token) and drugs.resolve(token) and not drugs.lookup(token):
            found.append(token)
    return list(dict.fromkeys(found))


def _parse_plan(text: str, subjects: list[str]) -> list[dict] | None:
    try:
        calls = json.loads(text).get("calls", [])
    except (json.JSONDecodeError, AttributeError):
        return None
    valid = []
    for item in calls if isinstance(calls, list) else []:
        if not isinstance(item, dict) or item.get("tool") not in tools.REGISTRY:
            continue
        args = item.get("args") or {}
        allowed = tools.REGISTRY[item["tool"]].params
        if isinstance(args, dict) and set(args) <= set(allowed):
            valid.append({"tool": item["tool"], "args": args})
    return valid or None


def fallback_plan(subjects: list[str]) -> list[dict]:
    return [{"tool": name, "args": {"compound": s}} for s in subjects for name in ("compute_descriptors", "drug_likeness")]


def feedback(checks: list[Check], unattributed: list[tuple[int, int, str]], text: str) -> str:
    lines = ["A verifier recomputed every number in your draft with the tools. Fix these problems:"]
    for check in checks:
        if check.verdict not in FAILING:
            continue
        quoted = f'"{check.claim.text}" ({check.claim.subject})'
        if check.verdict is Verdict.CONTRADICTED:
            lines.append(f"- {quoted} is WRONG. Tool values: {check.expected()}.")
        elif check.verdict is Verdict.AMBIGUOUS:
            lines.append(f"- {quoted} is true under one method and false under another ({check.note}). "
                         f"Tool values: {check.expected()}. Name the method you mean.")
        elif check.verdict is Verdict.UNQUALIFIED:
            lines.append(f"- {quoted} is a model prediction; give its uncertainty interval ({check.expected()}).")
        else:
            lines.append(f"- {quoted} cannot be checked by any tool ({check.note}). Remove it or say it is unverified.")
    for start, end, number in unattributed:
        context = text[max(0, start - 30):min(len(text), end + 20)].replace("\n", " ")
        lines.append(f'- the number {number} in "...{context}..." is not tied to any property the tools compute. '
                     f'If it is a property value, say which property and method it is, e.g. "TPSA 63.6 (PubChem)"; '
                     f'otherwise remove it.')
    lines.append("Rewrite the whole answer. Keep every correct statement as it is.")
    return "\n".join(lines)


def annotate(text: str, checks: list[Check], unattributed: list[tuple[int, int, str]]) -> str:
    """Mark every claim that did not verify, right where it appears."""
    marks = []
    for check in checks:
        if check.verdict in FAILING and check.claim.span:
            label = check.verdict.value.upper()
            detail = f": tools say {check.expected()}" if check.verdict is not Verdict.UNVERIFIABLE else ""
            marks.append((check.claim.span[1], f" [{label}{detail}]"))
    marks += [(end, " [UNVERIFIED]") for _, end, _ in unattributed]
    for position, mark in sorted(marks, reverse=True):
        text = text[:position] + mark + text[position:]
    return text


def verification_table(checks: list[Check]) -> str:
    rows = ["| claim | verdict | tool values |", "|---|---|---|"]
    rows += [f"| {c.claim.describe()} | {c.verdict.value} | {c.expected()} |" for c in checks]
    return "\n".join(rows)


def build_agent(llm: LLM, verifier: Verifier | None = None, use_tools: bool = True, max_revisions: int = 2):

    def plan(state: AgentState) -> dict:
        subjects = find_subjects(state["question"])
        if not use_tools:
            return {"subjects": subjects, "plan": [], "plan_source": "none", "llm_calls": 0}
        reply = llm.complete([
            {"role": "system", "content": PLAN_SYSTEM.format(catalog=tools.catalog())},
            {"role": "user", "content": state["question"]},
        ], json_mode=True)
        calls = _parse_plan(reply, subjects)
        source = "llm" if calls else "fallback"
        return {"subjects": subjects, "plan": calls or fallback_plan(subjects), "plan_source": source, "llm_calls": 1}

    def act(state: AgentState) -> dict:
        observations = [{"tool": c["tool"], "args": c["args"], "result": tools.call(c["tool"], c["args"])}
                        for c in state["plan"]]
        return {"observations": observations}

    def _context(state: AgentState) -> list[dict]:
        if not use_tools:
            return [{"role": "system", "content": DRAFT_CLOSED_SYSTEM},
                    {"role": "user", "content": state["question"]}]
        results = "\n".join(tools.as_json(o) for o in state["observations"])
        return [{"role": "system", "content": DRAFT_TOOLS_SYSTEM},
                {"role": "user", "content": f"Question: {state['question']}\n\nTool results:\n{results}"}]

    def draft(state: AgentState) -> dict:
        text = llm.complete(_context(state))
        return {"draft": text, "drafts": [text], "revisions": 0, "llm_calls": state["llm_calls"] + 1}

    def verify(state: AgentState) -> dict:
        subject = state["subjects"][0] if len(state.get("subjects", [])) == 1 else None
        dataset = next((o["args"].get("dataset", "demo") for o in state.get("observations", [])
                        if o["tool"] in ("differential_expression", "gene_result")), "demo")
        found = extract(state["draft"], subject=subject, dataset=dataset)
        checker = verifier or default_verifier(dataset)
        return {"checks": checker.check_all(found.claims), "unattributed": found.unattributed}

    def route(state: AgentState) -> str:
        failing = any(c.verdict in FAILING for c in state["checks"]) or state["unattributed"]
        return "revise" if failing and state["revisions"] < max_revisions else "report"

    def revise(state: AgentState) -> dict:
        messages = _context(state) + [
            {"role": "assistant", "content": state["draft"]},
            {"role": "user", "content": feedback(state["checks"], state["unattributed"], state["draft"])},
        ]
        text = llm.complete(messages)
        return {"draft": text, "drafts": state["drafts"] + [text], "revisions": state["revisions"] + 1,
                "llm_calls": state["llm_calls"] + 1}

    def report(state: AgentState) -> dict:
        body = annotate(state["draft"], state["checks"], state["unattributed"])
        return {"report": f"{body}\n\n### Verification\n{verification_table(state['checks'])}"}

    graph = StateGraph(AgentState)
    for name, fn in (("plan", plan), ("act", act), ("draft", draft), ("verify", verify),
                     ("revise", revise), ("report", report)):
        graph.add_node(name, fn)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "act")
    graph.add_edge("act", "draft")
    graph.add_edge("draft", "verify")
    graph.add_conditional_edges("verify", route, {"revise": "revise", "report": "report"})
    graph.add_edge("revise", "verify")
    graph.add_edge("report", END)
    return graph.compile()


def default_verifier(dataset: str = "demo") -> Verifier:
    """Chemistry and omics checks always; GNN checks when torch is installed."""
    from .omics import omics_source_for
    verifier = Verifier().add_source(omics_source_for(dataset))
    try:
        from .gnn.predict import gnn_source
        verifier.add_source(gnn_source)
    except ImportError:
        pass
    return verifier


def run(question: str, llm: LLM, **options: Any) -> AgentState:
    return build_agent(llm, **options).invoke({"question": question})
