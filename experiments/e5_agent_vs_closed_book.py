"""E5 - what reaches the reader: closed book vs tools vs tools + verifier.

The same question, for every drug in the reference table, under four conditions:

    closed_book           the LLM answers from memory
    closed_book_verified  ... then the verifier's feedback (which carries tool values) drives revision
    tools                 the LLM plans tool calls and answers from their results, once
    tools_verified        ... then verifier-driven revision, at most twice

Every final answer is scored by the same verifier, and `summary` always re-scores the
stored text with the current extractor and verifier, so every row is judged by the same
code. Runs are appended to a JSONL file as they finish, so an interrupted run resumes.

    python experiments/e5_agent_vs_closed_book.py run [--model llama3.1:8b] [--n 32] [--drugs aspirin,ibuprofen]
    python experiments/e5_agent_vs_closed_book.py summary
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drugagent import drugs  # noqa: E402
from drugagent.agent import build_agent, default_verifier  # noqa: E402
from drugagent.extract import extract  # noqa: E402
from drugagent.llm import OllamaLLM  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results" / "e5_runs.jsonl"
QUESTION = ("Summarise the drug-likeness of {drug} for a medicinal chemist: molecular weight, logP, TPSA, "
            "hydrogen-bond donors and acceptors, rotatable bonds, and whether it passes Lipinski's Rule of Five.")
ASKED = ("mw", "logp", "tpsa", "hbd", "hba", "rotb", "lipinski_pass")
CONDITIONS = {
    "closed_book": dict(use_tools=False, max_revisions=0),
    "closed_book_verified": dict(use_tools=False, max_revisions=2),
    "tools": dict(use_tools=True, max_revisions=0),
    "tools_verified": dict(use_tools=True, max_revisions=2),
}


def done() -> set[tuple[str, str]]:
    if not RESULTS.exists():
        return set()
    return {(r["drug"], r["condition"]) for r in map(json.loads, RESULTS.read_text(encoding="utf-8").splitlines())}


def record(drug: str, condition: str, state: dict, seconds: float, model: str) -> dict:
    return {
        "drug": drug, "condition": condition, "model": model, "seconds": round(seconds, 1),
        "plan_source": state.get("plan_source"), "revisions": state.get("revisions", 0),
        "llm_calls": state.get("llm_calls"), "drafts": state.get("drafts"), "final": state.get("draft"),
        "report": state.get("report"),
        "checks": [{"prop": c.claim.prop, "value": c.claim.value, "method": c.claim.method, "op": c.claim.op,
                    "subject": c.claim.subject, "text": c.claim.text, "verdict": c.verdict.value,
                    "expected": c.expected()} for c in state.get("checks", [])],
        "unattributed": [number for _, _, number in state.get("unattributed", [])],
    }


def run(model: str, names: list[str]) -> None:
    llm = OllamaLLM(model)
    agents = {name: build_agent(llm, **options) for name, options in CONDITIONS.items()}
    finished = done()
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    for drug in names:
        for condition, agent in agents.items():
            if (drug, condition) in finished:
                continue
            start = time.perf_counter()
            state = agent.invoke({"question": QUESTION.format(drug=drug)})
            row = record(drug, condition, state, time.perf_counter() - start, llm.name)
            with RESULTS.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            verdicts = Counter(c["verdict"] for c in row["checks"])
            print(f"{drug:16s} {condition:22s} {row['seconds']:6.1f}s rev={row['revisions']} "
                  f"{dict(verdicts)} unattributed={len(row['unattributed'])}", flush=True)


def score(text: str, drug: str) -> tuple[list[dict], list[str]]:
    found = extract(text, subject=drug)
    checks = [{"prop": c.claim.prop, "value": c.claim.value, "method": c.claim.method, "op": c.claim.op,
               "subject": c.claim.subject, "text": c.claim.text, "verdict": c.verdict.value, "expected": c.expected()}
              for c in default_verifier().check_all(found.claims)]
    return checks, [number for _, _, number in found.unattributed]


def rescore(row: dict) -> dict:
    checks, unattributed = score(row["final"], row["drug"])
    first, _ = score(row["drafts"][0], row["drug"])
    wrong = lambda cs: sum(c["verdict"] == "contradicted" for c in cs)  # noqa: E731
    return {**row, "checks": checks, "unattributed": unattributed,
            "first_draft_contradicted": wrong(first), "final_contradicted": wrong(checks)}


def unique_checks(row: dict) -> list[dict]:
    """A report that repeats "MW 206.28" in its summary sentence makes one claim, not two."""
    seen, out = set(), []
    for c in row["checks"]:
        key = (c["subject"], c["prop"], c["value"], c["method"], c["op"])
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def coverage(row: dict) -> tuple[int, int]:
    """Of the seven properties the question asks for: how many are stated, and stated without error."""
    stated, correct = 0, 0
    for prop in ASKED:
        verdicts = [c["verdict"] for c in row["checks"] if c["prop"] == prop]
        stated += bool(verdicts)
        correct += bool(verdicts) and all(v in ("supported", "ambiguous") for v in verdicts)
    return stated, correct


def summary() -> dict:
    rows = [rescore(json.loads(line)) for line in RESULTS.read_text(encoding="utf-8").splitlines()]
    by_condition = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)
    table = {}
    for condition in CONDITIONS:
        group = by_condition.get(condition, [])
        if not group:
            continue
        verdicts = Counter(c["verdict"] for r in group for c in unique_checks(r))
        stated, correct = map(sum, zip(*(coverage(r) for r in group)))
        claims = sum(verdicts.values())
        wrong_reports = sum(any(c["verdict"] == "contradicted" for c in r["checks"]) for r in group)
        table[condition] = {
            "reports": len(group),
            "claims": claims,
            **{v: verdicts.get(v, 0) for v in ("supported", "contradicted", "ambiguous", "unverifiable", "unqualified")},
            "contradicted_rate": round(verdicts.get("contradicted", 0) / claims, 3) if claims else None,
            "reports_with_a_wrong_number": wrong_reports,
            "asked_properties_stated": round(stated / (len(ASKED) * len(group)), 3),
            "asked_properties_stated_correctly": round(correct / (len(ASKED) * len(group)), 3),
            "unattributed_numbers": sum(len(r["unattributed"]) for r in group),
            "mean_llm_calls": round(sum(r["llm_calls"] for r in group) / len(group), 2),
            "mean_seconds": round(sum(r["seconds"] for r in group) / len(group), 1),
            "planner_fallbacks": sum(r["plan_source"] == "fallback" for r in group),
            "revised_reports": sum(r["revisions"] > 0 for r in group),
            "revision_fixed_a_wrong_number": sum(r["final_contradicted"] < r["first_draft_contradicted"] for r in group),
            "revision_made_it_worse": sum(r["final_contradicted"] > r["first_draft_contradicted"] for r in group),
        }
    by_prop = defaultdict(Counter)
    for row in by_condition.get("closed_book", []):
        for c in unique_checks(row):
            by_prop[c["prop"]][c["verdict"]] += 1
    out = {"conditions": table, "closed_book_by_property": {p: dict(c) for p, c in sorted(by_prop.items())}}
    (RESULTS.parent / "e5_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "summary"])
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--n", type=int, default=32, help="drugs to run, taken from a seeded shuffle of the table")
    parser.add_argument("--drugs", default="", help="comma-separated explicit subset instead")
    args = parser.parse_args()
    if args.command == "run":
        order = drugs.names()
        random.Random(0).shuffle(order)
        run(args.model, [d for d in args.drugs.split(",") if d] or order[:args.n])
    print(json.dumps(summary(), indent=2))
