"""The agent graph, driven by a scripted LLM so every path is deterministic."""
import json

from drugagent.agent import build_agent, feedback, find_subjects
from drugagent.claims import Verdict
from drugagent.llm import ScriptedLLM
from drugagent.verifier import Verifier

QUESTION = "Summarise the drug-likeness of ibuprofen."
PLAN = json.dumps({"calls": [{"tool": "compute_descriptors", "args": {"compound": "ibuprofen"}}]})
WRONG = "Ibuprofen has a molecular weight of 206.28 g/mol and a TPSA of 83.6."
RIGHT = "Ibuprofen has a molecular weight of 206.28 g/mol and a TPSA of 37.3."


def responder(drafts, plan=PLAN):
    """Plan on the planning prompt; otherwise return the next draft in `drafts`."""
    queue = list(drafts)

    def respond(messages):
        if messages[0]["content"].startswith("You plan"):
            return plan
        return queue.pop(0) if len(queue) > 1 else queue[0]
    return respond


def test_a_wrong_number_is_fixed_by_one_revision():
    llm = ScriptedLLM(responder([WRONG, RIGHT]))
    state = build_agent(llm, verifier=Verifier()).invoke({"question": QUESTION})
    assert state["revisions"] == 1
    assert all(check.verdict is Verdict.SUPPORTED for check in state["checks"])
    assert "[CONTRADICTED" not in state["report"]
    # the revision prompt carried the recomputed value back to the model
    assert "37.3" in llm.calls[-1][-1]["content"]


def test_a_model_that_never_fixes_is_stopped_and_marked():
    llm = ScriptedLLM(responder([WRONG]))
    state = build_agent(llm, verifier=Verifier(), max_revisions=2).invoke({"question": QUESTION})
    assert state["revisions"] == 2 and state["llm_calls"] == 1 + 1 + 2
    assert "83.6 [CONTRADICTED: tools say ertl=37.3" in state["report"]


def test_unparseable_plan_falls_back_to_the_default_tools():
    llm = ScriptedLLM(responder([RIGHT], plan="I would call some tools!"))
    state = build_agent(llm, verifier=Verifier()).invoke({"question": QUESTION})
    assert state["plan_source"] == "fallback"
    assert [o["tool"] for o in state["observations"]] == ["compute_descriptors", "drug_likeness"]


def test_unknown_tools_in_a_plan_are_dropped():
    plan = json.dumps({"calls": [{"tool": "rm_rf", "args": {}},
                                 {"tool": "compute_descriptors", "args": {"compound": "ibuprofen"}}]})
    state = build_agent(ScriptedLLM(responder([RIGHT], plan=plan)), verifier=Verifier()).invoke({"question": QUESTION})
    assert state["plan_source"] == "llm" and [c["tool"] for c in state["plan"]] == ["compute_descriptors"]


def test_closed_book_makes_no_tool_calls():
    llm = ScriptedLLM(responder([RIGHT]))
    state = build_agent(llm, verifier=Verifier(), use_tools=False, max_revisions=0).invoke({"question": QUESTION})
    assert state["observations"] == [] and state["llm_calls"] == 1
    assert "own knowledge" in llm.calls[0][0]["content"]


def test_unattributed_numbers_are_flagged_in_the_report():
    text = RIGHT + " Its plasma half-life is about 2 hours."
    state = build_agent(ScriptedLLM(responder([text])), verifier=Verifier(), max_revisions=0).invoke({"question": QUESTION})
    assert "2 [UNVERIFIED]" in state["report"]


def test_feedback_names_the_method_when_a_claim_is_ambiguous():
    from drugagent.extract import extract
    found = extract("Ibuprofen has a logP of 3.5.", subject="ibuprofen")
    message = feedback(Verifier().check_all(found.claims), found.unattributed, "")
    assert "Name the method" in message and "crippen=3.07" in message


def test_subjects_are_found_by_name_synonym_and_smiles():
    assert find_subjects("Compare paracetamol with CC(=O)Oc1ccccc1C(=O)O") == ["acetaminophen", "CC(=O)Oc1ccccc1C(=O)O"]
