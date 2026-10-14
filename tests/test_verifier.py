"""The verdict logic, on evidence whose truth is fixed by construction."""
from drugagent.claims import Claim, Evidence, Verdict
from drugagent.verifier import Verifier, judge, lipinski_variants


def verdict(claim, evidence):
    return judge(claim, evidence).verdict


def test_agreement_across_methods():
    evidence = Evidence({"a": 3.0, "b": 3.1})
    assert verdict(Claim("x", "logp", 3.05), evidence) is Verdict.SUPPORTED
    assert verdict(Claim("x", "logp", 4.0), evidence) is Verdict.CONTRADICTED


def test_methods_that_disagree_make_an_unnamed_claim_ambiguous():
    evidence = Evidence({"crippen": 3.07, "xlogp3": 3.5})
    assert verdict(Claim("x", "logp", 3.5), evidence) is Verdict.AMBIGUOUS
    assert verdict(Claim("x", "logp", 3.5, method="XLogP3"), evidence) is Verdict.SUPPORTED
    assert verdict(Claim("x", "logp", 3.5, method="crippen"), evidence) is Verdict.CONTRADICTED


def test_a_method_the_repo_cannot_compute_is_unverifiable_not_wrong():
    evidence = Evidence({"crippen": 3.07})
    assert verdict(Claim("x", "logp", 3.97, method="mlogp"), evidence) is Verdict.UNVERIFIABLE


def test_counts_must_match_exactly():
    evidence = Evidence({"lipinski": 4})
    assert verdict(Claim("x", "hba", 4), evidence) is Verdict.SUPPORTED
    assert verdict(Claim("x", "hba", 5), evidence) is Verdict.CONTRADICTED


def test_upper_bound_claims():
    evidence = Evidence({"welch_bh": 3e-5})
    assert verdict(Claim("G1", "padj", 1e-4, op="<"), evidence) is Verdict.SUPPORTED
    assert verdict(Claim("G1", "padj", 1e-6, op="<"), evidence) is Verdict.CONTRADICTED


def test_predictions_need_an_interval_and_must_fall_inside_the_models():
    evidence = Evidence({"gnn": -3.0}, interval=(-3.6, -2.4))
    assert verdict(Claim("x", "log_solubility", -3.1, interval=(-3.7, -2.5)), evidence) is Verdict.SUPPORTED
    assert verdict(Claim("x", "log_solubility", -3.1), evidence) is Verdict.UNQUALIFIED
    assert verdict(Claim("x", "log_solubility", -5.0, interval=(-5.5, -4.5)), evidence) is Verdict.CONTRADICTED


def test_unknown_subject_is_unverifiable():
    assert Verifier().check(Claim("unobtainium", "mw", 100.0)).verdict is Verdict.UNVERIFIABLE


def test_real_compounds_end_to_end():
    v = Verifier()
    assert v.check(Claim("aspirin", "mw", 180.16)).verdict is Verdict.SUPPORTED
    assert v.check(Claim("ibuprofen", "tpsa", 83.6)).verdict is Verdict.CONTRADICTED
    assert v.check(Claim("cyclosporine", "lipinski_pass", 0.0)).verdict is Verdict.SUPPORTED


def test_atorvastatin_passes_the_rule_of_five_under_one_logp_and_fails_under_another():
    # Crippen logP is above 5, XLogP3 is not: MW > 500 plus a logP violation is two
    # violations (fail) or one (pass) depending on a method choice nobody states.
    variants = lipinski_variants("atorvastatin")
    assert variants["crippen+lipinski"] == 2 and variants["xlogp3+lipinski"] == 1
    assert Verifier().check(Claim("atorvastatin", "lipinski_pass", 1.0)).verdict is Verdict.AMBIGUOUS


def test_a_value_labelled_with_the_wrong_method_is_caught():
    # The first E4 run reported 18 "false alarms". Every one was a template that
    # wrote Lipinski's donor count and labelled it RDKit's. Metformin: 5 vs 3.
    assert Verifier().check(Claim("metformin", "hbd", 5, method="rdkit")).verdict is Verdict.CONTRADICTED
    assert Verifier().check(Claim("metformin", "hbd", 5, method="lipinski")).verdict is Verdict.SUPPORTED
