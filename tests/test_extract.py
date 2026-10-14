"""The extractor, tested on phrasings an LLM actually produces."""
from drugagent.extract import extract


def claims(text, subject=None):
    return {(c.subject, c.prop): c for c in extract(text, subject=subject).claims}


def test_key_then_value_and_value_then_key():
    found = claims("Ibuprofen has a molecular weight of 206.28 g/mol, 2 hydrogen bond donors and 4 acceptors.")
    assert found[("ibuprofen", "mw")].value == 206.28
    assert found[("ibuprofen", "hbd")].value == 2
    assert found[("ibuprofen", "hba")].value == 4  # not attached to "donors"


def test_rule_thresholds_are_not_claims():
    text = "It passes the Rule of Five (MW under 500, logP below 5, at most 5 donors and no more than 10 acceptors)."
    found = extract(text, subject="ibuprofen")
    assert [c.prop for c in found.claims] == ["lipinski_pass"]
    assert found.unattributed == []


def test_logp_method_comes_from_the_name_or_a_parenthesis():
    found = extract("Aspirin has an XLogP3 of 1.2. Its logP is 1.31 (Crippen). A cLogP of 1.3 is typical.").claims
    assert [(c.value, c.method) for c in found] == [(1.2, "xlogp3"), (1.31, "crippen"), (1.3, None)]


def test_unicode_minus_and_word_numbers():
    found = claims("Metformin has a logP of −1.24, no rotatable bonds and three heavy atoms.")
    assert found[("metformin", "logp")].value == -1.24
    assert found[("metformin", "rotb")].value == 0
    assert found[("metformin", "heavy_atoms")].value == 3


def test_subject_is_the_closest_preceding_mention():
    text = "Imatinib (MW 493.6) is larger than gefitinib (MW 446.9)."
    found = claims(text)
    assert found[("imatinib", "mw")].value == 493.6
    assert found[("gefitinib", "mw")].value == 446.9


def test_pass_and_fail_and_partial_violations():
    assert claims("Cyclosporine fails Lipinski's rule of five.")[("cyclosporine", "lipinski_pass")].value == 0
    assert claims("Aspirin is Lipinski-compliant.")[("aspirin", "lipinski_pass")].value == 1
    assert claims("Aspirin does not pass the Ro5.")[("aspirin", "lipinski_pass")].value == 0
    found = claims("Atorvastatin violates one of Lipinski's rules.")
    assert found[("atorvastatin", "lipinski_violations")].value == 1
    assert ("atorvastatin", "lipinski_pass") not in found


def test_prediction_with_and_without_interval():
    with_pm = claims("The predicted log solubility is -3.1 ± 0.6 log units.", subject="aspirin")
    assert with_pm[("aspirin", "log_solubility")].interval == (-3.7, -2.5)
    with_range = claims("Predicted logS of -2.4 (95% interval -3.0 to -1.8).", subject="aspirin")
    assert with_range[("aspirin", "log_solubility")].interval == (-3.0, -1.8)
    bare = claims("Its aqueous solubility is -2.4.", subject="aspirin")
    assert bare[("aspirin", "log_solubility")].interval is None


def test_omics_claims_attach_to_the_gene_in_the_same_sentence():
    text = "In total 37 genes were differentially expressed. G0042 had a log2 fold change of 2.31 and padj = 3.2e-05."
    found = claims(text)
    assert found[("dataset", "n_significant")].value == 37
    assert found[("G0042", "log2fc")].value == 2.31
    assert found[("G0042", "padj")].value == 3.2e-05


def test_upper_bound_p_value_is_an_inequality_claim():
    found = claims("G0007 was significant (FDR < 1e-4).")
    assert found[("G0007", "padj")].op == "<"


def test_numbers_no_claim_covers_are_reported():
    found = extract("Aspirin has a half-life of 15 minutes and 70% oral bioavailability.", subject="aspirin")
    assert found.claims == []
    assert [number for _, _, number in found.unattributed] == ["15", "70"]


def test_identifiers_and_list_markers_are_not_numbers():
    found = extract("1. Aspirin inhibits COX-1 and CYP2C9 weakly; formula C9H8O4.\n2. Rule of 5 applies.")
    assert found.unattributed == []


# Regression cases: sentences llama3.1:8b actually wrote in the E5 pilot.

def test_long_name_then_abbreviation_falls_through_to_the_value():
    found = claims("The topological polar surface area (TPSA) is 83.5 Å², which is relatively high.", subject="ibuprofen")
    assert found[("ibuprofen", "tpsa")].value == 83.5


def test_two_logp_values_each_with_its_method():
    found = extract("Its logP values are 3.07 (Crippen) and 3.5 (XlogP3), indicating moderate lipophilicity.",
                    subject="ibuprofen")
    assert [(c.value, c.method) for c in found.claims] == [(3.07, "crippen"), (3.5, "xlogp3")]
    assert found.unattributed == []


def test_methods_in_parentheses_after_counts_and_weights():
    found = claims("It has a molecular weight of 206.28 (RDKit) and two hydrogen-bond acceptors (Lipinski, PubChem).",
                   subject="ibuprofen")
    assert found[("ibuprofen", "mw")].method == "rdkit"
    assert found[("ibuprofen", "hba")].method == "lipinski"


def test_a_method_word_that_does_not_fit_the_property_is_ignored():
    found = claims("Its molecular weight is 206.28 (well inside the Lipinski limit).", subject="ibuprofen")
    assert found[("ibuprofen", "mw")].method is None


def test_short_pass_form_only_in_a_sentence_that_names_the_rule():
    assert claims("According to Lipinski's Rule of Five, ibuprofen passes the rule.")[("ibuprofen", "lipinski_pass")].value == 1
    assert ("ibuprofen", "lipinski_pass") not in claims("Ibuprofen passes the test.")
