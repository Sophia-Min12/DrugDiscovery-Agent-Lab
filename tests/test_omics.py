"""RNA-seq analysis, tested against hand-computed answers and the generator's planted truth."""
import numpy as np
import pytest

from drugagent import omics
from drugagent.claims import Claim, Verdict
from drugagent.extract import extract
from drugagent.verifier import Verifier


def test_benjamini_hochberg_by_hand():
    # sorted p = .005 .01 .03 .04 ; times n/i = .02 .02 .04 .04 ; already monotone
    p = np.array([0.01, 0.04, 0.03, 0.005])
    assert omics.benjamini_hochberg(p) == pytest.approx([0.02, 0.04, 0.04, 0.02])


def test_benjamini_hochberg_is_monotone_and_never_below_p():
    p = np.random.default_rng(0).uniform(size=500)
    adjusted = omics.benjamini_hochberg(p)
    order = np.argsort(p)
    assert np.all(adjusted >= p) and np.all(np.diff(adjusted[order]) >= -1e-12) and adjusted.max() <= 1


def test_simulation_is_seeded_and_plants_what_it_says():
    a, b = omics.simulate(seed=3), omics.simulate(seed=3)
    assert np.array_equal(a.counts, b.counts)
    assert a.is_de.sum() == 100 and np.all(a.true_log2fc[~a.is_de] == 0)
    assert np.all(np.abs(a.true_log2fc[a.is_de]) >= 0.8)


def test_strong_effects_with_many_replicates_are_found_with_the_right_sign():
    ds = omics.simulate(n_per_group=10, seed=5)
    result = omics.differential_expression(ds)
    strong = ds.is_de & (np.abs(ds.true_log2fc) > 2) & result.tested
    assert result.significant[strong].mean() > 0.9
    assert np.all(np.sign(result.log2fc[strong & result.significant]) == np.sign(ds.true_log2fc[strong & result.significant]))


def test_on_null_data_discoveries_are_rare():
    # No gene is differentially expressed, so every discovery is false. BH at 5%
    # bounds the chance of *any* discovery near 5% when every null is true.
    with_any = sum(omics.differential_expression(omics.simulate(frac_de=0.0, seed=s)).significant.any()
                   for s in range(20))
    assert with_any <= 3


def test_named_datasets_are_stable():
    assert omics.summary("demo")["n_significant"] == omics.summary("demo")["n_significant"]
    with pytest.raises(ValueError):
        omics.dataset("nope")


def test_verifier_reruns_the_analysis():
    top = omics.summary("demo")["top_genes"][0]
    v = Verifier([omics.omics_source_for("demo")])
    assert v.check(Claim(top["gene"], "log2fc", top["log2fc"])).verdict is Verdict.SUPPORTED
    assert v.check(Claim(top["gene"], "log2fc", top["log2fc"] + 1.0)).verdict is Verdict.CONTRADICTED
    assert v.check(Claim(top["gene"], "padj", top["padj"] * 1.1)).verdict is Verdict.SUPPORTED   # log-scale
    assert v.check(Claim(top["gene"], "padj", top["padj"] * 100)).verdict is Verdict.CONTRADICTED
    assert v.check(Claim("G9999", "log2fc", 1.0)).verdict is Verdict.UNVERIFIABLE


def test_a_summary_written_from_the_tool_verifies_end_to_end():
    s = omics.summary("demo")
    g = s["top_genes"][0]
    text = (f"In total {s['n_significant']} genes were differentially expressed. "
            f"{g['gene']} had a log2 fold change of {g['log2fc']} and padj = {g['padj']}.")
    found = extract(text, dataset="demo")
    checks = Verifier([omics.omics_source_for("demo")]).check_all(found.claims)
    assert len(checks) == 3 and all(c.verdict is Verdict.SUPPORTED for c in checks)
