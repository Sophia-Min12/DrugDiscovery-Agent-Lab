"""Bulk RNA-seq differential expression, on counts whose truth is planted.

The counts are simulated, not measured: a negative-binomial (gamma-Poisson) model
with a mean-dispersion trend, library-size variation, and a known set of genes whose
fold change was set by the generator. Simulation is the only way to know which
discoveries are false, which is what lets experiment E3 measure the false-discovery
proportion instead of quoting the nominal 5%.

The analysis is a Welch t-test per gene on log2 normalised counts, then
Benjamini-Hochberg - not DESeq2 or limma-voom. Normalisation is the part that
matters: plain CPM (divide by library size), which is what a quickly written script
does, lets a few strongly induced genes eat library share and pushes every other
gene down. Experiment E3 measures that at 32% false discoveries for a nominal 5%,
and DESeq2's median-of-ratios size factors as the fix. The default is the fix.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import stats

from .claims import Claim, Evidence

FDR = 0.05

# Named datasets the agent can analyse. Same name -> same counts, always.
DATASETS = {
    "demo": dict(seed=7, n_per_group=4),
    "pilot_n3": dict(seed=11, n_per_group=3),
    "large_n10": dict(seed=13, n_per_group=10),
}


@dataclass
class Dataset:
    name: str
    counts: np.ndarray      # [genes, samples], integer
    genes: list[str]
    group: np.ndarray       # [samples]: 0 = control, 1 = treated
    true_log2fc: np.ndarray  # [genes]: 0 for genes the generator left alone
    is_de: np.ndarray       # [genes] bool


def simulate(name: str = "custom", n_genes: int = 2000, n_per_group: int = 4, frac_de: float = 0.05,
             seed: int = 0) -> Dataset:
    rng = np.random.default_rng(seed)
    base = np.exp(rng.normal(np.log(80), 1.4, n_genes))  # baseline mean counts, log-normal
    dispersion = 0.04 + 1.5 / base                      # small genes are noisier, as in real data
    n_de = int(round(frac_de * n_genes))
    is_de = np.zeros(n_genes, dtype=bool)
    is_de[rng.choice(n_genes, n_de, replace=False)] = True
    log2fc = np.zeros(n_genes)
    log2fc[is_de] = rng.choice([-1, 1], n_de) * rng.uniform(0.8, 3.0, n_de)
    group = np.repeat([0, 1], n_per_group)
    size_factor = rng.uniform(0.7, 1.3, group.size)
    mean = base[:, None] * size_factor[None, :] * 2.0 ** (log2fc[:, None] * group[None, :])
    shape = 1.0 / dispersion[:, None]
    counts = rng.poisson(rng.gamma(shape, mean / shape))
    genes = [f"G{i:04d}" for i in range(1, n_genes + 1)]
    return Dataset(name, counts, genes, group, log2fc, is_de)


def dataset(name: str) -> Dataset:
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; available: {', '.join(DATASETS)}")
    return simulate(name, **DATASETS[name])


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """Adjusted p-values: p_(i) * n / i, made monotone from the largest p down, capped at 1."""
    p = np.asarray(p, dtype=float)
    n = p.size
    order = np.argsort(p)
    scaled = p[order] * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(scaled[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adjusted, 1.0)
    return out


@dataclass
class DEResult:
    genes: list[str]
    tested: np.ndarray       # [genes] bool: passed the expression filter
    log2fc: np.ndarray
    mean_log2cpm: np.ndarray
    pvalue: np.ndarray
    padj: np.ndarray
    fdr: float

    @property
    def significant(self) -> np.ndarray:
        return self.tested & (self.padj < self.fdr)

    def row(self, gene: str) -> dict | None:
        if gene not in self.genes:
            return None
        i = self.genes.index(gene)
        if not self.tested[i]:
            return {"gene": gene, "tested": False}
        return {"gene": gene, "tested": True, "log2fc": float(self.log2fc[i]), "pvalue": float(self.pvalue[i]),
                "padj": float(self.padj[i]), "significant": bool(self.significant[i])}


def size_factors(counts: np.ndarray, method: str) -> np.ndarray:
    """Per-sample scaling, relative to the mean library size.

    library      - total counts (what CPM uses)
    median_ratio - DESeq2: median over genes of count / per-gene geometric mean,
                   using only genes with no zero count. Robust to a minority of
                   strongly changed genes, because a median ignores them.
    """
    library = counts.sum(axis=0).astype(float)
    if method == "library":
        return library / library.mean()
    if method == "median_ratio":
        expressed = counts[np.all(counts > 0, axis=1)].astype(float)
        log = np.log(expressed)
        factors = np.exp(np.median(log - log.mean(axis=1, keepdims=True), axis=0))
        return factors / factors.mean()
    raise ValueError(f"unknown normalisation {method!r}")


def differential_expression(ds: Dataset, fdr: float = FDR, min_cpm: float = 1.0,
                            normalization: str = "median_ratio") -> DEResult:
    scale = size_factors(ds.counts, normalization) * ds.counts.sum(axis=0).mean()
    cpm = ds.counts / scale[None, :] * 1e6
    logcpm = np.log2(cpm + 0.5)
    tested = cpm.mean(axis=1) >= min_cpm
    control, treated = logcpm[:, ds.group == 0], logcpm[:, ds.group == 1]
    log2fc = treated.mean(axis=1) - control.mean(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # constant genes: t is undefined, p set to 1 below
        pvalue = stats.ttest_ind(treated, control, axis=1, equal_var=False).pvalue
    pvalue = np.where(np.isnan(pvalue), 1.0, pvalue)
    padj = np.ones(len(ds.genes))
    padj[tested] = benjamini_hochberg(pvalue[tested])  # only tested genes enter the correction
    return DEResult(ds.genes, tested, log2fc, logcpm.mean(axis=1), pvalue, padj, fdr)


def evaluate(ds: Dataset, result: DEResult) -> dict:
    called = result.significant
    true_positive = called & ds.is_de
    discoveries = int(called.sum())
    return {
        "planted": int(ds.is_de.sum()),
        "discoveries": discoveries,
        "sensitivity": float(true_positive.sum() / ds.is_de.sum()),
        "false_discovery_proportion": float((called & ~ds.is_de).sum() / discoveries) if discoveries else 0.0,
        "sign_errors": int((true_positive & (np.sign(result.log2fc) != np.sign(ds.true_log2fc))).sum()),
    }


def summary(name: str = "demo", fdr: float = FDR, top: int = 10) -> dict:
    """What the agent's tool returns: counts, the top genes, and the method that produced them."""
    result = analysed(name, fdr)
    sig = np.where(result.significant)[0]
    ranked = sig[np.argsort(result.padj[sig])][:top]
    return {
        "dataset": name,
        "method": "median-of-ratios normalisation, log2, Welch t-test per gene, Benjamini-Hochberg",
        "fdr": fdr,
        "n_genes_tested": int(result.tested.sum()),
        "n_significant": int(sig.size),
        "n_up": int((result.log2fc[sig] > 0).sum()),
        "n_down": int((result.log2fc[sig] < 0).sum()),
        "top_genes": [{"gene": result.genes[i], "log2fc": round(float(result.log2fc[i]), 2),
                       "padj": float(f"{result.padj[i]:.3g}")} for i in ranked],
    }


@lru_cache(maxsize=16)
def analysed(name: str, fdr: float = FDR) -> DEResult:
    """The analysis of a named dataset - deterministic, so computed once."""
    return differential_expression(dataset(name), fdr)


def omics_source_for(name: str = "demo"):
    """A verifier source that re-runs the analysis of dataset `name` and reads the claim's gene."""
    def source(claim: Claim) -> Evidence | None:
        if claim.prop == "n_significant":
            target = claim.subject if claim.subject in DATASETS else name
            result = analysed(target)
            return Evidence({f"fdr_{FDR}": float(result.significant.sum())})
        if claim.prop not in ("log2fc", "padj"):
            return None
        row = analysed(name).row(claim.subject)
        if row is None or not row["tested"]:
            return None
        return Evidence({"welch_bh": row[claim.prop]})
    return source


omics_source = omics_source_for("demo")
