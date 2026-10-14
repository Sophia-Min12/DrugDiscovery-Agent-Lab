"""E3 - does "FDR < 0.05" mean 5% of the discoveries are false?

Simulated RNA-seq with 100 of 2,000 genes truly changed, 20 seeds per design, three
analyses of the same counts:

    cpm_welch_bh           library-size (CPM) normalisation, Welch t-test, Benjamini-Hochberg
    median_ratio_welch_bh  DESeq2 median-of-ratios normalisation, same test
    fold_change_only       |log2 FC| >= 1 after median-of-ratios, no statistics at all

    python experiments/e3_omics_fdr.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drugagent import figures, omics  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results" / "e3_omics.json"
REPLICATES = [3, 4, 6, 10]
SEEDS = range(20)


def fold_change_only(ds: omics.Dataset) -> dict:
    result = omics.differential_expression(ds)
    called = result.tested & (np.abs(result.log2fc) >= 1.0)
    discoveries = int(called.sum())
    return {"sensitivity": float((called & ds.is_de).sum() / ds.is_de.sum()),
            "false_discovery_proportion": float((called & ~ds.is_de).sum() / discoveries) if discoveries else 0.0}


def main() -> None:
    analyses = {
        "cpm_welch_bh": lambda ds: omics.evaluate(ds, omics.differential_expression(ds, normalization="library")),
        "median_ratio_welch_bh": lambda ds: omics.evaluate(ds, omics.differential_expression(ds)),
        "fold_change_only": fold_change_only,
    }
    table = {name: {} for name in analyses}
    for n in REPLICATES:
        runs = {name: [] for name in analyses}
        for seed in SEEDS:
            ds = omics.simulate(n_per_group=n, seed=1000 + seed)
            for name, analyse in analyses.items():
                runs[name].append(analyse(ds))
        for name, rows in runs.items():
            fdp = np.array([r["false_discovery_proportion"] for r in rows])
            table[name][n] = {
                "mean_sensitivity": round(float(np.mean([r["sensitivity"] for r in rows])), 3),
                "mean_fdp": round(float(fdp.mean()), 3),
                "seeds_with_fdp_over_10pct": int((fdp > 0.10).sum()),
            }
    out = {"design": {"genes": 2000, "planted_de": 100, "seeds": len(SEEDS), "nominal_fdr": omics.FDR},
           "results": table}
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))

    labels = {"cpm_welch_bh": "CPM + Welch + BH", "median_ratio_welch_bh": "median-of-ratios + Welch + BH",
              "fold_change_only": "|log2 FC| >= 1, no test"}
    fig, (left, right) = figures.figure(10.0, 4.0, ncols=2)
    figures.style(left, "Sensitivity: planted genes found", "replicates per group", "mean sensitivity")
    figures.style(right, "False-discovery proportion", "replicates per group", "mean FDP")
    for k, name in enumerate(analyses):
        colour = figures.SERIES[k]
        sens = [table[name][n]["mean_sensitivity"] for n in REPLICATES]
        fdp = [table[name][n]["mean_fdp"] for n in REPLICATES]
        left.plot(REPLICATES, sens, color=colour, linewidth=2, marker="o", markersize=5, label=labels[name])
        right.plot(REPLICATES, fdp, color=colour, linewidth=2, marker="o", markersize=5, label=labels[name])
    right.axhline(omics.FDR, color=figures.INK_3, linewidth=0.8, linestyle=(0, (4, 3)))
    right.text(REPLICATES[-1], omics.FDR, "nominal 5% ", color=figures.INK_2, fontsize=8, ha="right", va="bottom")
    for ax in (left, right):
        ax.set_xticks(REPLICATES)
        ax.set_ylim(bottom=0)
    legend = left.legend(frameon=False, fontsize=8.5, loc="lower right")
    for text in legend.get_texts():
        text.set_color(figures.INK_2)
    figures.save(fig, "e3_fdr_by_replicates")

    figures.volcano(omics.analysed("demo"), "volcano_demo",
                    "Dataset 'demo': 4 vs 4 replicates, median-of-ratios + Welch + BH")


if __name__ == "__main__":
    main()
