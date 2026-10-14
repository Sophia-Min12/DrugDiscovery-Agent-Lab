"""Figures for E2, drawn from experiments/results/e2_gnn.json (no retraining).

    python experiments/e2_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drugagent import figures  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results" / "e2_gnn.json"


def main() -> None:
    report = json.loads(RESULTS.read_text(encoding="utf-8"))
    fig, (left, middle, right) = figures.figure(12.0, 4.0, ncols=3)

    # 1. Test RMSE per model, random vs scaffold split.
    models = [("linear_4_descriptors", "linear,\n4 descriptors"), ("knn_5_tanimoto", "5-NN\nTanimoto"),
              ("gnn_ensemble", "GNN\nensemble")]
    figures.style(left, "Test RMSE (log units)", "", "RMSE")
    width = 0.38
    for k, split in enumerate(("random", "scaffold")):
        values = [report[split]["test"][m]["rmse"] for m, _ in models]
        xs = [i + (k - 0.5) * width for i in range(len(models))]
        left.bar(xs, values, width=width - 0.04, color=figures.SERIES[k], label=f"{split} split")
        for x, v in zip(xs, values):
            left.text(x, v + 0.03, f"{v:.2f}", ha="center", fontsize=8, color=figures.INK_2)
    left.set_xticks(range(len(models)), [label for _, label in models], fontsize=8.5)
    left.set_ylim(0, 1.95)
    left.grid(axis="x", visible=False)

    # 2. Scaffold split: error against similarity to the nearest training molecule.
    bins = report["scaffold"]["error_by_similarity_to_training"]
    figures.style(middle, "Scaffold split: error vs similarity", "max Tanimoto to training set", "GNN RMSE")
    xs = range(len(bins))
    middle.bar(xs, [b["gnn_rmse"] for b in bins], width=0.62, color=figures.SERIES[0])
    for x, b in zip(xs, bins):
        middle.text(x, b["gnn_rmse"] + 0.03, f"{b['gnn_rmse']:.2f}\nn={b['n']}", ha="center", fontsize=8,
                    color=figures.INK_2)
    middle.set_xticks(list(xs), [b["similarity"] for b in bins], fontsize=8.5)
    middle.set_ylim(0, max(b["gnn_rmse"] for b in bins) * 1.3)
    middle.grid(axis="x", visible=False)

    # 3. Interval coverage against the 95% target.
    rows = [("coverage_ensemble_2sigma", "ensemble\n±2σ"), ("coverage_conformal_scaled", "conformal,\nσ-scaled"),
            ("coverage_conformal_constant", "conformal,\nconstant width")]
    figures.style(right, "95% interval coverage on test", "", "share of test molecules inside")
    for k, split in enumerate(("random", "scaffold")):
        values = [report[split]["uncertainty"][key] for key, _ in rows]
        xs = [i + (k - 0.5) * width for i in range(len(rows))]
        right.bar(xs, values, width=width - 0.04, color=figures.SERIES[k], label=f"{split} split")
        for x, v in zip(xs, values):
            right.text(x, v + 0.015, f"{v:.0%}", ha="center", fontsize=8, color=figures.INK_2)
    right.axhline(0.95, color=figures.INK_3, linewidth=0.8, linestyle=(0, (4, 3)))
    right.text(-0.45, 0.955, "target 95%", ha="left", va="bottom", fontsize=8, color=figures.INK_2)
    right.set_xticks(range(len(rows)), [label for _, label in rows], fontsize=8.5)
    right.set_ylim(0, 1.3)
    right.grid(axis="x", visible=False)
    for ax in (left, right):
        legend = ax.legend(frameon=False, fontsize=8.5, loc="upper left")
        for text in legend.get_texts():
            text.set_color(figures.INK_2)
    figures.save(fig, "e2_gnn")


if __name__ == "__main__":
    main()
