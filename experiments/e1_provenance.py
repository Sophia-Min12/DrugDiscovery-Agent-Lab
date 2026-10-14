"""E1 - is "the logP of X" one number?

For the 56 approved drugs in the reference table, compare every method this repo can
compute or look up (RDKit vs PubChem's programs) for each property, and count how
often the Rule-of-Five verdict depends on which methods were used.

    python experiments/e1_provenance.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drugagent import drugs, figures  # noqa: E402
from drugagent.claims import tolerance  # noqa: E402
from drugagent.verifier import lipinski_variants  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results" / "e1_provenance.json"
PROPS = ["mw", "logp", "tpsa", "hbd", "hba", "rotb"]


def main() -> None:
    names = drugs.names()
    out = {"n_drugs": len(names), "properties": {}, "lipinski": {}}
    for prop in PROPS:
        spreads, disagree, worst = [], 0, ("", 0.0, {})
        for name in names:
            values = drugs.all_methods(name, prop)
            spread = max(values.values()) - min(values.values())
            spreads.append(spread)
            disagree += spread > tolerance(prop) + 1e-9
            if spread > worst[1]:
                worst = (name, spread, {m: round(v, 2) for m, v in values.items()})
        out["properties"][prop] = {
            "methods": list(drugs.all_methods(names[0], prop)),
            "tolerance": tolerance(prop),
            "drugs_where_methods_disagree_beyond_tolerance": disagree,
            "median_spread": round(float(np.median(spreads)), 3),
            "max_spread": round(float(max(spreads)), 3),
            "worst": {"drug": worst[0], "values": worst[2]},
        }

    flips = {}
    for name in names:
        variants = lipinski_variants(name)
        verdicts = {v: n <= 1 for v, n in variants.items()}
        if len(set(verdicts.values())) > 1:
            flips[name] = variants
    out["lipinski"] = {"verdict_depends_on_method": len(flips), "drugs": flips}
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))

    # logP: Crippen (RDKit) against XLogP3 (PubChem), drug by drug.
    pairs = [(n, drugs.all_methods(n, "logp")) for n in names]
    pairs = [(n, v["crippen"], v["xlogp3"]) for n, v in pairs if "xlogp3" in v]
    x = np.array([p[1] for p in pairs])
    y = np.array([p[2] for p in pairs])
    fig, ax = figures.figure(5.6, 5.0)
    figures.style(ax, "logP of 56 approved drugs, two standard methods", "Crippen logP (RDKit)", "XLogP3 (PubChem)")
    lo, hi = min(x.min(), y.min()) - 0.9, max(x.max(), y.max()) + 0.6
    ax.fill_between([lo, hi], [lo - 0.3, hi - 0.3], [lo + 0.3, hi + 0.3], color=figures.GRID, linewidth=0,
                    label="within the 0.3 tolerance")
    ax.plot([lo, hi], [lo, hi], color=figures.INK_3, linewidth=0.8)
    outside = np.abs(x - y) > 0.3
    ax.scatter(x[~outside], y[~outside], s=22, color=figures.SERIES[0], edgecolors=figures.SURFACE, linewidths=0.6,
               label=f"agree ({(~outside).sum()})")
    ax.scatter(x[outside], y[outside], s=22, color=figures.SERIES[1], edgecolors=figures.SURFACE, linewidths=0.6,
               label=f"disagree by > 0.3 ({outside.sum()})")
    ax.axvline(5, color=figures.INK_3, linewidth=0.8, linestyle=(0, (4, 3)))
    ax.axhline(5, color=figures.INK_3, linewidth=0.8, linestyle=(0, (4, 3)))
    ax.text(5, lo + 0.2, " Ro5 limit", color=figures.INK_2, fontsize=8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    worst = np.argsort(-np.abs(x - y))[:4]
    flipped = [i for i, p in enumerate(pairs) if p[0] in flips]
    figures.label_points(ax, [(x[i], y[i], pairs[i][0]) for i in sorted(set(worst) | set(flipped))])
    legend = ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    for text in legend.get_texts():
        text.set_color(figures.INK_2)
    figures.save(fig, "e1_logp_methods")


if __name__ == "__main__":
    main()
