"""E4 - who checks the checker? Plant errors of known size and count what is caught.

For every drug, sentences are written from the tool values in four phrasings - three
that name the method of each value, one that does not - so the truth of every claim
is known. Then exactly one value per sentence is perturbed by a multiple of that
property's tolerance (or by +/-1, +/-2 for counts) and the text goes through the same
extractor and verifier the agent uses.

Measured: extraction recall, how often a planted error is caught (CONTRADICTED) as a
function of its size, and what the verifier says about the claims left correct.

    python experiments/e4_planted_errors.py
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drugagent import drugs  # noqa: E402
from drugagent.claims import tolerance  # noqa: E402
from drugagent.extract import extract  # noqa: E402
from drugagent.verifier import Verifier  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results" / "e4_planted.json"

TEMPLATES = {
    "prose_named": ("{name} has a molecular weight of {mw} g/mol, a logP of {logp} (Crippen), a TPSA of {tpsa} Å² "
                    "(Ertl), {hbd} hydrogen bond donors (Lipinski), {hba} hydrogen bond acceptors (Lipinski) and "
                    "{rotb} rotatable bonds (RDKit)."),
    "list_named": ("{name}. MW: {mw} (RDKit); XLogP3: {xlogp}; TPSA: {tpsa_pc} (PubChem); HBD: {hbd_pc} (PubChem); "
                   "HBA: {hba_pc} (PubChem); rotatable bonds: {rotb_pc} (PubChem)."),
    "inverted_named": ("With {hbd_rd} H-bond donors (RDKit), {hba_rd} H-bond acceptors (RDKit) and a molecular "
                       "weight of {mw}, {name} has a Crippen logP of {logp} and a polar surface area of {tpsa} (Ertl)."),
    "prose_unnamed": ("{name} has a molecular weight of {mw}, a logP of {logp}, a TPSA of {tpsa}, {hbd} hydrogen bond "
                      "donors, {hba} hydrogen bond acceptors and {rotb} rotatable bonds."),
}
# template field -> (property, decimals written)
FIELDS = {"mw": ("mw", 2), "logp": ("logp", 2), "xlogp": ("logp", 1), "tpsa": ("tpsa", 1), "tpsa_pc": ("tpsa", 1),
          "hbd": ("hbd", 0), "hbd_pc": ("hbd", 0), "hbd_rd": ("hbd", 0), "hba": ("hba", 0), "hba_pc": ("hba", 0),
          "hba_rd": ("hba", 0), "rotb": ("rotb", 0), "rotb_pc": ("rotb", 0)}
CONTINUOUS_SIZES = [0.5, 2, 5, 20]  # multiples of the property's tolerance
COUNT_SIZES = [1, 2]


def true_values(name: str) -> dict:
    v = {p: drugs.all_methods(name, p) for p in ("mw", "logp", "tpsa", "hbd", "hba", "rotb")}
    return {"name": name.capitalize(), "mw": v["mw"]["rdkit"], "logp": v["logp"]["crippen"],
            "xlogp": v["logp"].get("xlogp3"), "tpsa": v["tpsa"]["ertl"], "tpsa_pc": v["tpsa"].get("pubchem"),
            "hbd": v["hbd"]["lipinski"], "hbd_pc": v["hbd"].get("pubchem"), "hbd_rd": v["hbd"]["rdkit"],
            "hba": v["hba"]["lipinski"], "hba_pc": v["hba"].get("pubchem"), "hba_rd": v["hba"]["rdkit"],
            "rotb": v["rotb"]["rdkit"], "rotb_pc": v["rotb"].get("pubchem")}


def fmt(value, digits: int) -> str:
    return str(int(value)) if digits == 0 else f"{value:.{digits}f}"


def main() -> None:
    rng = random.Random(0)
    verifier = Verifier()
    caught = defaultdict(Counter)       # (kind, size) -> verdict counts for the planted claim
    untouched = defaultdict(Counter)    # template -> verdict counts for the correct claims
    written = found = 0
    for name in drugs.names():
        truth = true_values(name)
        for template_name, template in TEMPLATES.items():
            fields = [f for f in FIELDS if "{" + f + "}" in template]
            if any(truth[f] is None for f in fields):
                continue  # PubChem lacks a value this template needs
            target = rng.choice(fields)
            prop = FIELDS[target][0]
            if FIELDS[target][1] == 0:
                size = rng.choice(COUNT_SIZES)
                delta = size * rng.choice([-1, 1]) if truth[target] >= size else size
                kind = "count"
            else:
                size = rng.choice(CONTINUOUS_SIZES)
                delta = size * tolerance(prop) * rng.choice([-1, 1])
                kind = "continuous"
            values = {f: fmt(truth[f], FIELDS[f][1]) for f in fields}
            values[target] = fmt(truth[target] + delta, FIELDS[target][1])
            text = template.format(name=truth["name"], **values)
            extraction = extract(text, subject=name)
            written += len(fields)
            found += len(extraction.claims)
            planted_value = float(values[target])
            for check in verifier.check_all(extraction.claims):
                is_planted = check.claim.prop == prop and abs(check.claim.value - planted_value) < 1e-9
                if is_planted:
                    caught[(kind, size)][check.verdict.value] += 1
                else:
                    untouched[template_name][check.verdict.value] += 1

    out = {
        "sentences": sum(sum(c.values()) for c in caught.values()),
        "extraction_recall": round(found / written, 3),
        "planted_error_verdicts": {f"{kind} x{size}": dict(counts) for (kind, size), counts in sorted(caught.items())},
        "detection_rate": {f"{kind} x{size}": round(counts["contradicted"] / sum(counts.values()), 3)
                           for (kind, size), counts in sorted(caught.items())},
        "correct_claim_verdicts": {t: dict(c) for t, c in untouched.items()},
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
