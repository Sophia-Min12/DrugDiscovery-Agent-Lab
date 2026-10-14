"""The approved-drug reference table and the name -> structure resolver.

data/approved_drugs.csv is fetched from PubChem by scripts/fetch_pubchem.py. Its
property columns are PubChem's own computations, kept as a second, independent
method next to RDKit's: `reference_values("ibuprofen", "logp")` returns
{"xlogp3": 3.5}, while chem.compute_all_methods returns {"crippen": 3.07}.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from . import chem

DATA = Path(__file__).resolve().parents[1] / "data" / "approved_drugs.csv"

# Common alternative names an LLM (or a chemist) might use.
SYNONYMS = {
    "paracetamol": "acetaminophen",
    "albuterol": "salbutamol",
    "ciclosporin": "cyclosporine",
    "cyclosporin": "cyclosporine",
    "rapamycin": "sirolimus",
    "frusemide": "furosemide",
    "acetylsalicylic acid": "aspirin",
}

# reference column -> (property, method name)
_REFERENCE_COLUMNS = {
    "pubchem_mw": ("mw", "pubchem"),
    "pubchem_xlogp3": ("logp", "xlogp3"),
    "pubchem_tpsa": ("tpsa", "pubchem"),
    "pubchem_hbd": ("hbd", "pubchem"),
    "pubchem_hba": ("hba", "pubchem"),
    "pubchem_rotb": ("rotb", "pubchem"),
}


@dataclass(frozen=True)
class Drug:
    name: str
    cid: int
    smiles: str
    formula: str
    reference: dict  # (prop, method) -> value


@lru_cache(maxsize=1)
def table() -> dict[str, Drug]:
    drugs = {}
    with DATA.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            reference = {}
            for column, key in _REFERENCE_COLUMNS.items():
                if row[column] != "":
                    value = float(row[column])
                    reference[key] = int(value) if key[0] in chem.COUNT_PROPERTIES else value
            drugs[row["name"]] = Drug(row["name"], int(row["cid"]), row["smiles"], row["formula"], reference)
    return drugs


def names() -> list[str]:
    return sorted(table())


def lookup(subject: str) -> Drug | None:
    key = subject.strip().lower()
    key = SYNONYMS.get(key, key)
    return table().get(key)


def resolve(subject: str) -> str | None:
    """A drug name or a SMILES string -> SMILES, or None if it is neither."""
    drug = lookup(subject)
    if drug is not None:
        return drug.smiles
    return subject if chem.is_valid_smiles(subject) else None


def reference_values(subject: str, prop: str) -> dict[str, float]:
    """PubChem's value(s) for a property, keyed by method. Empty if not in the table."""
    drug = lookup(subject)
    if drug is None:
        return {}
    return {method: value for (p, method), value in drug.reference.items() if p == prop}


def all_methods(subject: str, prop: str) -> dict[str, float]:
    """Every value this repo can produce for (subject, prop): RDKit methods + PubChem."""
    smiles = resolve(subject)
    if smiles is None or prop not in chem.PROPERTY_METHODS:
        return {}
    return {**chem.compute_all_methods(prop, smiles), **reference_values(subject, prop)}


def library() -> dict[str, str]:
    return {name: drug.smiles for name, drug in table().items()}
