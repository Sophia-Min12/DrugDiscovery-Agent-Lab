"""ESOL (Delaney 2004): 1,128 molecules with measured aqueous solubility, log10(mol/L).

Downloaded on first use from the MoleculeNet mirror and checked against a pinned
SHA-256, so every number in the experiments refers to the same file. The CSV is
cached under data/cache/ and not committed.
"""
from __future__ import annotations

import csv
import hashlib
import random
import urllib.request
from collections import defaultdict
from pathlib import Path

from .. import chem

URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
SHA256 = "8c06a76f0c6487d29ab0f903e6a7a7139f189ab3c1178f159c8be8964602f189"
CACHE = Path(__file__).resolve().parents[2] / "data" / "cache" / "delaney-processed.csv"
TARGET = "measured log solubility in mols per litre"


def download(path: Path = CACHE) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=60) as response:
            path.write_bytes(response.read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256:
        raise RuntimeError(f"{path} does not match the pinned ESOL file (sha256 {digest})")
    return path


def load() -> list[tuple[str, float]]:
    """(canonical SMILES, measured logS) for every molecule RDKit can parse."""
    rows = []
    with download().open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            smiles = row["smiles"].strip()
            if chem.is_valid_smiles(smiles):
                rows.append((chem.canonical(smiles), float(row[TARGET])))
    return rows


def random_split(n: int, seed: int = 0, fractions=(0.8, 0.1, 0.1)) -> tuple[list[int], list[int], list[int]]:
    order = list(range(n))
    random.Random(seed).shuffle(order)
    a, b = int(fractions[0] * n), int((fractions[0] + fractions[1]) * n)
    return order[:a], order[a:b], order[b:]


def scaffold_split(smiles: list[str], fractions=(0.8, 0.1, 0.1)) -> tuple[list[int], list[int], list[int]]:
    """Whole Bemis-Murcko scaffold groups go to one side, largest groups to training.

    Test molecules then share no scaffold with training molecules - closer to the
    question a chemist asks of a model ("what about this new series?") than a random
    split, where a test molecule's sibling is usually in the training set.
    """
    groups = defaultdict(list)
    for i, s in enumerate(smiles):
        groups[chem.murcko_scaffold(s)].append(i)
    ordered = sorted(groups.values(), key=lambda g: (-len(g), g[0]))
    n = len(smiles)
    train, valid, test = [], [], []
    for group in ordered:
        if len(train) + len(group) <= fractions[0] * n:
            train += group
        elif len(valid) + len(group) <= fractions[1] * n:
            valid += group
        else:
            test += group
    return train, valid, test
