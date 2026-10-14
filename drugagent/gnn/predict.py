"""The deployed solubility model, as a tool and as a verifier source.

A prediction is returned with three things beside the number: a 95% interval
(ensemble spread scaled by a split-conformal factor calibrated on unseen scaffolds),
the similarity of the nearest training molecule, and whether that similarity clears
the applicability-domain threshold. The verifier refuses a solubility claim that
drops the interval (UNQUALIFIED) and rejects one outside it (CONTRADICTED).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from rdkit import DataStructs

from .. import chem, drugs
from ..claims import Claim, Evidence
from .model import SolubilityGNN, collate, featurize

MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "esol_gnn"


@dataclass
class Deployed:
    models: list[tuple[SolubilityGNN, float, float]]
    conformal_q: float
    threshold: float
    train_smiles: list[str]
    train_fps: list


@lru_cache(maxsize=2)
def load(model_dir: Path = MODEL_DIR) -> Deployed | None:
    meta_path = model_dir / "meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    models = []
    for k, (mean, std) in enumerate(meta["scalers"]):
        model = SolubilityGNN(meta["hidden"], meta["layers"])
        model.load_state_dict(torch.load(model_dir / f"model_{k}.pt", weights_only=True))
        model.eval()
        models.append((model, mean, std))
    train = meta["train_smiles"]
    return Deployed(models, meta["conformal_q"], meta["domain_threshold"], train, [chem.fingerprint(s) for s in train])


def predict(compound: str, model_dir: Path = MODEL_DIR) -> dict:
    deployed = load(model_dir)
    if deployed is None:
        raise RuntimeError("no trained model; run `python -m drugagent.gnn.train`")
    smiles = drugs.resolve(compound)
    if smiles is None:
        raise ValueError(f"cannot resolve {compound!r} to a structure")
    batch = collate([featurize(smiles)])
    with torch.no_grad():
        preds = np.array([model(batch).item() * std + mean for model, mean, std in deployed.models])
    mu, sigma = float(preds.mean()), max(float(preds.std()), 1e-3)
    half = deployed.conformal_q * sigma
    sims = DataStructs.BulkTanimotoSimilarity(chem.fingerprint(smiles), deployed.train_fps)
    best = int(np.argmax(sims))
    return {
        "compound": compound,
        "log_solubility": mu,
        "interval_95": [mu - half, mu + half],
        "units": "log10(mol/L)",
        "nearest_training_similarity": float(sims[best]),
        "nearest_training_smiles": deployed.train_smiles[best],
        "in_domain": bool(sims[best] >= deployed.threshold),
        "method": "GNN, 5-model ensemble; 95% split-conformal interval calibrated on unseen scaffolds",
    }


def gnn_source(claim: Claim) -> Evidence | None:
    if claim.prop != "log_solubility" or load() is None or drugs.resolve(claim.subject) is None:
        return None
    p = predict(claim.subject)
    domain = "inside" if p["in_domain"] else "OUTSIDE"
    note = f"{domain} the applicability domain (nearest training similarity {p['nearest_training_similarity']:.2f})"
    return Evidence({"gnn": p["log_solubility"]}, interval=tuple(p["interval_95"]), note=note)
