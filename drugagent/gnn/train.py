"""Train the solubility ensemble, calibrate its intervals, and save the deployed model.

    python -m drugagent.gnn.train          # experiment E2 + models/esol_gnn/

Three questions, each answered on a random split and on a scaffold split:

1. Is the GNN better than a four-descriptor linear model (Delaney's own ESOL equation,
   refitted) and a 5-nearest-neighbour fingerprint baseline?
2. Does the error grow as a molecule gets less similar to anything in training?
3. Does the ensemble's +/-2 sigma interval cover 95% of test molecules, and does a
   split-conformal correction fix it if not?
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from .. import chem
from . import data
from .model import SolubilityGNN, collate, featurize

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "esol_gnn"
RESULTS = ROOT / "experiments" / "results" / "e2_gnn.json"
ENSEMBLE = 5
ALPHA = 0.05  # target miscoverage of the interval
SIMILARITY_BINS = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)]


def fit(graphs, y, train_idx, valid_idx, seed, hidden=64, layers=4, epochs=250, batch_size=32, lr=1e-3,
        patience=40):
    """One model, early-stopped on validation RMSE. Targets are standardised on the training set."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    mean, std = float(np.mean(y[train_idx])), float(np.std(y[train_idx]))
    model = SolubilityGNN(hidden, layers)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    target = torch.tensor((y - mean) / std, dtype=torch.float32)
    valid_batch = collate([graphs[i] for i in valid_idx])
    best, best_state, stale = math.inf, None, 0
    order = list(train_idx)
    for _ in range(epochs):
        model.train()
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            chunk = order[start:start + batch_size]
            loss = torch.mean((model(collate([graphs[i] for i in chunk])) - target[chunk]) ** 2)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            rmse = float(torch.sqrt(torch.mean((model(valid_batch) - target[valid_idx]) ** 2))) * std
        if rmse < best - 1e-4:
            best, best_state, stale = rmse, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, mean, std


def ensemble_predict(models, graphs) -> tuple[np.ndarray, np.ndarray]:
    batch = collate(graphs)
    with torch.no_grad():
        preds = np.stack([model(batch).numpy() * std + mean for model, mean, std in models])
    return preds.mean(axis=0), preds.std(axis=0)


def delaney_features(smiles: str) -> list[float]:
    """Delaney's ESOL equation uses exactly these four: logP, MW, rotatable bonds, aromatic proportion."""
    mol = chem.parse(smiles)
    aromatic = sum(atom.GetIsAromatic() for atom in mol.GetAtoms()) / mol.GetNumHeavyAtoms()
    return [chem.compute("logp", smiles), chem.compute("mw", smiles), chem.compute("rotb", smiles), aromatic, 1.0]


def linear_baseline(X, y, train_idx, test_idx) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(X[train_idx], y[train_idx], rcond=None)
    return X[test_idx] @ coef


def knn_baseline(fps, y, train_idx, test_idx, k=5) -> np.ndarray:
    from rdkit import DataStructs
    train_fps = [fps[i] for i in train_idx]
    out = []
    for i in test_idx:
        sims = np.array(DataStructs.BulkTanimotoSimilarity(fps[i], train_fps))
        top = np.argsort(-sims)[:k]
        weights = sims[top] + 1e-6
        out.append(float(np.sum(weights * y[train_idx][top]) / np.sum(weights)))
    return np.array(out)


def max_similarity(fps, train_idx, idx) -> np.ndarray:
    from rdkit import DataStructs
    train_fps = [fps[i] for i in train_idx]
    return np.array([max(DataStructs.BulkTanimotoSimilarity(fps[i], train_fps)) for i in idx])


def metrics(y_true, y_pred) -> dict:
    err = y_pred - y_true
    ss_res, ss_tot = float(np.sum(err ** 2)), float(np.sum((y_true - y_true.mean()) ** 2))
    return {"rmse": round(float(np.sqrt(np.mean(err ** 2))), 3), "mae": round(float(np.mean(np.abs(err))), 3),
            "r2": round(1 - ss_res / ss_tot, 3)}


def conformal_quantile(scores: np.ndarray, alpha: float = ALPHA) -> float:
    """Split conformal: the ceil((n+1)(1-alpha))-th smallest calibration score."""
    n = len(scores)
    rank = min(n, math.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(scores)[rank - 1])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def evaluate_split(name, smiles, y, graphs, fps, X, split) -> tuple[dict, list]:
    train_idx, valid_idx, test_idx = split
    start = time.perf_counter()
    models = [fit(graphs, y, train_idx, valid_idx, seed) for seed in range(ENSEMBLE)]
    mu_v, sd_v = ensemble_predict(models, [graphs[i] for i in valid_idx])
    mu_t, sd_t = ensemble_predict(models, [graphs[i] for i in test_idx])
    y_v, y_t = y[valid_idx], y[test_idx]
    sd_v, sd_t = np.maximum(sd_v, 1e-3), np.maximum(sd_t, 1e-3)

    q_norm = conformal_quantile(np.abs(y_v - mu_v) / sd_v)
    q_abs = conformal_quantile(np.abs(y_v - mu_v))
    sims = max_similarity(fps, train_idx, test_idx)
    abs_err = np.abs(y_t - mu_t)
    bins = []
    for lo, hi in SIMILARITY_BINS:
        mask = (sims >= lo) & (sims < hi)
        if mask.sum():
            bins.append({"similarity": f"[{lo:.1f}, {min(hi, 1.0):.1f})", "n": int(mask.sum()),
                         "gnn_rmse": round(float(np.sqrt(np.mean(abs_err[mask] ** 2))), 3),
                         "mean_sigma": round(float(np.mean(sd_t[mask])), 3)})
    result = {
        "sizes": {"train": len(train_idx), "valid": len(valid_idx), "test": len(test_idx)},
        "test": {
            "predict_train_mean": metrics(y_t, np.full_like(y_t, y[train_idx].mean())),
            "linear_4_descriptors": metrics(y_t, linear_baseline(X, y, train_idx, test_idx)),
            "knn_5_tanimoto": metrics(y_t, knn_baseline(fps, y, train_idx, test_idx)),
            "gnn_ensemble": metrics(y_t, mu_t),
        },
        "error_by_similarity_to_training": bins,
        "uncertainty": {
            "spearman_sigma_vs_abs_error": round(spearman(sd_t, abs_err), 3),
            "coverage_ensemble_2sigma": round(float(np.mean(abs_err <= 2 * sd_t)), 3),
            "conformal_scaled_q": round(q_norm, 3),
            "coverage_conformal_scaled": round(float(np.mean(abs_err <= q_norm * sd_t)), 3),
            "mean_width_conformal_scaled": round(float(np.mean(2 * q_norm * sd_t)), 3),
            "conformal_constant_q": round(q_abs, 3),
            "coverage_conformal_constant": round(float(np.mean(abs_err <= q_abs)), 3),
            "mean_width_conformal_constant": round(2 * q_abs, 3),
        },
        "seconds": round(time.perf_counter() - start, 1),
    }
    points = [(float(s), float(e), float(sd)) for s, e, sd in zip(sims, abs_err, sd_t)]
    return {"result": result, "models": models, "q_norm": q_norm, "train_idx": train_idx}, points


def main() -> None:
    torch.set_num_threads(4)
    rows = data.load()
    smiles = [s for s, _ in rows]
    y = np.array([v for _, v in rows])
    graphs = [featurize(s) for s in smiles]
    fps = [chem.fingerprint(s) for s in smiles]
    X = np.array([delaney_features(s) for s in smiles])

    report, points = {"n_molecules": len(rows), "data_sha256": data.SHA256, "ensemble": ENSEMBLE, "alpha": ALPHA}, {}
    deployed = None
    for name, split in (("random", data.random_split(len(rows))), ("scaffold", data.scaffold_split(smiles))):
        out, points[name] = evaluate_split(name, smiles, y, graphs, fps, X, split)
        report[name] = out["result"]
        print(name, json.dumps(out["result"]["test"]), flush=True)
        if name == "scaffold":
            deployed = out

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (RESULTS.parent / "e2_points.json").write_text(json.dumps(points), encoding="utf-8")

    # The deployed model is the scaffold-split ensemble: its conformal factor was
    # calibrated on unseen scaffolds, so its intervals are the honest ones.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for k, (model, _, _) in enumerate(deployed["models"]):
        torch.save(model.state_dict(), MODEL_DIR / f"model_{k}.pt")
    meta = {
        "hidden": 64, "layers": 4, "scalers": [[m, s] for _, m, s in deployed["models"]],
        "conformal_q": deployed["q_norm"], "alpha": ALPHA, "data_sha256": data.SHA256,
        "domain_threshold": 0.3,
        "train_smiles": [smiles[i] for i in deployed["train_idx"]],
        "test_metrics": report["scaffold"]["test"]["gnn_ensemble"],
    }
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    print(json.dumps({k: report[k]["uncertainty"] for k in ("random", "scaffold")}, indent=2))


if __name__ == "__main__":
    main()
