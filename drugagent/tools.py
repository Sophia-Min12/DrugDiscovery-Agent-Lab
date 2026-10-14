"""The tool registry - one definition used by the agent graph and the MCP server.

A tool takes JSON-able arguments and returns a JSON-able dict. Values that depend on
a method are returned keyed by method, so whatever reads the result (an LLM, a
person, another agent over MCP) sees the disagreement instead of one number.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

from . import chem, drugs, omics
from .verifier import lipinski_variants


@dataclass
class Tool:
    name: str
    description: str
    params: dict[str, str] = field(default_factory=dict)  # name -> description
    fn: Callable[..., dict] = None


REGISTRY: dict[str, Tool] = {}


def tool(name: str, description: str, **params: str):
    def register(fn):
        REGISTRY[name] = Tool(name, description, params, fn)
        return fn
    return register


def call(name: str, args: dict) -> dict:
    """Run a tool; failures come back as {"error": ...} so a bad call never crashes the graph."""
    if name not in REGISTRY:
        return {"error": f"unknown tool {name!r}"}
    try:
        return REGISTRY[name].fn(**args)
    except (ValueError, TypeError, KeyError, RuntimeError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def catalog() -> str:
    lines = []
    for t in REGISTRY.values():
        params = ", ".join(f"{p}: {d}" for p, d in t.params.items())
        lines.append(f"- {t.name}({params}): {t.description}")
    return "\n".join(lines)


def _round(value):
    if isinstance(value, float):
        return round(value, 2)
    if isinstance(value, dict):
        return {k: _round(v) for k, v in value.items()}
    return value


def _smiles(compound: str) -> str:
    smiles = drugs.resolve(compound)
    if smiles is None:
        raise ValueError(f"cannot resolve {compound!r} to a structure (not a known drug name or valid SMILES)")
    return smiles


@tool("resolve_compound", "Look up a drug name (or check a SMILES) and return its structure and PubChem CID.",
      compound="drug name or SMILES")
def resolve_compound(compound: str) -> dict:
    drug = drugs.lookup(compound)
    if drug is not None:
        return {"compound": drug.name, "smiles": drug.smiles, "formula": drug.formula, "pubchem_cid": drug.cid}
    smiles = _smiles(compound)
    return {"compound": compound, "smiles": chem.canonical(smiles), "pubchem_cid": None}


@tool("compute_descriptors", "Molecular weight, logP, TPSA, H-bond donors/acceptors, rotatable bonds, QED. "
      "Method-dependent properties are returned keyed by method.", compound="drug name or SMILES")
def compute_descriptors(compound: str) -> dict:
    smiles = _smiles(compound)
    out = {"compound": compound, "smiles": smiles, "formula": chem.descriptors(smiles)["formula"]}
    for prop in ("mw", "logp", "tpsa", "hbd", "hba", "rotb"):
        out[prop] = _round(drugs.all_methods(compound, prop))
    for prop in ("qed", "heavy_atoms", "aromatic_rings", "fsp3"):
        out[prop] = _round(chem.compute(prop, smiles))
    return out


@tool("drug_likeness", "Lipinski Rule of Five and Veber rules. The Rule-of-Five verdict is given under every "
      "logP x H-bond counting method, because they can disagree.", compound="drug name or SMILES")
def drug_likeness(compound: str) -> dict:
    smiles = _smiles(compound)
    variants = lipinski_variants(compound)
    verdicts = {name: n <= 1 for name, n in variants.items()}
    agreed = len(set(verdicts.values())) == 1
    return {
        "compound": compound,
        "lipinski_violations_by_method": variants,
        "lipinski_passes_by_method": verdicts,
        "lipinski_verdict": ("passes" if next(iter(verdicts.values())) else "fails") if agreed
        else "depends on the logP / H-bond method",
        "veber": chem.veber(smiles) | {"tpsa": round(chem.compute("tpsa", smiles), 2)},
    }


@tool("structural_alerts", "PAINS and Brenk substructure alerts (flags for a chemist, not verdicts).",
      compound="drug name or SMILES")
def structural_alerts(compound: str) -> dict:
    return {"compound": compound, "alerts": chem.structural_alerts(_smiles(compound))}


@tool("similar_compounds", "The most similar approved drugs by Morgan-fingerprint Tanimoto similarity.",
      compound="drug name or SMILES", k="number of neighbours (default 5)")
def similar_compounds(compound: str, k: int = 5) -> dict:
    smiles = _smiles(compound)
    library = {name: s for name, s in drugs.library().items() if chem.canonical(s) != chem.canonical(smiles)}
    return {"compound": compound, "neighbours": [{"name": n, "tanimoto": round(s, 3)}
                                                 for n, s in chem.nearest(smiles, library, int(k))]}


@tool("differential_expression", "Differential expression (treated vs control) of a named RNA-seq dataset: "
      "log2 CPM, Welch t-test, Benjamini-Hochberg. Returns counts of significant genes and the top genes.",
      dataset=f"one of {', '.join(omics.DATASETS)} (default demo)", top="number of top genes (default 10)")
def differential_expression(dataset: str = "demo", top: int = 10) -> dict:
    return omics.summary(dataset, top=int(top))


@tool("gene_result", "Log2 fold change, p-value and adjusted p-value of one gene in a named RNA-seq dataset.",
      gene="gene id such as G0042", dataset="dataset name (default demo)")
def gene_result(gene: str, dataset: str = "demo") -> dict:
    row = omics.analysed(dataset).row(gene)
    if row is None:
        raise ValueError(f"no gene {gene!r} in dataset {dataset!r}")
    return {"dataset": dataset, **row}


try:  # torch is an optional dependency; without it the tool is simply not offered
    from .gnn import predict as _gnn
except ImportError:
    _gnn = None

if _gnn is not None:
    @tool("predict_solubility", "Predicted aqueous solubility (log10 mol/L) from a GNN ensemble, with a 95% "
          "interval and an applicability-domain check. A prediction, not a measurement.",
          compound="drug name or SMILES")
    def predict_solubility(compound: str) -> dict:
        return _round(_gnn.predict(compound))


def as_json(result: dict) -> str:
    return json.dumps(result, ensure_ascii=False, default=str)
