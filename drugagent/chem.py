"""Cheminformatics tools - the deterministic ground the agent stands on.

Every public function takes a SMILES string and returns plain Python values, so one
implementation backs the agent's tool calls, the MCP server and the verifier.

A property is not one number. "logP" can mean Wildman-Crippen (RDKit), XLogP3
(PubChem) or Lipinski's original MLogP, and "H-bond acceptors" can mean "every N and
O" or a pattern that skips amide nitrogens. `PROPERTY_METHODS` names each variant
explicitly so a claim can say which one it means - and so the verifier can tell a
wrong number from a number computed a different way.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Callable

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import QED, Crippen, Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")


@lru_cache(maxsize=8192)
def parse(smiles: str) -> Chem.Mol:
    """Parse a SMILES string, raising ValueError instead of returning None."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")
    return mol


def is_valid_smiles(smiles: str) -> bool:
    try:
        parse(smiles)
    except ValueError:
        return False
    return True


def canonical(smiles: str) -> str:
    return Chem.MolToSmiles(parse(smiles))


# property -> method -> function(mol). The first method listed is the default.
PROPERTY_METHODS: dict[str, dict[str, Callable[[Chem.Mol], float]]] = {
    "mw": {"rdkit": Descriptors.MolWt},
    "logp": {"crippen": Crippen.MolLogP},
    # Ertl's TPSA counts N and O; including S and P is a common variant.
    "tpsa": {
        "ertl": lambda m: rdMolDescriptors.CalcTPSA(m),
        "ertl_sp": lambda m: rdMolDescriptors.CalcTPSA(m, includeSandP=True),
    },
    # Lipinski's original rule counts every OH and NH as a donor and every N and O
    # as an acceptor; RDKit's Lipinski module uses SMARTS patterns that do not.
    "hbd": {"lipinski": rdMolDescriptors.CalcNumLipinskiHBD, "rdkit": Lipinski.NumHDonors},
    "hba": {"lipinski": rdMolDescriptors.CalcNumLipinskiHBA, "rdkit": Lipinski.NumHAcceptors},
    "rotb": {"rdkit": rdMolDescriptors.CalcNumRotatableBonds},
    "heavy_atoms": {"rdkit": lambda m: m.GetNumHeavyAtoms()},
    "rings": {"rdkit": rdMolDescriptors.CalcNumRings},
    "aromatic_rings": {"rdkit": rdMolDescriptors.CalcNumAromaticRings},
    "fsp3": {"rdkit": rdMolDescriptors.CalcFractionCSP3},
    "qed": {"rdkit": QED.qed},
}

COUNT_PROPERTIES = {"hbd", "hba", "rotb", "heavy_atoms", "rings", "aromatic_rings"}


def compute(prop: str, smiles: str, method: str | None = None) -> float:
    """One property by one named method. Counts come back as int."""
    methods = PROPERTY_METHODS[prop]
    fn = methods[method] if method else next(iter(methods.values()))
    value = fn(parse(smiles))
    return int(value) if prop in COUNT_PROPERTIES else float(value)


def compute_all_methods(prop: str, smiles: str) -> dict[str, float]:
    return {method: compute(prop, smiles, method) for method in PROPERTY_METHODS[prop]}


def descriptors(smiles: str) -> dict:
    """The default-method value of every property, plus formula and canonical SMILES."""
    mol = parse(smiles)
    out: dict = {"smiles": Chem.MolToSmiles(mol), "formula": rdMolDescriptors.CalcMolFormula(mol)}
    for prop in PROPERTY_METHODS:
        out[prop] = compute(prop, smiles)
    out["logp_method"] = "crippen"
    out["hbd_method"] = out["hba_method"] = "lipinski"
    return out


def lipinski(smiles: str, logp_method: str = "crippen", hbd_method: str = "lipinski",
             hba_method: str = "lipinski") -> dict:
    """Lipinski's Rule of Five: poor absorption is more likely when more than one rule fails.

    Lipinski (1997) used MLogP; RDKit does not implement it, so Crippen stands in.
    That substitution is itself a method choice, and the result records it.
    """
    values = {
        "mw": compute("mw", smiles),
        "logp": compute("logp", smiles, logp_method),
        "hbd": compute("hbd", smiles, hbd_method),
        "hba": compute("hba", smiles, hba_method),
    }
    limits = {"mw": 500, "logp": 5, "hbd": 5, "hba": 10}
    violations = [prop for prop, limit in limits.items() if values[prop] > limit]
    return {
        "values": values,
        "violations": violations,
        "n_violations": len(violations),
        "passes": len(violations) <= 1,
        "methods": {"logp": logp_method, "hbd": hbd_method, "hba": hba_method},
    }


def veber(smiles: str) -> dict:
    """Veber (2002): oral bioavailability favours <= 10 rotatable bonds and TPSA <= 140."""
    rotb, tpsa = compute("rotb", smiles), compute("tpsa", smiles)
    return {"rotb": rotb, "tpsa": tpsa, "passes": rotb <= 10 and tpsa <= 140}


@lru_cache(maxsize=1)
def _alert_catalog() -> FilterCatalog:
    params = FilterCatalogParams()
    for catalog in (FilterCatalogParams.FilterCatalogs.PAINS, FilterCatalogParams.FilterCatalogs.BRENK):
        params.AddCatalog(catalog)
    return FilterCatalog(params)


def structural_alerts(smiles: str) -> list[str]:
    """PAINS and Brenk substructure alerts. An alert is a flag for a chemist, not a verdict."""
    entries = _alert_catalog().GetMatches(parse(smiles))
    return sorted({entry.GetDescription() for entry in entries})


_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fingerprint(smiles: str):
    return _MORGAN.GetFingerprint(parse(smiles))


def tanimoto(a: str, b: str) -> float:
    return DataStructs.TanimotoSimilarity(fingerprint(a), fingerprint(b))


def bulk_tanimoto(query: str, library: list) -> list[float]:
    """Similarity of one SMILES to a list of SMILES or precomputed fingerprints."""
    fps = [fingerprint(item) if isinstance(item, str) else item for item in library]
    return list(DataStructs.BulkTanimotoSimilarity(fingerprint(query), fps))


def nearest(query: str, library: dict[str, str], k: int = 5) -> list[tuple[str, float]]:
    """The k most similar entries of a {name: smiles} library, most similar first."""
    names = list(library)
    sims = bulk_tanimoto(query, [library[name] for name in names])
    ranked = sorted(zip(names, sims), key=lambda pair: (-pair[1], pair[0]))
    return ranked[:k]


def murcko_scaffold(smiles: str) -> str:
    """Bemis-Murcko scaffold: ring systems plus the linkers between them. Acyclic -> ''."""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=parse(smiles))
