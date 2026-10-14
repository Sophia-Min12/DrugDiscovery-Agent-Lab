"""The verifier: recompute every claim with a tool, never with another LLM call.

Evidence comes from *sources*, functions that take a claim and return the values the
repo's tools produce for it (one per accepted method), or None if they cannot speak
to it. The chemistry source is built in; the GNN and omics modules register theirs.

The judgement is the same for every source:

* every accepted method agrees with the claim  -> SUPPORTED
* none does                                    -> CONTRADICTED
* some do, and the claim did not name a method -> AMBIGUOUS
* no source can produce a value                -> UNVERIFIABLE
* a prediction stated without its uncertainty  -> UNQUALIFIED
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Callable, Iterable

from . import chem, drugs
from .claims import LOG10_TOLERANCE, PREDICTED, Check, Claim, Evidence, Verdict, tolerance

Source = Callable[[Claim], "Evidence | None"]

# Names a writer might use for a method, mapped to the repo's method names.
METHOD_ALIASES = {
    "xlogp": "xlogp3", "xlogp3": "xlogp3", "pubchem": "pubchem",
    "crippen": "crippen", "wildman-crippen": "crippen", "rdkit": "rdkit",
}


def lipinski_variants(subject: str) -> dict[str, int] | None:
    """Rule-of-Five violation counts under every logP x H-bond method this repo has."""
    smiles = drugs.resolve(subject)
    if smiles is None:
        return None
    mw = chem.compute("mw", smiles)
    logps = {"crippen": chem.compute("logp", smiles), **drugs.reference_values(subject, "logp")}
    hbonds = {m: (chem.compute("hbd", smiles, m), chem.compute("hba", smiles, m)) for m in ("lipinski", "rdkit")}
    ref_hbd, ref_hba = drugs.reference_values(subject, "hbd"), drugs.reference_values(subject, "hba")
    if ref_hbd and ref_hba:
        hbonds["pubchem"] = (ref_hbd["pubchem"], ref_hba["pubchem"])
    variants = {}
    for logp_method, logp in logps.items():
        for hb_method, (hbd, hba) in hbonds.items():
            variants[f"{logp_method}+{hb_method}"] = int(mw > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10)
    return variants


def chem_source(claim: Claim) -> Evidence | None:
    prop = claim.prop
    if prop in chem.PROPERTY_METHODS:
        values = drugs.all_methods(claim.subject, prop)
        return Evidence(values) if values else None
    if prop in ("lipinski_pass", "lipinski_violations"):
        variants = lipinski_variants(claim.subject)
        if variants is None:
            return None
        if prop == "lipinski_pass":
            return Evidence({name: float(n <= 1) for name, n in variants.items()})
        return Evidence({name: float(n) for name, n in variants.items()})
    if prop == "veber_pass":
        smiles = drugs.resolve(claim.subject)
        if smiles is None:
            return None
        tpsas, rotbs = drugs.all_methods(claim.subject, "tpsa"), drugs.all_methods(claim.subject, "rotb")
        return Evidence({f"{t}+{r}": float(tv <= 140 and rv <= 10)
                         for t, tv in tpsas.items() for r, rv in rotbs.items()})
    return None


def judge(claim: Claim, evidence: Evidence | None) -> Check:
    if evidence is None or not evidence.values:
        return Check(claim, Verdict.UNVERIFIABLE, evidence, "no tool can compute this")

    if claim.prop in PREDICTED:
        low, high = evidence.interval
        if not low <= claim.value <= high:
            return Check(claim, Verdict.CONTRADICTED, evidence, "outside the model's own interval")
        if claim.interval is None:
            return Check(claim, Verdict.UNQUALIFIED, evidence, "a prediction needs its uncertainty")
        return Check(claim, Verdict.SUPPORTED, evidence, agreeing=list(evidence.values))

    values = evidence.values
    if claim.method:
        method = METHOD_ALIASES.get(claim.method.lower(), claim.method.lower())
        matching = {m: v for m, v in values.items() if m == method or m.startswith(method + "+")}
        if not matching:
            return Check(claim, Verdict.UNVERIFIABLE, evidence,
                         f"method {claim.method!r} not available (have: {', '.join(values)})")
        values = matching

    tol = tolerance(claim.prop) + 1e-9
    if claim.op == "<":
        agreeing = [m for m, v in values.items() if v < claim.value]
    elif claim.prop in LOG10_TOLERANCE:
        log = lambda x: math.log10(max(x, 1e-300))  # noqa: E731
        agreeing = [m for m, v in values.items()
                    if claim.value > 0 and abs(log(claim.value) - log(v)) <= LOG10_TOLERANCE[claim.prop]]
    else:
        agreeing = [m for m, v in values.items() if abs(claim.value - v) <= tol]
    if len(agreeing) == len(values):
        return Check(claim, Verdict.SUPPORTED, evidence, agreeing=agreeing)
    if not agreeing:
        return Check(claim, Verdict.CONTRADICTED, evidence)
    disagreeing = [m for m in values if m not in agreeing]
    return Check(claim, Verdict.AMBIGUOUS, evidence,
                 f"true under {', '.join(agreeing)}; false under {', '.join(disagreeing)}", agreeing)


class Verifier:
    def __init__(self, sources: Iterable[Source] | None = None):
        self.sources: list[Source] = list(sources) if sources is not None else [chem_source]

    def add_source(self, source: Source) -> "Verifier":
        self.sources.append(source)
        return self

    def evidence(self, claim: Claim) -> Evidence | None:
        for source in self.sources:
            found = source(claim)
            if found is not None:
                return found
        return None

    def check(self, claim: Claim) -> Check:
        return judge(claim, self.evidence(claim))

    def check_all(self, claims: Iterable[Claim]) -> list[Check]:
        return [self.check(claim) for claim in claims]


def summarize(checks: Iterable[Check]) -> dict[str, int]:
    counts = Counter(check.verdict.value for check in checks)
    return {verdict.value: counts.get(verdict.value, 0) for verdict in Verdict}
