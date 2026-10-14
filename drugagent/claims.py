"""Claims, evidence and verdicts - the vocabulary the verifier speaks.

A claim is one checkable statement pulled out of agent text: a subject, a property,
a value and, ideally, the method the value was computed with. The verifier never
judges prose; it judges claims.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Verdict(str, Enum):
    SUPPORTED = "supported"        # matches every accepted method within tolerance
    CONTRADICTED = "contradicted"  # matches no accepted method
    AMBIGUOUS = "ambiguous"        # true under one method, false under another; the claim did not say which
    UNVERIFIABLE = "unverifiable"  # no tool in this repo can check it
    UNQUALIFIED = "unqualified"    # a model prediction stated without its uncertainty


# What a property is, which decides how it is checked.
COMPUTED = {"mw", "logp", "tpsa", "hbd", "hba", "rotb", "heavy_atoms", "rings", "aromatic_rings",
            "fsp3", "qed", "lipinski_pass", "lipinski_violations", "veber_pass"}
PREDICTED = {"log_solubility"}
STATISTICAL = {"log2fc", "padj", "n_significant"}

# Absolute tolerance per property. Counts and booleans must match exactly. The others
# are set at roughly the precision a value is reported to, not at what a method
# could achieve - logP methods disagree by far more than 0.3 (experiment E1).
TOLERANCE = {
    "mw": 0.5, "logp": 0.3, "tpsa": 1.0, "fsp3": 0.02, "qed": 0.02,
    "log2fc": 0.1, "log_solubility": 0.0,
}


# p-values are compared on a log10 scale: "3.2e-05" for 3.21e-05 is right, and an
# absolute tolerance would call 0.04 and 0.0004 equal.
LOG10_TOLERANCE = {"padj": 0.1}


def tolerance(prop: str) -> float:
    return TOLERANCE.get(prop, 0.0)


@dataclass(frozen=True)
class Claim:
    subject: str
    prop: str
    value: float
    method: str | None = None
    op: str = "="                                 # "=" or "<" (an upper bound: "padj < 1e-4")
    interval: tuple[float, float] | None = None  # stated uncertainty, for predictions
    span: tuple[int, int] | None = None          # character offsets in the source text
    text: str = ""                                # the matched text, for reports

    def describe(self) -> str:
        method = f" ({self.method})" if self.method else ""
        interval = f" [{self.interval[0]:g}, {self.interval[1]:g}]" if self.interval else ""
        return f"{self.subject}: {self.prop}{method} {self.op} {self.value:g}{interval}"


@dataclass
class Evidence:
    """What a tool says the truth is: one value per accepted method."""
    values: dict[str, float]
    interval: tuple[float, float] | None = None  # a model's own uncertainty, for predictions
    note: str = ""


@dataclass
class Check:
    claim: Claim
    verdict: Verdict
    evidence: Evidence | None = None
    note: str = ""
    agreeing: list[str] = field(default_factory=list)  # methods the claim matches

    @property
    def ok(self) -> bool:
        return self.verdict is Verdict.SUPPORTED

    def expected(self) -> str:
        if self.evidence is None or not self.evidence.values:
            return "-"
        parts = [f"{method}={value:g}" for method, value in self.evidence.values.items()]
        if self.evidence.interval:
            low, high = self.evidence.interval
            parts.append(f"interval=[{low:.2f}, {high:.2f}]")
        return ", ".join(parts)
