"""Pull checkable claims out of free text.

Rule-based on purpose. An LLM-based extractor would be one more model whose output
needs checking; a regex either matches or it does not, and its misses can be counted
(experiment E5 audits them by hand).

Two kinds of number are deliberately *not* claims:

* rule thresholds - "MW under 500", "TPSA <= 140", "at most 5 donors";
* identifiers - "G0042", "CYP3A4", "C9H8O4", "Rule of 5".

Every other number that no claim covers is returned as *unattributed*: the report
cannot vouch for it, so the agent is told to source it or remove it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import drugs
from .claims import Claim

WORD_NUMBERS = {"no": 0, "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}

NUM = r"[-−–]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
SCI = NUM + r"(?:\s*[eE]\s*[-−]?\d+|\s*[x×]\s*10\s*\^?\s*[-−]?\d+)?"
COUNT = r"\d+|" + "|".join(WORD_NUMBERS)
FILLER = r"[^\d\n.;!?]{0,30}?"
FLAGS = re.IGNORECASE

KEYS = {
    "mw": r"molecular\s+weight|molecular\s+mass|molar\s+mass|mol\.?\s?wt\.?|\bMW\b",
    "logp": r"\b(?:wildman[-\s]crippen\s+|crippen\s+)?(?:c|x|a|m)?log\s?p(?:3|o/w)?\b|partition\s+coefficient",
    "tpsa": r"\bT?PSA\b|(?:topological\s+)?polar\s+surface\s+area",
    "hbd": r"(?:hydrogen|H)[-\s]?bond\s+donors?|\bHBDs?\b",
    "hba": r"(?:hydrogen|H)[-\s]?bond\s+acceptors?|\bHBAs?\b",
    "rotb": r"rotatable\s+bonds?",
    "qed": r"\bQED\b(?:\s+score)?",
    "heavy_atoms": r"heavy\s+atoms?",
    "log_solubility": r"\blog\s?S\b|log\s+solubility|aqueous\s+solubility|\bsolubility\b",
    "log2fc": r"\blog2?\s?\(?\s*fold[-\s]?change\s*\)?|\blog2\s?FC\b|\bLFC\b|\blog\s?FC\b",
    "padj": r"adjusted\s+p[-\s]?values?|\bp[-_\s]?adj\b|\bFDR\b|\bq[-\s]?values?\b",
}
KEY_RES = {prop: re.compile(key, FLAGS) for prop, key in KEYS.items()}
VALUE_AFTER = re.compile(rf"(?P<fill>{FILLER})(?P<num>{NUM})", FLAGS)
SCI_AFTER = re.compile(rf"(?P<fill>{FILLER})(?P<num>{SCI})", FLAGS)
ANY_KEY = re.compile("|".join(f"(?:{key})" for key in KEYS.values()), FLAGS)

# Method names a writer puts in parentheses after a value, and which property each fits.
METHOD_WORDS = {"wildman crippen": "crippen", "crippen": "crippen", "xlogp3": "xlogp3", "xlogp": "xlogp3",
                "rdkit": "rdkit", "pubchem": "pubchem", "lipinski": "lipinski", "ertl": "ertl",
                "ertl_sp": "ertl_sp", "ertl sp": "ertl_sp", "ertlsp": "ertl_sp"}
VALID_METHODS = {"logp": {"crippen", "xlogp3"}, "mw": {"rdkit", "pubchem"}, "tpsa": {"ertl", "ertl_sp", "pubchem"},
                 "hbd": {"lipinski", "rdkit", "pubchem"}, "hba": {"lipinski", "rdkit", "pubchem"},
                 "rotb": {"rdkit", "pubchem"}}
# RDKit's logP *is* Crippen, PubChem's is XLogP3, RDKit's TPSA is Ertl's.
METHOD_FOR = {("logp", "rdkit"): "crippen", ("logp", "pubchem"): "xlogp3", ("tpsa", "rdkit"): "ertl"}
METHOD_WORD = r"wildman[-\s]crippen|crippen|xlogp3?|rdkit|pubchem|lipinski|ertl[_\s-]?sp\b|ertl"
METHOD_WORD_RE = re.compile(METHOD_WORD, FLAGS)
# "3.5 (XLogP3)": a logP method name alone identifies the property.
VALUE_THEN_LOGP_METHOD = re.compile(rf"(?P<num>{NUM})\s*\(\s*(?P<m>XLogP3?|Wildman[-\s]Crippen|Crippen)\s*\)", FLAGS)

# "2 hydrogen bond donors", "no rotatable bonds": the count comes first.
COUNT_FIRST = re.compile(
    rf"\b(?P<num>{COUNT})\s+(?P<key>(?:(?:hydrogen|H)[-\s]?bond\s+)?donors?|(?:(?:hydrogen|H)[-\s]?bond\s+)?acceptors?"
    rf"|HBDs?|HBAs?|rotatable\s+bonds?|heavy\s+atoms?)\b", FLAGS)

THRESHOLD = re.compile(r"[<>≤≥]|less\s+than|greater\s+than|more\s+than|below|above|under|over|exceed|limit|"
                       r"threshold|cut-?off|maximum|\bmax\b|up\s+to|at\s+most|no\s+more|not\s+exceed|within", FLAGS)
INEQUALITY = re.compile(r"(?P<op><|≤|less\s+than|below)\s*$", FLAGS)

UNIT = r"(?:\s*(?:g/mol|da|å²|å2|a\^?2))?"
METHOD_AFTER = re.compile(rf"^{UNIT}\s*\((?P<inside>[^()\d]{{0,40}})\)", FLAGS)
# "4 acceptors according to RDKit": the method in words rather than parentheses.
METHOD_PHRASE_AFTER = re.compile(rf"^{UNIT}\s*,?\s*(?:according\s+to|by|per|using|via|from)\s+(?:the\s+)?"
                                 rf"(?P<inside>{METHOD_WORD})", FLAGS)
# "306.24 (RDKit) and 306.2 (PubChem)": further values of the same property, each with its method.
CONTINUATION = re.compile(rf"^(?:\s*,\s*(?:and\s+|or\s+)?|\s+(?:and|or)\s+|\s*;\s*)(?P<num>{NUM}){UNIT}"
                          rf"\s*\((?P<inside>[^()\d]{{0,40}})\)", FLAGS)
PLUS_MINUS = re.compile(rf"^\s*(?:log\s?units?|log\s?\(mol/l\)|mol/l)?\s*(?:±|\+/-|\+-)\s*(?P<pm>{NUM})", FLAGS)
RANGE = re.compile(rf"^[^\d\n]{{0,30}}?(?:\d+\s*%\s*)?(?:prediction\s+|confidence\s+)?(?:interval|CI|range|between)[^\d\n]{{0,12}}?"
                   rf"(?P<lo>{NUM})\s*(?:to|–|—|and|,|\.\.)\s*(?P<hi>{NUM})", FLAGS)

LIP = r"(?:lipinski(?:'s|’s)?(?:\s+rule\s+of\s+(?:five|5))?(?:\s+rules?)?|(?:the\s+)?rule\s+of\s+(?:five|5)|Ro5)"
FAIL_RE = re.compile(
    rf"\b(?:fails?|failing|violates?|violating|breaks?|does\s+not\s+(?:pass|satisfy|obey|comply\s+with|meet|follow|adhere\s+to)"
    rf"|doesn['’]t\s+(?:pass|satisfy|obey|comply\s+with|meet|follow)|is\s+not\s+compliant\s+with|non[-\s]?compliant\s+with"
    rf"|outside(?:\s+of)?|beyond)\s+(?:all\s+(?:of\s+)?)?(?:the\s+)?{LIP}|\b(?:not|non)[-\s]{LIP}[-\s]compliant", FLAGS)
PASS_RE = re.compile(
    rf"\b(?:passes|pass|passing|satisf(?:ies|y|ying)|obeys?|obeying|compl(?:ies|y|ying)\s+with|meets?|meeting|follows?"
    rf"|adheres?\s+to|conforms?\s+to|fulfil+s?|falls\s+within|is\s+(?:fully\s+)?compliant\s+with)\s+(?:all\s+(?:of\s+)?|with\s+)?"
    rf"(?:the\s+)?{LIP}|\b{LIP}[-\s]compliant", FLAGS)
PARTIAL = re.compile(r"^\s+(?:for|on|regarding|concerning|about|in\s+terms\s+of)\b", FLAGS)
# "... Lipinski's Rule of Five; ibuprofen passes the rule" - short forms, accepted only in a
# sentence that names the rule.
FAIL_SHORT = re.compile(r"\b(?:fails|violates|does\s+not\s+(?:pass|satisfy|meet))\s+(?:the|these|its|all)?\s*"
                        r"(?:rules?|criteria|test)\b", FLAGS)
PASS_SHORT = re.compile(r"\b(?:passes|satisfies|meets|fulfil+s|obeys|complies\s+with)\s+(?:all\s+(?:of\s+)?)?"
                        r"(?:the|these|its|all)?\s*(?:rules?|criteria|conditions|test)\b", FLAGS)
NAMES_THE_RULE = re.compile(LIP, FLAGS)
# "... does not pass according to Crippen and Lipinski, but does pass according to XLogP3 and Lipinski"
FAIL_BY_METHOD = re.compile(r"\b(?:does\s+not\s+pass|doesn['’]t\s+pass|fails)"
                            r"(?=\s*,?\s+(?:according\s+to|using|under|with|by)\b)", FLAGS)
PASS_BY_METHOD = re.compile(r"\b(?:does\s+pass|passes|still\s+passes)"
                            r"(?=\s*,?\s+(?:according\s+to|using|under|with|by)\b)", FLAGS)
# The logP x H-bond method a Rule-of-Five verdict was computed under, named in its sentence.
LIPINSKI_METHOD = re.compile(r"(?:according\s+to|using|with|under|by|for)\s+(?:the\s+)?(?P<logp>crippen|xlogp3?)"
                             r"(?:\s+log\s?p)?(?:\s*(?:and|\+|/|,)\s*(?:the\s+)?(?P<hb>lipinski|rdkit|pubchem)"
                             r"(?!['’]s|\s+rule))?", FLAGS)
VIOLATIONS_RE = re.compile(
    rf"\b(?P<num>{COUNT})\s+(?:{LIP}\s+)?violations?\b"
    rf"|\bviolat\w*\s+(?P<num2>{COUNT})\s+(?:of\s+)?(?:(?:the|its|lipinski['’]s|lipinski)\s+)*(?:rules?|criteria|criterion)", FLAGS)

N_SIGNIFICANT = re.compile(
    rf"\b(?P<num>\d+)\s+(?:(?:genes?|transcripts?)\s+(?:were|are|was)\s+(?:significantly\s+)?(?:differentially\s+expressed|significant)"
    rf"|(?:significantly\s+)?differentially\s+expressed\s+genes|DEGs?\b|significant\s+genes)", FLAGS)
GENE = re.compile(r"\bG\d{4}\b")
GENE_PROPS = {"log2fc", "padj"}

NUMBER_TOKEN = re.compile(rf"(?<![\w.\-−–]){NUM}(?![\w])")
IDENTIFIER_CONTEXT = re.compile(r"(?:rule\s+of|log|ro|top|\d+\s*%\s*(?:CI|confidence|interval|prediction))\s*$", FLAGS)


@dataclass
class Extraction:
    claims: list[Claim] = field(default_factory=list)
    unattributed: list[tuple[int, int, str]] = field(default_factory=list)


def _number(text: str) -> float:
    text = text.strip().lower().replace("−", "-").replace("–", "-").replace(",", "")
    if text in WORD_NUMBERS:
        return float(WORD_NUMBERS[text])
    sci = re.match(r"^(-?[\d.]+)\s*[x×]\s*10\s*\^?\s*(-?\d+)$", text)
    if sci:
        return float(sci.group(1)) * 10 ** int(sci.group(2))
    return float(text.replace(" ", ""))


def _logp_method(key: str) -> str | None:
    key = key.lower()
    for marker, method in (("xlog", "xlogp3"), ("crippen", "crippen"), ("alog", "alogp"), ("mlog", "mlogp")):
        if marker in key:
            return method
    return None


def _method_in(prop: str, inside: str) -> str | None:
    """The first method named in `inside` that fits `prop` ('Lipinski' is no method for MW)."""
    for word in METHOD_WORD_RE.findall(inside):
        method = METHOD_WORDS[re.sub(r"[-\s]+", " ", word.lower())]
        method = METHOD_FOR.get((prop, method), method)
        if method in VALID_METHODS.get(prop, ()):
            return method
    return None


def _method_after(prop: str, text: str, end: int) -> tuple[str | None, int]:
    """A method named right after a value: '206.28 (RDKit)', '2 (Lipinski, PubChem)', '4 according to RDKit'."""
    for pattern in (METHOD_AFTER, METHOD_PHRASE_AFTER):
        match = pattern.match(text[end:end + 60])
        if match:
            method = _method_in(prop, match.group("inside"))
            if method is not None:
                return method, end + match.end()
    return None, end


def _continuations(prop: str, text: str, end: int) -> list[tuple[tuple[int, int], dict]]:
    out = []
    while True:
        match = CONTINUATION.match(text[end:end + 80])
        method = _method_in(prop, match.group("inside")) if match else None
        if method is None:
            return out
        span = (end + match.start("num"), end + match.end())
        out.append((span, dict(prop=prop, value=_number(match.group("num")), method=method)))
        end = span[1]


def _lipinski_method(text: str, pos: int) -> str | None:
    lo, hi = _sentence_bounds(text, pos)
    phrases = sorted(LIPINSKI_METHOD.finditer(text, lo, hi), key=lambda m: abs(m.start() - pos))
    if not phrases:
        return None
    logp = "xlogp3" if phrases[0].group("logp").lower().startswith("x") else "crippen"
    hb = phrases[0].group("hb")
    return f"{logp}+{hb.lower()}" if hb else logp


def _sentence_bounds(text: str, pos: int) -> tuple[int, int]:
    start = max(text.rfind(". ", 0, pos), text.rfind("\n", 0, pos), text.rfind("? ", 0, pos), text.rfind("! ", 0, pos))
    ends = [i for i in (text.find(". ", pos), text.find("\n", pos)) if i != -1]
    return (start + 1 if start != -1 else 0), (min(ends) if ends else len(text))


def mentions(text: str, subjects: list[str]) -> list[tuple[int, str]]:
    found = []
    for name in subjects:
        for match in re.finditer(rf"(?<![\w-]){re.escape(name)}(?![\w-])", text, FLAGS):
            found.append((match.start(), drugs.SYNONYMS.get(name.lower(), name)))
    return sorted(found)


def _overlaps(span: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    return any(span[0] < end and start < span[1] for start, end in spans)


def extract(text: str, subject: str | None = None, extra_subjects: list[str] | None = None,
            dataset: str = "dataset") -> Extraction:
    """Claims in `text`, each attributed to the closest preceding drug mention (else `subject`)."""
    names = list(drugs.names()) + list(drugs.SYNONYMS) + list(extra_subjects or [])
    if subject:
        names.append(subject)
    seen = mentions(text, sorted(set(names), key=len, reverse=True))

    def subject_at(pos: int) -> str | None:
        before = [name for start, name in seen if start < pos]
        return before[-1] if before else subject

    def gene_at(pos: int) -> str | None:
        lo, hi = _sentence_bounds(text, pos)
        genes = [(abs(m.start() - pos), m.group()) for m in GENE.finditer(text, lo, hi)]
        return min(genes)[1] if genes else None

    raw: list[tuple[tuple[int, int], dict]] = []  # (span, claim fields), resolved below

    for match in COUNT_FIRST.finditer(text):
        if THRESHOLD.search(text[max(0, match.start() - 12):match.start()]):
            continue  # "at most 5 donors" is the rule, not the molecule
        key = match.group("key").lower()
        prop = ("hbd" if "donor" in key or key.startswith("hbd") else "hba" if "accept" in key or key.startswith("hba")
                else "rotb" if "rotat" in key else "heavy_atoms")
        method, end = _method_after(prop, text, match.end())
        raw.append(((match.start(), end), dict(prop=prop, value=_number(match.group("num")), method=method)))
        if method is not None:
            raw += _continuations(prop, text, end)
    count_first_spans = [span for span, _ in raw]

    # Every key occurrence is tried on its own, so "polar surface area (TPSA) is 37.3"
    # falls through from the long name to the abbreviation instead of being lost.
    for prop, key_re in KEY_RES.items():
        for key in key_re.finditer(text):
            match = (SCI_AFTER if prop == "padj" else VALUE_AFTER).match(text, key.end())
            if not match:
                continue
            fill, num_span = match.group("fill"), match.span("num")
            if _overlaps(num_span, count_first_spans) or ANY_KEY.search(fill):
                continue
            op = "="
            if THRESHOLD.search(fill):
                if prop != "padj" or not INEQUALITY.search(fill):
                    continue  # a rule threshold, not a claim about this molecule
                op = "<"
            fields = dict(prop=prop, value=_number(match.group("num")), op=op)
            end = match.end()
            method = _logp_method(key.group()) if prop == "logp" else None
            if method is None:
                method, end = _method_after(prop, text, end)
            fields["method"] = method
            if prop == "log_solubility":
                after = text[end:end + 60]
                pm, rng = PLUS_MINUS.match(after), RANGE.match(after)
                if pm:
                    half = _number(pm.group("pm"))
                    fields["interval"] = (fields["value"] - half, fields["value"] + half)
                    end += pm.end()
                elif rng:
                    fields["interval"] = tuple(sorted((_number(rng.group("lo")), _number(rng.group("hi")))))
                    end += rng.end()
            raw.append(((key.start(), end), fields))
            if method is not None and prop in VALID_METHODS:
                raw += _continuations(prop, text, end)

    for match in VALUE_THEN_LOGP_METHOD.finditer(text):
        method = METHOD_WORDS[re.sub(r"[-\s]+", " ", match.group("m").lower())]
        raw.append((match.span(), dict(prop="logp", value=_number(match.group("num")), method=method)))
        raw += _continuations("logp", text, match.end())

    lipinski_spans: list[tuple[int, int]] = []
    for pattern, value, short in ((FAIL_RE, 0.0, False), (PASS_RE, 1.0, False), (FAIL_SHORT, 0.0, True),
                                  (PASS_SHORT, 1.0, True), (FAIL_BY_METHOD, 0.0, True), (PASS_BY_METHOD, 1.0, True)):
        for match in pattern.finditer(text):
            if _overlaps(match.span(), lipinski_spans) or PARTIAL.match(text[match.end():]):
                continue
            if short and not NAMES_THE_RULE.search(text, *_sentence_bounds(text, match.start())):
                continue
            lipinski_spans.append(match.span())
            raw.append((match.span(), dict(prop="lipinski_pass", value=value,
                                           method=_lipinski_method(text, match.start()))))
    for match in VIOLATIONS_RE.finditer(text):
        count = _number(match.group("num") or match.group("num2"))
        raw.append((match.span(), dict(prop="lipinski_violations", value=count,
                                       method=_lipinski_method(text, match.start()))))
    for match in N_SIGNIFICANT.finditer(text):
        raw.append((match.span(), dict(prop="n_significant", value=_number(match.group("num")), subject=dataset)))

    out = Extraction()
    taken: list[tuple[int, int]] = []
    for span, fields in sorted(raw, key=lambda item: (item[0][0], -(item[0][1] - item[0][0]))):
        if _overlaps(span, taken):
            continue
        prop = fields.pop("prop")
        op = fields.pop("op", "=")
        who = fields.pop("subject", None) or (gene_at(span[0]) if prop in GENE_PROPS else subject_at(span[0]))
        if who is None:
            continue
        taken.append(span)
        out.claims.append(Claim(subject=who, prop=prop, op=op, span=span, text=text[span[0]:span[1]], **fields))

    for match in NUMBER_TOKEN.finditer(text):
        span = match.span()
        if _overlaps(span, taken):
            continue
        before = text[max(0, span[0] - 25):span[0]]
        line_start = text.rfind("\n", 0, span[0]) + 1
        is_list_marker = text[line_start:span[0]].strip() == "" and text[span[1]:span[1] + 1] in (".", ")")
        if THRESHOLD.search(before[-12:]) or IDENTIFIER_CONTEXT.search(before) or is_list_marker:
            continue
        out.unattributed.append((span[0], span[1], match.group()))
    return out
