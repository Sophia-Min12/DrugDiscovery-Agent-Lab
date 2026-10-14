"""Fetch reference properties for a fixed list of approved drugs from PubChem.

The output, data/approved_drugs.csv, is the only real-world chemistry data the repo
ships. PubChem's computed properties come from different programs than RDKit's
(XLogP3 rather than Wildman-Crippen, Cactvs rather than RDKit counting rules), which
is exactly what experiments/e1_provenance.py measures.

    python scripts/fetch_pubchem.py

Each row records the PubChem CID it came from, so every value is traceable.
"""
from __future__ import annotations

import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

# A spread of sizes on purpose: small analgesics, kinase inhibitors, and a handful of
# macrocycles and natural products that sit far outside the Rule of Five.
DRUGS = [
    "aspirin", "ibuprofen", "acetaminophen", "caffeine", "naproxen", "diclofenac",
    "celecoxib", "metformin", "nicotine", "morphine", "codeine", "diazepam",
    "fluoxetine", "sertraline", "haloperidol", "warfarin", "propranolol", "metoprolol",
    "atenolol", "amlodipine", "losartan", "lisinopril", "furosemide",
    "hydrochlorothiazide", "omeprazole", "cetirizine", "loratadine", "salbutamol",
    "theophylline", "levodopa", "ciprofloxacin", "amoxicillin", "chloramphenicol",
    "sildenafil", "tamoxifen", "imatinib", "gefitinib", "erlotinib", "osimertinib",
    "sorafenib", "atorvastatin", "simvastatin", "rosuvastatin", "prednisolone",
    "dexamethasone", "testosterone", "estradiol", "cyclosporine", "sirolimus",
    "erythromycin", "azithromycin", "vancomycin", "paclitaxel", "venetoclax",
    "remdesivir", "sofosbuvir",
]

PROPERTIES = [
    "SMILES", "MolecularFormula", "MolecularWeight", "XLogP", "TPSA",
    "HBondDonorCount", "HBondAcceptorCount", "RotatableBondCount",
]
BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/{}/JSON"
OUT = Path(__file__).resolve().parents[1] / "data" / "approved_drugs.csv"


def fetch(name: str) -> dict:
    url = BASE.format(urllib.parse.quote(name), ",".join(PROPERTIES))
    with urllib.request.urlopen(url, timeout=30) as response:
        rows = json.load(response)["PropertyTable"]["Properties"]
    # A name can map to several CIDs (salts, stereoisomers); PubChem lists the
    # parent compound first.
    return rows[0]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "name", "cid", "smiles", "formula", "pubchem_mw", "pubchem_xlogp3",
            "pubchem_tpsa", "pubchem_hbd", "pubchem_hba", "pubchem_rotb",
        ])
        for name in DRUGS:
            row = fetch(name)
            writer.writerow([
                name, row["CID"], row["SMILES"], row["MolecularFormula"],
                row["MolecularWeight"], row.get("XLogP", ""), row.get("TPSA", ""),
                row["HBondDonorCount"], row["HBondAcceptorCount"], row["RotatableBondCount"],
            ])
            print(f"{name:22s} CID {row['CID']}")
            time.sleep(0.25)  # PubChem asks for at most five requests per second


if __name__ == "__main__":
    main()
