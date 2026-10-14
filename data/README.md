# Data

| file | what | source | in git |
|---|---|---|---|
| `approved_drugs.csv` | 56 approved drugs: structure, formula, and PubChem's computed MW, XLogP3, TPSA, H-bond donor/acceptor and rotatable-bond counts | [PubChem PUG REST](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest), fetched by `scripts/fetch_pubchem.py`; every row keeps its CID | yes |
| `cache/delaney-processed.csv` | ESOL: 1,128 molecules with measured aqueous solubility (Delaney, *J. Chem. Inf. Comput. Sci.* 2004, 44, 1000) | MoleculeNet mirror, downloaded on first use by `drugagent/gnn/data.py` and checked against a pinned SHA-256 | no |

PubChem's property columns are kept as a *second method* next to RDKit's, not as
ground truth: XLogP3 is not Crippen logP, and Cactvs's acceptor count is not
Lipinski's. Experiment E1 measures how far apart they are.

The RNA-seq counts are not data at all: `drugagent/omics.py` simulates them, with the
differentially expressed genes planted by the generator, so every false discovery is
known.
