"""The GNN: invariances it must have by construction, and the deployed model's contract."""
import math
import random

import pytest

torch = pytest.importorskip("torch")
from rdkit import Chem  # noqa: E402

from drugagent.claims import Claim, Verdict  # noqa: E402
from drugagent.gnn import data, predict  # noqa: E402
from drugagent.gnn.model import BOND_DIM, SolubilityGNN, collate, featurize  # noqa: E402
from drugagent.gnn.train import conformal_quantile  # noqa: E402
from drugagent.verifier import Verifier, chem_source  # noqa: E402

IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"


def model():
    torch.manual_seed(0)
    return SolubilityGNN().eval()


def test_graph_shapes():
    water, ethanol = featurize("O"), featurize("CCO")
    assert water.x.shape[0] == 1 and water.edge_index.shape == (2, 0)
    assert ethanol.x.shape[0] == 3 and ethanol.edge_index.shape == (2, 4)  # 2 bonds, both directions
    assert ethanol.edge_attr.shape == (4, BOND_DIM)


def test_collate_offsets_atom_indices():
    batch = collate([featurize("CCO"), featurize("CC")])
    assert batch.batch.tolist() == [0, 0, 0, 1, 1]
    assert batch.edge_index.max().item() == 4 and batch.edge_index[:, 4:].min().item() == 3


def test_prediction_does_not_depend_on_atom_order():
    mol = Chem.MolFromSmiles(IBUPROFEN)
    order = list(range(mol.GetNumAtoms()))
    random.Random(1).shuffle(order)
    m = model()
    with torch.no_grad():
        a = m(collate([featurize(mol)])).item()
        b = m(collate([featurize(Chem.RenumberAtoms(mol, order))])).item()
    assert a == pytest.approx(b, abs=1e-5)


def test_prediction_does_not_depend_on_batch_neighbours():
    m = model()
    with torch.no_grad():
        alone = m(collate([featurize(IBUPROFEN)])).item()
        batched = m(collate([featurize("O"), featurize(IBUPROFEN), featurize("c1ccccc1")]))[1].item()
    assert alone == pytest.approx(batched, abs=1e-5)


def test_gradients_reach_every_parameter_and_a_tiny_set_can_be_memorised():
    smiles = ["CCO", "CCCCO", "c1ccccc1", "CC(=O)O", "CCN", "ClC(Cl)Cl", "OCC(O)CO", "CCCCCCCC"]
    target = torch.linspace(-2, 2, len(smiles))
    m, batch = SolubilityGNN(hidden=32, layers=2), collate([featurize(s) for s in smiles])
    optimiser = torch.optim.Adam(m.parameters(), lr=1e-2)
    torch.manual_seed(0)
    for _ in range(300):
        loss = torch.mean((m(batch) - target) ** 2)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    assert all(p.grad is not None for p in m.parameters())
    assert loss.item() < 1e-2


def test_conformal_rank_is_the_textbook_one():
    scores = list(range(1, 20))  # 19 calibration scores; ceil(20 * 0.95) = 19 -> the largest
    assert conformal_quantile(torch.tensor(scores).numpy(), 0.05) == 19
    assert conformal_quantile(torch.arange(1, 100).numpy(), 0.10) == math.ceil(100 * 0.9)


def test_scaffold_split_never_shares_a_scaffold():
    smiles = ["c1ccccc1C", "c1ccccc1CC", "c1ccccc1O", "C1CCCCC1N", "C1CCCCC1O", "c1ccncc1", "CCCC", "CCO", "c1ccoc1"]
    train, valid, test = data.scaffold_split(smiles, fractions=(0.5, 0.25, 0.25))
    from drugagent.chem import murcko_scaffold
    scaffolds = lambda idx: {murcko_scaffold(smiles[i]) for i in idx}  # noqa: E731
    assert not scaffolds(train) & scaffolds(test) and not scaffolds(train) & scaffolds(valid)
    assert sorted(train + valid + test) == list(range(len(smiles)))


deployed = pytest.mark.skipif(predict.load() is None, reason="models/esol_gnn not trained")


@deployed
def test_deployed_prediction_carries_interval_and_domain():
    p = predict.predict("ibuprofen")
    low, high = p["interval_95"]
    assert low < p["log_solubility"] < high
    assert 0.0 <= p["nearest_training_similarity"] <= 1.0
    assert p["in_domain"] == (p["nearest_training_similarity"] >= 0.3)


@deployed
def test_verifier_treats_a_prediction_as_a_claim_with_an_interval():
    v = Verifier([chem_source, predict.gnn_source])
    p = predict.predict("ibuprofen")
    mu, interval = p["log_solubility"], tuple(p["interval_95"])
    assert v.check(Claim("ibuprofen", "log_solubility", round(mu, 2), interval=interval)).verdict is Verdict.SUPPORTED
    assert v.check(Claim("ibuprofen", "log_solubility", round(mu, 2))).verdict is Verdict.UNQUALIFIED
    assert v.check(Claim("ibuprofen", "log_solubility", interval[1] + 1.0)).verdict is Verdict.CONTRADICTED
