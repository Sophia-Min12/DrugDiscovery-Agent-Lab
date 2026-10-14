"""Molecule -> graph tensors, and a message-passing network over them.

A molecule is atoms (nodes) and bonds (edges, stored once per direction). One layer:

    m_uv  = ReLU(W_m [h_u ; e_uv])           message from neighbour u along bond uv
    a_v   = sum over u in N(v) of m_uv        aggregation: index_add_ over the edge list
    h_v' = h_v + ReLU(W_u [h_v ; a_v])       residual update

and the readout concatenates sum- and mean-pooling over atoms. Summing over
neighbours and over atoms is what makes the prediction independent of atom order -
tests/test_gnn.py checks exactly that.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from rdkit import Chem
from torch import nn

from .. import chem

ELEMENTS = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
HYBRIDIZATIONS = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3]
BOND_TYPES = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]


def one_hot(value, choices) -> list[float]:
    """One-hot with a final 'other' slot, so an unseen value never crashes featurisation."""
    out = [0.0] * (len(choices) + 1)
    out[choices.index(value) if value in choices else len(choices)] = 1.0
    return out


def atom_features(atom: Chem.Atom) -> list[float]:
    return (one_hot(atom.GetSymbol(), ELEMENTS)
            + one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
            + one_hot(atom.GetFormalCharge(), [-1, 0, 1])
            + one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3])
            + one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
            + [float(atom.GetIsAromatic()), float(atom.IsInRing())])


def bond_features(bond: Chem.Bond) -> list[float]:
    return one_hot(bond.GetBondType(), BOND_TYPES)[:-1] + [float(bond.GetIsConjugated()), float(bond.IsInRing())]


ATOM_DIM = len(atom_features(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))
BOND_DIM = 6


@dataclass
class Graph:
    x: torch.Tensor           # [atoms, ATOM_DIM]
    edge_index: torch.Tensor  # [2, 2 * bonds]: source row, target row
    edge_attr: torch.Tensor   # [2 * bonds, BOND_DIM]


def featurize(smiles: str | Chem.Mol) -> Graph:
    mol = chem.parse(smiles) if isinstance(smiles, str) else smiles
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32)
    src, dst, attr = [], [], []
    for bond in mol.GetBonds():
        i, j, f = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_features(bond)
        src += [i, j]
        dst += [j, i]
        attr += [f, f]
    edge_index = torch.tensor([src, dst], dtype=torch.long).reshape(2, -1)
    edge_attr = torch.tensor(attr, dtype=torch.float32).reshape(-1, BOND_DIM)
    return Graph(x, edge_index, edge_attr)


@dataclass
class Batch:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    batch: torch.Tensor  # [atoms]: which molecule each atom belongs to
    size: int


def collate(graphs: list[Graph]) -> Batch:
    """Many molecules as one disconnected graph: offset each molecule's atom indices."""
    xs, edges, attrs, owners, offset = [], [], [], [], 0
    for k, g in enumerate(graphs):
        xs.append(g.x)
        edges.append(g.edge_index + offset)
        attrs.append(g.edge_attr)
        owners.append(torch.full((g.x.shape[0],), k, dtype=torch.long))
        offset += g.x.shape[0]
    return Batch(torch.cat(xs), torch.cat(edges, dim=1), torch.cat(attrs), torch.cat(owners), len(graphs))


class MessagePassing(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.message = nn.Linear(dim + BOND_DIM, dim)
        self.update = nn.Linear(2 * dim, dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        source, target = edge_index
        messages = torch.relu(self.message(torch.cat([h[source], edge_attr], dim=1)))
        aggregated = torch.zeros_like(h).index_add_(0, target, messages)
        return h + torch.relu(self.update(torch.cat([h, aggregated], dim=1)))


class SolubilityGNN(nn.Module):
    def __init__(self, hidden: int = 64, layers: int = 4):
        super().__init__()
        self.embed = nn.Linear(ATOM_DIM, hidden)
        self.layers = nn.ModuleList(MessagePassing(hidden) for _ in range(layers))
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, batch: Batch) -> torch.Tensor:
        h = torch.relu(self.embed(batch.x))
        for layer in self.layers:
            h = layer(h, batch.edge_index, batch.edge_attr)
        summed = torch.zeros(batch.size, h.shape[1]).index_add_(0, batch.batch, h)
        counts = torch.bincount(batch.batch, minlength=batch.size).clamp(min=1).unsqueeze(1)
        return self.head(torch.cat([summed, summed / counts], dim=1)).squeeze(1)
