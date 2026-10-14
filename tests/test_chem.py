"""Chemistry tools, tested against answers known independently of RDKit."""
import pytest

from drugagent import chem, drugs

ASPIRIN = "CC(=O)OC1=CC=CC=C1C(=O)O"


def test_aspirin_formula_and_weight_from_atomic_masses():
    # C9H8O4 with IUPAC standard atomic weights: 9(12.011) + 8(1.008) + 4(15.999)
    assert chem.descriptors(ASPIRIN)["formula"] == "C9H8O4"
    assert chem.compute("mw", ASPIRIN) == pytest.approx(9 * 12.011 + 8 * 1.008 + 4 * 15.999, abs=0.01)


def test_ethanol_tpsa_is_one_hydroxyl_contribution():
    # Ertl (2000): an aliphatic OH contributes 20.23 A^2, and ethanol has nothing else polar.
    assert chem.compute("tpsa", "CCO") == pytest.approx(20.23, abs=0.01)
    assert chem.compute("hbd", "CCO") == 1 and chem.compute("hba", "CCO") == 1


def test_counts_on_hydrocarbons():
    assert chem.compute("hbd", "c1ccccc1") == 0 and chem.compute("hba", "c1ccccc1") == 0
    assert chem.compute("aromatic_rings", "c1ccccc1") == 1
    assert chem.compute("fsp3", "c1ccccc1") == 0.0
    assert chem.compute("fsp3", "C1CCCCC1") == 1.0
    assert chem.compute("heavy_atoms", "C1CCCCC1") == 6


def test_invalid_smiles_raises_instead_of_returning_none():
    with pytest.raises(ValueError):
        chem.parse("C1CC")  # unclosed ring
    assert not chem.is_valid_smiles("not a molecule")


def test_two_writings_of_one_molecule_share_a_canonical_form():
    assert chem.canonical("OCC") == chem.canonical("CCO")
    assert chem.tanimoto("OCC", "CCO") == 1.0


def test_hba_definitions_disagree_on_aspirin():
    # Aspirin has four oxygens. Lipinski's rule counts all four; RDKit's SMARTS-based
    # count skips the ester's alkoxy oxygen. Both are "the number of H-bond acceptors".
    methods = chem.compute_all_methods("hba", ASPIRIN)
    assert methods == {"lipinski": 4, "rdkit": 3}


def test_tanimoto_is_symmetric_and_bounded():
    a, b = ASPIRIN, "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
    assert chem.tanimoto(a, b) == chem.tanimoto(b, a)
    assert 0.0 <= chem.tanimoto(a, b) < 1.0


def test_nearest_finds_the_query_itself_first():
    neighbours = chem.nearest(ASPIRIN, drugs.library(), k=3)
    assert neighbours[0] == ("aspirin", 1.0)
    assert neighbours[1][1] <= neighbours[0][1]


def test_murcko_scaffold():
    assert chem.murcko_scaffold("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O") == "c1ccccc1"
    assert chem.murcko_scaffold("CCCCCC") == ""


def test_lipinski_on_both_sides_of_the_rule():
    assert chem.lipinski(ASPIRIN)["n_violations"] == 0
    cyclosporine = drugs.resolve("cyclosporine")
    result = chem.lipinski(cyclosporine)
    assert not result["passes"] and "mw" in result["violations"]


def test_rhodanine_raises_a_pains_alert():
    alerts = chem.structural_alerts("O=C1NC(=S)SC1=Cc1ccccc1")
    assert any("rhod" in alert.lower() for alert in alerts)


def test_resolver_accepts_names_synonyms_and_smiles():
    assert drugs.resolve("Paracetamol") == drugs.resolve("acetaminophen")
    assert drugs.resolve("CCO") == "CCO"
    assert drugs.resolve("unobtainium") is None


def test_reference_table_is_internally_consistent():
    # Every PubChem row must parse, and RDKit's average MW must match PubChem's
    # to rounding - the one property where the two programs have no reason to differ.
    for drug in drugs.table().values():
        assert abs(chem.compute("mw", drug.smiles) - drug.reference[("mw", "pubchem")]) < 0.1, drug.name
