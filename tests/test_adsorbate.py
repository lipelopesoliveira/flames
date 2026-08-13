from pathlib import Path

import pytest
from ase.atoms import Atoms
from ase.build import molecule
from ase.io import read

from flames.adsorbate import Adsorbate
from flames.eos import PengRobinsonEOS
from flames.move_weights import MoveWeights


@pytest.fixture
def test_dir() -> Path:
    """Returns the directory containing this test file."""
    return Path(__file__).parent


@pytest.fixture
def h2o_path(test_dir) -> Path:
    """Path to the H2O.xyz file inside the adsorbates folder."""
    return test_dir / "adsorbates" / "H2O.xyz"


@pytest.fixture
def co2_path(test_dir) -> Path:
    """Path to the CO2.xyz file inside the adsorbates folder."""
    return test_dir / "adsorbates" / "CO2.xyz"


@pytest.fixture
def dummy_water():
    """Returns a simple ASE Atoms object for water."""
    return molecule("H2O")


@pytest.fixture
def dummy_co2():
    """Returns a simple ASE Atoms object for carbon dioxide."""
    return molecule("CO2")


def test_adsorbate_initialization_h2o_ase(dummy_water):

    adsorbate = Adsorbate("H2O", structure=dummy_water)

    # Test basic attributes
    assert adsorbate.name == "H2O"
    assert isinstance(adsorbate.structure, Atoms)
    assert adsorbate.structure == molecule("H2O")
    assert adsorbate.molar_mass == pytest.approx(18.015, 0.1)


def test_adsorbate_initialization_co2_ase(dummy_co2):

    adsorbate = Adsorbate("CO2", structure=dummy_co2)

    # Test basic attributes
    assert adsorbate.name == "CO2"
    assert isinstance(adsorbate.structure, Atoms)
    assert adsorbate.structure == molecule("CO2")
    assert adsorbate.molar_mass == pytest.approx(44.01, 0.1)


def test_adsorbate_initialization_h2o_file(h2o_path):

    adsorbate = Adsorbate("H2O", structure=str(h2o_path))

    # Test basic attributes
    assert adsorbate.name == "H2O"
    assert isinstance(adsorbate.structure, Atoms)
    assert adsorbate.structure == read(str(h2o_path))
    assert adsorbate.molar_mass == pytest.approx(18.015, 0.1)


def test_adsorbate_initialization_co2_file(co2_path):

    adsorbate = Adsorbate("CO2", structure=str(co2_path))

    # Test basic attributes
    assert adsorbate.name == "CO2"
    assert isinstance(adsorbate.structure, Atoms)
    assert adsorbate.structure == read(str(co2_path))
    assert adsorbate.molar_mass == pytest.approx(44.01, 0.1)


def test_molar_mass_explicit_override(dummy_water):
    adsorbate = Adsorbate("H2O", structure=dummy_water)
    adsorbate.molar_mass = 20.0  # Explicitly override molar mass
    assert adsorbate.molar_mass == 20.0  # Should reflect the overridden


def test_mol_fraction_validation(dummy_water):
    """Test that mole fraction strictly stays between 0 and 1."""
    adsorbate = Adsorbate("H2O", structure=dummy_water, mol_fraction=0.5)

    with pytest.raises(ValueError, match="Mole fraction must be between 0 and 1."):
        adsorbate.mol_fraction = 1.5

    with pytest.raises(ValueError, match="Mole fraction must be between 0 and 1."):
        adsorbate.mol_fraction = -0.1

    adsorbate.mol_fraction = 0.8
    assert adsorbate.mol_fraction == 0.8


def test_eos_dict_initialization(dummy_water):
    """Test that passing a dict to EOS properly routes to PengRobinsonEOS."""
    eos_params = {
        "criticalTemperature": 647.14,
        "criticalPressure": 22.064e6,
        "acentricFactor": 0.344,
    }
    adsorbate = Adsorbate("H2O", structure=dummy_water, eos=eos_params)

    assert adsorbate.eos is not None
    assert isinstance(adsorbate.eos, PengRobinsonEOS)
    assert adsorbate.eos.Tc == 647.14
    assert adsorbate.eos.Pc == 22.064e6
    assert adsorbate.eos.omega == 0.344
    assert adsorbate.eos.molar_mass == pytest.approx(
        18.015, 0.1
    )  # Should match the molar mass of H2O
    assert adsorbate.eos == PengRobinsonEOS(**eos_params, molar_mass=adsorbate.molar_mass)


def test_set_structure_raises_error(dummy_water):
    """Test that EOS dict fails if there is no structure to get the molar mass from."""
    adsorbate = Adsorbate("H2O", structure=dummy_water)
    
    with pytest.raises(ValueError, match="Structure must be an ASE Atoms object"):
        adsorbate.structure = None


def test_weights_dict_initialization(dummy_water):
    """Test that passing a dict sets up MoveWeights."""
    move_probs = {"translation": 0.4, "rotation": 0.2, "insertion": 0.2, "deletion": 0.2}
    adsorbate = Adsorbate("H2O", structure=dummy_water, weights=move_probs)

    assert adsorbate.weights is not None
    assert isinstance(adsorbate.weights, MoveWeights)
    assert adsorbate.weights == MoveWeights(**move_probs)
