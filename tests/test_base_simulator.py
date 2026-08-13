import json
from pathlib import Path

import pytest
from ase.data import vdw_radii
from ase.io import read

from flames.adsorbate import Adsorbate
from flames.base_simulator import BaseSimulator
from flames.calculators.lennard_jones import CustomLennardJones
from flames.move_weights import MoveWeights


@pytest.fixture
def test_dir() -> Path:
    """Returns the directory containing this test file."""
    return Path(__file__).parent


@pytest.fixture
def mof_path(test_dir) -> Path:
    """Path to the MOF.cif file inside the framework folder."""
    return test_dir / "mofs" / "MgMOF-74.cif"


@pytest.fixture
def h2o_path(test_dir) -> Path:
    """Path to the H2O.xyz file inside the adsorbates folder."""
    return test_dir / "adsorbates" / "H2O.xyz"


@pytest.fixture
def co2_path(test_dir) -> Path:
    """Path to the CO2.xyz file inside the adsorbates folder."""
    return test_dir / "adsorbates" / "CO2.xyz"


@pytest.fixture
def lj_calculator() -> CustomLennardJones:
    with open(Path(__file__).parent.parent / "flames" / "data" / "UFF_lj_params.json", "r") as f:
        uff_lj_params = json.loads(f.read())

    return CustomLennardJones(uff_lj_params, vdw_cutoff=12.5)


@pytest.mark.filterwarnings("ignore:crystal system")
def test_base_simulator_initialization_single_adsorbates(co2_path, mof_path, lj_calculator):

    # Create Adsorbate instances
    co2_adsorbate = Adsorbate("CO2", structure=str(co2_path), mol_fraction=1)

    # Initialize BaseSimulator with a single adsorbate
    simulator = BaseSimulator(
        model=lj_calculator,
        framework_atoms=read(str(mof_path)),
        adsorbates=co2_adsorbate,
        temperature=300.0,
        pressure=1e5,
        device="cpu",
        vdw_radii=vdw_radii,
    )

    # Check that the adsorbates attribute is a list containing the single adsorbate
    assert isinstance(simulator.adsorbates, list)
    assert len(simulator.adsorbates) == 1
    assert simulator.adsorbates[0].name == "CO2"
    assert simulator.adsorbates[0].mol_fraction == 1
    assert simulator.adsorbates[0].molar_mass == 44.009
    assert simulator.n_adsorbate_atoms == {"CO2": 3}
    assert isinstance(simulator.adsorbates[0].weights, MoveWeights)
    assert simulator.get_framework_mass() == pytest.approx(7.254469967765757e-24)


@pytest.mark.filterwarnings("ignore:crystal system")
def test_base_simulator_initialization_multiple_adsorbates(
    co2_path, h2o_path, mof_path, lj_calculator
):

    # Create Adsorbate instances
    co2_adsorbate = Adsorbate("CO2", structure=str(co2_path), mol_fraction=0.5)
    h2o_adsorbate = Adsorbate("H2O", structure=str(h2o_path), mol_fraction=0.5)

    # Initialize BaseSimulator with a single adsorbate
    simulator = BaseSimulator(
        model=lj_calculator,
        framework_atoms=read(str(mof_path)),
        adsorbates=[h2o_adsorbate, co2_adsorbate],
        temperature=300.0,
        pressure=1e5,
        device="cpu",
        vdw_radii=vdw_radii,
    )

    # Check that the adsorbates attribute is a list containing the single adsorbate
    assert isinstance(simulator.adsorbates, list)
    assert len(simulator.adsorbates) == 2
    assert simulator.adsorbates[0].name == "H2O"
    assert simulator.adsorbates[0].mol_fraction == 0.5
    assert simulator.adsorbates[0].molar_mass == 18.015
    assert simulator.adsorbates[1].name == "CO2"
    assert simulator.adsorbates[1].mol_fraction == 0.5
    assert simulator.adsorbates[1].molar_mass == 44.009
    assert simulator.n_adsorbate_atoms == {"CO2": 3, "H2O": 3}
    assert isinstance(simulator.adsorbates[0].weights, MoveWeights)
    assert simulator.get_framework_mass() == pytest.approx(7.254469967765757e-24)
