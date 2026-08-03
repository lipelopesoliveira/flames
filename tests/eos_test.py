import pytest
from ase import units

from flames.eos import BaseEOS, PengRobinsonEOS

# Assuming your code is saved in eos.py
# from eos import BaseEOS, PengRobinsonEOS

# --- Shared Test Data: CO2 at 298.15 K and 1 atm
T_TEST = 298.15  # Kelvin
P_TEST = 101325.0  # Pascals (1 atm)
MOLAR_MASS_TEST = 44.01  # g/mol
TC_TEST = 304.1282  # Kelvin
PC_TEST = 7377300.0  # Pascals
OMEGA_TEST = 0.22394  # Acentric factor


# 1. Concrete subclass to test BaseEOS's default "return 1" behavior
class IdealGasEOS(BaseEOS):
    """
    Because BaseEOS has @abstractmethod decorators, it cannot be instantiated.
    This concrete class allows us to test the base class's default return values.
    """

    def get_compressibility(self) -> float:
        return super().get_compressibility()

    def get_fugacity_coefficient(self) -> float:
        return super().get_fugacity_coefficient()


@pytest.fixture
def base_eos() -> IdealGasEOS:
    """Fixture providing an IdealGasEOS instance to test BaseEOS logic."""
    return IdealGasEOS(
        temperature=T_TEST,
        pressure=P_TEST,
        molarMass=MOLAR_MASS_TEST,
    )


@pytest.fixture
def pr_eos() -> PengRobinsonEOS:
    """Fixture providing a PengRobinsonEOS instance."""
    # Notice how temperature, pressure, and molarMass are passed as kwargs
    # to satisfy the *args, **kwargs passed to super().__init__ in PR EOS.
    return PengRobinsonEOS(
        criticalTemperature=TC_TEST,
        criticalPressure=PC_TEST,
        acentricFactor=OMEGA_TEST,
        temperature=T_TEST,
        pressure=P_TEST,
        molarMass=MOLAR_MASS_TEST,
    )


# --- Tests for BaseEOS (via IdealGasEOS) ---


def test_base_eos_initialization(base_eos):
    """Test that properties are properly assigned in the base class."""
    assert base_eos.T == T_TEST
    assert base_eos.P == P_TEST
    assert base_eos.molar_mass == MOLAR_MASS_TEST
    assert hasattr(base_eos, "R")
    assert base_eos.R > 0  # Universal gas constant should be populated


def test_base_eos_abstract_defaults(base_eos):
    """Test that BaseEOS default methods return 1 as defined."""
    assert base_eos.get_compressibility() == 1.0
    assert base_eos.get_fugacity_coefficient() == 1.0


def test_base_eos_density(base_eos):
    """Test bulk density calculation with Ideal Gas assumption (Z=1)."""
    expected_molar_volume = base_eos.R * T_TEST * 1.0 / P_TEST
    expected_density = (1e-3 * MOLAR_MASS_TEST) / expected_molar_volume * units.mol

    assert base_eos.get_bulk_phase_density() == pytest.approx(expected_density, rel=1e-5)


def test_base_eos_molar_density(base_eos):
    """Test molar density calculation with Ideal Gas assumption (Z=1)."""
    expected_molar_volume = base_eos.R * T_TEST * 1.0 / P_TEST
    expected_molar_density = (1.0 / expected_molar_volume) * units.mol

    assert base_eos.get_bulk_phase_molar_density() == pytest.approx(
        expected_molar_density, rel=1e-5
    )


# --- Tests for PengRobinsonEOS ---


def test_pr_eos_initialization(pr_eos):
    """Test that PR specific constants and base kwargs are initialized correctly."""
    assert pr_eos.Tc == TC_TEST
    assert pr_eos.Pc == PC_TEST
    assert pr_eos.omega == OMEGA_TEST
    assert pr_eos.T == T_TEST
    assert pr_eos.P == P_TEST
    assert pr_eos.reducedTemperature == T_TEST / TC_TEST

    # Internal PR Constants
    assert pr_eos.a > 0
    assert pr_eos.b > 0
    assert pr_eos.kappa > 0
    assert pr_eos.alpha > 0


def test_pr_eos_parameters(pr_eos):
    """Test calculation of A and B dimensionless parameters."""
    A, B = pr_eos.calculate_eos_parameters()
    assert A > 0
    assert B > 0
    assert isinstance(A, float)
    assert isinstance(B, float)


def test_pr_eos_compressibility(pr_eos):
    """
    Test PR compressibility.
    For CO2 gas at 1 atm and 298K, Z should be real, a float, and slightly less than 1.
    """
    z = pr_eos.get_compressibility()
    assert isinstance(z, float)
    assert 0.98 < z < 1.0


def test_pr_eos_fugacity_coefficient(pr_eos):
    """
    Test PR fugacity coefficient.
    For CO2 gas at 1 atm and 298K, phi should be slightly less than 1.
    """
    phi = pr_eos.get_fugacity_coefficient()
    assert isinstance(phi, float)
    assert 0.98 < phi < 1.0


def test_pr_eos_density_vs_ideal(base_eos, pr_eos):
    """
    Compare PR density to Ideal Gas density.
    Because Z < 1 for CO2 at 1 atm (attractive forces), the real gas occupies
    less volume than an ideal gas, making its density slightly higher.
    """
    ideal_density = base_eos.get_bulk_phase_density()
    pr_density = pr_eos.get_bulk_phase_density()

    assert pr_density > ideal_density
