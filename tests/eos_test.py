import pytest

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

COMPRESSIBILITY_RASPA = {
    (250, 100000000): 1.6106631629,
    (250, 10000000): 0.1897261743,
    (250, 1000000): 0.9020656185,
    (250, 100000): 0.9908049323,
    (250, 500000000): 7.0710324702,
    (250, 50000000): 0.8552847391,
    (250, 5000000): 0.0971565493,
    (250, 500000): 0.9528113012,
    (280, 100000000): 1.5092229709,
    (280, 10000000): 0.2013847898,
    (280, 1000000): 0.9313689577,
    (280, 100000): 0.9933843885,
    (280, 500000000): 6.4057345212,
    (280, 50000000): 0.8231883674,
    (280, 5000000): 0.108854879,
    (280, 500000): 0.9664023488,
    (298, 100000000): 1.4631578348,
    (298, 10000000): 0.2272577531,
    (298, 1000000): 0.943686222,
    (298, 100000): 0.9945133087,
    (298, 500000000): 6.0725331127,
    (298, 50000000): 0.8145121154,
    (298, 5000000): 0.6596598902,
    (298, 500000): 0.9722583989,
    (323, 100000000): 1.4136623984,
    (323, 10000000): 0.4344461506,
    (323, 1000000): 0.9566144963,
    (323, 100000): 0.9957260179,
    (323, 500000000): 5.6731688092,
    (323, 50000000): 0.8142576612,
    (323, 5000000): 0.7636006677,
    (323, 500000): 0.9784903519,
    (353, 100000000): 1.3719132642,
    (353, 10000000): 0.665665114,
    (353, 1000000): 0.9677730258,
    (353, 100000): 0.9967951576,
    (353, 500000000): 5.2709642579,
    (353, 50000000): 0.8298037395,
    (353, 5000000): 0.8348881131,
    (353, 500000): 0.9839362997,
    (400, 100000000): 1.3342088691,
    (400, 10000000): 0.8109759155,
    (400, 1000000): 0.9793379863,
    (400, 100000): 0.9979246421,
    (400, 500000000): 4.7662441406,
    (400, 50000000): 0.8784412001,
    (400, 5000000): 0.8994890171,
    (400, 500000): 0.989642907,
}

FUGACITY_RASPA = {
    (250, 100000000): 0.0817247912,
    (250, 10000000): 0.1747524418,
    (250, 1000000): 0.9098532491,
    (250, 100000): 0.9908749551,
    (250, 500000000): 6.0903197424,
    (250, 50000000): 0.0714515133,
    (250, 5000000): 0.3175238947,
    (250, 500000): 0.954639074,
    (280, 100000000): 0.1546170157,
    (280, 10000000): 0.3464033336,
    (280, 1000000): 0.9349892451,
    (280, 100000): 0.9934185105,
    (280, 500000000): 6.9782373788,
    (280, 50000000): 0.1411649648,
    (280, 5000000): 0.6241794704,
    (280, 500000): 0.9672767924,
    (298, 100000000): 0.2090806949,
    (298, 10000000): 0.4681289307,
    (298, 1000000): 0.9460174803,
    (298, 100000): 0.9945357133,
    (298, 500000000): 7.3678894896,
    (298, 50000000): 0.1942655305,
    (298, 5000000): 0.7409245394,
    (298, 500000): 0.9728280185,
    (323, 100000000): 0.2947090003,
    (323, 10000000): 0.6177776407,
    (323, 1000000): 0.9578899337,
    (323, 100000): 0.9957385348,
    (323, 500000000): 7.7534204589,
    (323, 50000000): 0.2781490693,
    (323, 5000000): 0.7995789496,
    (323, 500000): 0.9788058097,
    (353, 100000000): 0.4066586015,
    (353, 10000000): 0.7193552966,
    (353, 1000000): 0.9683830469,
    (353, 100000): 0.9968012668,
    (353, 500000000): 8.0179338341,
    (353, 50000000): 0.386831656,
    (353, 5000000): 0.8498954121,
    (353, 500000): 0.9840889368,
    (400, 100000000): 0.5848829598,
    (400, 10000000): 0.8186278954,
    (400, 1000000): 0.9794994087,
    (400, 100000): 0.9979263174,
    (400, 500000000): 8.1150262773,
    (400, 50000000): 0.5537583662,
    (400, 5000000): 0.9027273983,
    (400, 500000): 0.9896841213,
}

DENSITY_RASPA = {
    (250.0, 100000.0): 2.1368988271,
    (250.0, 1000000.0): 23.4711295326,
    (250.0, 10000000.0): 1115.9503454489,
    (250.0, 100000000.0): 1314.5205940771,
    (250.0, 500000.0): 11.1105414848,
    (250.0, 5000000.0): 1089.6073982699,
    (250.0, 50000000.0): 1237.7456307428,
    (250.0, 500000000.0): 1497.1292429576,
    (280.0, 100000.0): 1.9029911446,
    (280.0, 1000000.0): 20.2970227737,
    (280.0, 10000000.0): 938.7013269163,
    (280.0, 100000000.0): 1252.56620853,
    (280.0, 500000.0): 9.7806141341,
    (280.0, 5000000.0): 868.3127997715,
    (280.0, 50000000.0): 1148.2193925781,
    (280.0, 500000000.0): 1475.554199299,
    (298.0, 100000.0): 1.7860156739,
    (298.0, 1000000.0): 18.822107558,
    (298.0, 10000000.0): 781.5866932186,
    (298.0, 100000000.0): 1213.9608694363,
    (298.0, 500000.0): 9.1344870827,
    (298.0, 5000000.0): 134.6312231124,
    (298.0, 50000000.0): 1090.3560079856,
    (298.0, 500000000.0): 1462.5003472706,
    (323.0, 100000.0): 1.6457723125,
    (323.0, 1000000.0): 17.130602948,
    (323.0, 10000000.0): 377.2017104234,
    (323.0, 100000000.0): 1159.2147551458,
    (323.0, 500000.0): 8.3738092452,
    (323.0, 5000000.0): 107.3033576531,
    (323.0, 50000000.0): 1006.2774899272,
    (323.0, 500000000.0): 1444.2883387617,
    (353.0, 100000.0): 1.5042897778,
    (353.0, 1000000.0): 15.4940128126,
    (353.0, 10000000.0): 225.2587276565,
    (353.0, 100000000.0): 1092.9763603162,
    (353.0, 500000.0): 7.6197451329,
    (353.0, 5000000.0): 89.8005818199,
    (353.0, 50000000.0): 903.5080795132,
    (353.0, 500000000.0): 1422.3856326593,
    (400.0, 100000.0): 1.3260331795,
    (400.0, 1000000.0): 13.5119969271,
    (400.0, 10000000.0): 163.1714531581,
    (400.0, 100000000.0): 991.8096160281,
    (400.0, 500000.0): 6.6856498277,
    (400.0, 5000000.0): 73.557384303,
    (400.0, 50000000.0): 753.1984986193,
    (400.0, 500000000.0): 1388.1802390905,
}


# 1. Concrete subclass to test BaseEOS's default "return 1" behavior
class IdealGasEOS(BaseEOS):
    """
    Because BaseEOS has @abstractmethod decorators, it cannot be instantiated.
    This concrete class allows us to test the base class's default return values.
    """

    def get_compressibility(self, temperature: float, pressure: float) -> float:
        return super().get_compressibility(temperature, pressure)

    def get_fugacity_coefficient(self, temperature: float, pressure: float) -> float:
        return super().get_fugacity_coefficient(temperature, pressure)


@pytest.fixture
def base_eos() -> IdealGasEOS:
    """Fixture providing an IdealGasEOS instance to test BaseEOS logic."""
    return IdealGasEOS(
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
        molarMass=MOLAR_MASS_TEST,
    )


# --- Tests for BaseEOS (via IdealGasEOS) ---


def test_base_eos_initialization(base_eos):
    """Test that properties are properly assigned in the base class."""
    assert base_eos.molar_mass == MOLAR_MASS_TEST
    assert hasattr(base_eos, "R")
    assert base_eos.R > 0  # Universal gas constant should be populated


def test_base_eos_abstract_defaults(base_eos):
    """Test that BaseEOS default methods return 1 as defined."""
    assert base_eos.get_compressibility(T_TEST, P_TEST) == 1.0
    assert base_eos.get_fugacity_coefficient(T_TEST, P_TEST) == 1.0


def test_base_eos_density(base_eos):
    """Test bulk density calculation with Ideal Gas assumption (Z=1)."""
    expected_molar_volume = base_eos.R * T_TEST * 1.0 / P_TEST
    expected_density = (1e-3 * MOLAR_MASS_TEST) / expected_molar_volume

    assert base_eos.get_bulk_phase_density(T_TEST, P_TEST) == pytest.approx(expected_density, rel=1e-5)


def test_base_eos_molar_density(base_eos):
    """Test molar density calculation with Ideal Gas assumption (Z=1)."""
    expected_molar_volume = base_eos.R * T_TEST * 1.0 / P_TEST
    expected_molar_density = 1.0 / expected_molar_volume

    assert base_eos.get_bulk_phase_molar_density(T_TEST, P_TEST) == pytest.approx(
        expected_molar_density, rel=1e-5
    )


# --- Tests for PengRobinsonEOS ---


def test_pr_eos_initialization(pr_eos):
    """Test that PR specific constants and base kwargs are initialized correctly."""
    assert pr_eos.Tc == TC_TEST
    assert pr_eos.Pc == PC_TEST
    assert pr_eos.omega == OMEGA_TEST
    assert pr_eos.reducedTemperature(T_TEST) == T_TEST / TC_TEST

    # Internal PR Constants
    assert pr_eos.a > 0
    assert pr_eos.b > 0
    assert pr_eos.kappa > 0
    assert pr_eos.alpha(T_TEST) > 0


def test_pr_eos_parameters(pr_eos):
    """Test calculation of A and B dimensionless parameters."""
    A, B = pr_eos.calculate_eos_parameters(T_TEST, P_TEST)
    assert A > 0
    assert B > 0
    assert isinstance(A, float)
    assert isinstance(B, float)


def test_pr_eos_compressibility(pr_eos):
    """
    Test PR compressibility.
    For CO2 gas at 1 atm and 298K, Z should be real, a float, and slightly less than 1.
    """
    z = pr_eos.get_compressibility(T_TEST, P_TEST)
    assert isinstance(z, float)
    assert 0.98 < z < 1.0


def test_pr_eos_fugacity_coefficient(pr_eos):
    """
    Test PR fugacity coefficient.
    For CO2 gas at 1 atm and 298K, phi should be slightly less than 1.
    """
    phi = pr_eos.get_fugacity_coefficient(T_TEST, P_TEST)
    assert isinstance(phi, float)
    assert 0.98 < phi < 1.0


def test_pr_eos_density_vs_ideal(base_eos, pr_eos):
    """
    Compare PR density to Ideal Gas density.
    Because Z < 1 for CO2 at 1 atm (attractive forces), the real gas occupies
    less volume than an ideal gas, making its density slightly higher.
    """
    ideal_density = base_eos.get_bulk_phase_density(T_TEST, P_TEST)
    pr_density = pr_eos.get_bulk_phase_density(T_TEST, P_TEST)

    assert pr_density > ideal_density


def test_compressibility_against_raspa(pr_eos):
    """
    Test PR EOS compressibility factor against RASPA reference values.
    """
    for (T, P), Z_ref in COMPRESSIBILITY_RASPA.items():
        Z_pr = pr_eos.get_compressibility(T, P)
        assert Z_pr == pytest.approx(
            Z_ref, rel=1e-3
        ), f"Failed for T={T}, P={P}: Z_pr={Z_pr}, Z_ref={Z_ref}"


def test_fugacity_coefficient_against_raspa(pr_eos):
    """
    Test PR EOS fugacity coefficient against RASPA reference values.
    """
    for (T, P), phi_ref in FUGACITY_RASPA.items():
        phi_pr = pr_eos.get_fugacity_coefficient(T, P)
        assert phi_pr == pytest.approx(
            phi_ref, rel=1e-3
        ), f"Failed for T={T}, P={P}: phi_pr={phi_pr}, phi_ref={phi_ref}"


def test_density_against_raspa(pr_eos):
    """
    Test PR EOS density against RASPA reference values.
    """
    for (T, P), density_ref in DENSITY_RASPA.items():
        density_pr = pr_eos.get_bulk_phase_density(T, P)

        assert density_pr == pytest.approx(
            density_ref, rel=1e-3
        ), f"Failed for T={T}, P={P}: density_pr={density_pr}, density_ref={density_ref}"
