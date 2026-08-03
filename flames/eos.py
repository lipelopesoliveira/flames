from abc import ABC, abstractmethod

import numpy as np
from ase import units


class BaseEOS(ABC):
    """
    Abstract base class for Equations of State.
    Contains generic thermodynamic properties and relationships.
    """

    def __init__(
        self,
        temperature: float,
        pressure: float,
        molarMass: float,
    ) -> None:
        self.T = temperature
        self.P = pressure
        self.molar_mass = molarMass

        # Universal gas constant in J/(mol*K)
        self.R = units.kB / units.J * units.mol

    @abstractmethod
    def get_compressibility(self) -> float:
        """Calculate the compressibility factor Z."""
        return 1

    @abstractmethod
    def get_fugacity_coefficient(self) -> float:
        """Calculate the fugacity coefficient phi."""
        return 1

    def get_bulk_phase_density(self) -> float:
        """
        Calculate the bulk phase density using the compressibility factor.
        rho = MM / Vm (kg/m^3)
        """
        Z = self.get_compressibility()
        molar_volume = self.R * self.T * Z / self.P
        density = 1e-3 * self.molar_mass / molar_volume * units.mol
        return float(density)

    def get_bulk_phase_molar_density(self) -> float:
        """
        Calculate the equivalent bulk phase number of molecules per cubic meter.
        (mol/m^3)
        """
        Z = self.get_compressibility()
        molar_volume = self.R * self.T * Z / self.P
        molar_density = 1 / molar_volume * units.mol
        return float(molar_density)


class PengRobinsonEOS(BaseEOS):
    """
    Peng-Robinson Equation of State implementation.

    This class calculates the compressibility factor and fugacity coefficient
    based on the Peng-Robinson EOS, which is widely used for real gas behavior.

    Attributes:
        Tc (float): Critical temperature of the substance (K).
        Pc (float): Critical pressure of the substance (Pa).
        omega (float): Acentric factor of the substance.

    Methods:
        get_compressibility(): Calculates the compressibility factor Z.
        get_fugacity_coefficient(): Calculates the fugacity coefficient phi.
        get_bulk_phase_density(): Calculates the bulk phase density (kg/m^3).
        get_bulk_phase_molar_density(): Calculates the bulk phase molar density (mol/m^3).

    """

    def __init__(
        self,
        criticalTemperature: float,
        criticalPressure: float,
        acentricFactor: float,
        *args,
        **kwargs,
    ) -> None:
        # Initialize generic properties from BaseEOS
        super().__init__(*args, **kwargs)

        self.Tc = criticalTemperature
        self.Pc = criticalPressure
        self.omega = acentricFactor
        self.reducedTemperature = self.T / criticalTemperature

        # Peng-Robinson specific constants calculation
        nc = (1 + (4 - np.sqrt(8)) ** (1 / 3) + (4 + np.sqrt(8)) ** (1 / 3)) ** (-1)
        self.omega_a = (8 + 40 * nc) / (49 - 37 * nc)
        self.omega_b = nc / (3 + nc)

        self.a = self.omega_a * self.R**2 * self.Tc**2 / self.Pc
        self.b = self.omega_b * self.R * self.Tc / self.Pc

        self.kappa = 0.37464 + 1.54226 * self.omega - 0.26992 * self.omega**2
        self.alpha = (1 + self.kappa * (1 - np.sqrt(self.reducedTemperature))) ** 2

    def calculate_eos_parameters(self) -> tuple[float, float]:
        """Calculate parameters A and B for the PR EOS."""
        A = self.a * self.alpha * self.P / (self.R**2 * self.T**2)
        B = self.b * self.P / (self.R * self.T)
        return A, B

    def get_compressibility(self) -> float:
        """
        Calculate the compressibility factor Z by solving the PR cubic equation:
        Z^3 - (1 - B)*Z^2 + (A - 2*B - 3*B^2)*Z - (A*B - B^2 - B^3) = 0
        """
        A, B = self.calculate_eos_parameters()
        coefficients = [1, -(1 - B), (A - 2 * B - 3 * B**2), -(A * B - B**2 - B**3)]
        roots = np.roots(coefficients)

        # Select the largest real root for the gas phase
        Z = np.max(roots[np.isreal(roots)]).real
        return float(Z)

    def get_fugacity_coefficient(self) -> float:
        """Calculate the fugacity coefficient using the PR EOS analytical expression."""
        Z = self.get_compressibility()
        A, B = self.calculate_eos_parameters()

        ln_phi = (
            (Z - 1)
            - np.log(Z - B)
            - A
            / (2 * np.sqrt(2) * B)
            * np.log((Z + (1 + np.sqrt(2)) * B) / (Z + (1 - np.sqrt(2)) * B))
        )
        return float(np.exp(ln_phi))
