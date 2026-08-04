from abc import ABC

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
        self._T = temperature
        self._P = pressure
        self._molar_mass = molarMass

        # Universal gas constant in J/(mol*K)
        self.R = units.kB / units.J * units.mol

    def get_compressibility(self) -> float:
        """Calculate the compressibility factor Z for the thermodynamically stable phase."""
        Z, _, _ = self.get_stable_phase_properties()
        return Z

    @property
    def T(self) -> float:
        return self._T

    @T.setter
    def T(self, value: float) -> None:
        if value <= 0:
            raise ValueError("Temperature must be greater than zero.")
        self._T = value

    @property
    def P(self) -> float:
        return self._P

    @P.setter
    def P(self, value: float) -> None:
        if value <= 0:
            raise ValueError("Pressure must be greater than zero.")
        self._P = value

    @property
    def molar_mass(self) -> float:
        return self._molar_mass

    @molar_mass.setter
    def molar_mass(self, value: float) -> None:
        if value <= 0:
            raise ValueError("Molar mass must be greater than zero.")
        self._molar_mass = value

    def get_fugacity_coefficient(self) -> float:
        """Calculate the fugacity coefficient phi for the thermodynamically stable phase."""
        _, phi, _ = self.get_stable_phase_properties()
        return phi

    def get_phase_state(self) -> str:
        """Returns a string indicating the current thermodynamically stable phase."""
        _, _, phase = self.get_stable_phase_properties()
        return phase

    def get_bulk_phase_density(self) -> float:
        """
        Calculate the bulk phase density using the compressibility factor.
        rho = MM / Vm (kg/m^3)
        """
        Z = self.get_compressibility()
        molar_volume = self.R * self._T * Z / self._P

        density = 1e-3 * self.molar_mass / molar_volume
        return float(density)

    def get_bulk_phase_molar_density(self) -> float:
        """
        Calculate the equivalent bulk phase molar density.
        (mol/m^3)
        """
        Z = self.get_compressibility()
        molar_volume = self.R * self._T * Z / self._P

        molar_density = 1 / molar_volume
        return float(molar_density)

    def get_stable_phase_properties(self) -> tuple[float, float, str]:
        """
        Returns the compressibility (Z), fugacity coefficient (phi), and phase name of the stable phase.
        This method should be implemented by subclasses to provide specific EOS calculations.
        """
        return 1, 1, "Ideal Gas"  # Default implementation for ideal gas behavior


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
        super().__init__(*args, **kwargs)

        self.Tc = criticalTemperature
        self.Pc = criticalPressure
        self.omega = acentricFactor

        # Peng-Robinson specific constants calculation
        nc = (1 + (4 - np.sqrt(8)) ** (1 / 3) + (4 + np.sqrt(8)) ** (1 / 3)) ** (-1)
        self.omega_a = (8 + 40 * nc) / (49 - 37 * nc)
        self.omega_b = nc / (3 + nc)

        self.a = self.omega_a * self.R**2 * self.Tc**2 / self.Pc
        self.b = self.omega_b * self.R * self.Tc / self.Pc

        self.kappa = 0.37464 + 1.54226 * self.omega - 0.26992 * self.omega**2

    @property
    def reducedTemperature(self) -> float:
        return self.T / self.Tc

    @property
    def alpha(self) -> float:
        """Calculate the temperature-dependent alpha parameter for the PR EOS."""
        return (1 + self.kappa * (1 - np.sqrt(self.reducedTemperature))) ** 2

    def calculate_eos_parameters(self) -> tuple[float, float]:
        """Calculate parameters A and B for the PR EOS."""
        A = self.a * self.alpha * self.P / (self.R**2 * self.T**2)
        B = self.b * self.P / (self.R * self.T)
        return A, B

    def _calculate_phi_for_z(self, Z: float, A: float, B: float) -> float:
        """Helper method to calculate fugacity coefficient for a specific Z root."""
        ln_phi = (
            (Z - 1)
            - np.log(Z - B)
            - A
            / (2 * np.sqrt(2) * B)
            * np.log((Z + (1 + np.sqrt(2)) * B) / (Z + (1 - np.sqrt(2)) * B))
        )
        return float(np.exp(ln_phi))

    def get_stable_phase_properties(self) -> tuple[float, float, str]:
        """
        Finds all physical roots of the PR cubic equation and determines the stable phase.
        Returns:
        --------
        Z (float): Compressibility factor of the stable phase.
        phi (float): Fugacity coefficient of the stable phase.
        phase (str): Description of the stable phase. Can be:
            - "Fluid is supercritical"
            - "Fluid is a vapour"
            - "Fluid is a liquid"
            - "Vapour=stable, Liquid=metastable"
            - "Liquid=stable, Vapour=metastable"
            - "Vapor-Liquid Equilibrium"
        """
        A, B = self.calculate_eos_parameters()
        coefficients = [1, -(1 - B), (A - 2 * B - 3 * B**2), -(A * B - B**2 - B**3)]

        # Calculate all roots
        roots = np.roots(coefficients)

        # Filter for real roots
        real_roots = roots[np.isclose(roots.imag, 0)].real

        # Filter for physical roots (compressibility must be greater than excluded volume B)
        physical_roots = np.sort(real_roots[real_roots > B])

        if len(physical_roots) == 0:
            raise ValueError("No physical roots (Z > B) found for given conditions.")

        elif len(physical_roots) == 1:
            # Single phase region
            Z = physical_roots[0]
            phi = self._calculate_phi_for_z(Z, A, B)

            # Classify the single phase based on critical point
            if self.T > self.Tc and self.P > self.Pc:
                phase = "Fluid is supercritical"
            elif self.T < self.Tc and self.P < self.Pc:
                phase = "Fluid is a vapour"
            else:
                phase = "Fluid is a liquid"

        # Two-phase region (2 or 3 real roots)
        # If 3 real roots, the middle root is non-physical and
        # usually filtered/ignored by taking min/max
        elif len(physical_roots) == 2 or len(physical_roots) == 3:
            Z_liquid = physical_roots[0]  # Smallest root
            Z_vapour = physical_roots[-1]  # Largest root

            phi_liquid = self._calculate_phi_for_z(Z_liquid, A, B)
            phi_vapor = self._calculate_phi_for_z(Z_vapour, A, B)

            # The most stable phase has the lowest fugacity coefficient
            if np.isclose(phi_vapor, phi_liquid, rtol=1e-5):
                Z, phi, phase = Z_vapour, phi_vapor, "Vapor-Liquid Equilibrium"

            elif phi_vapor < phi_liquid:
                Z, phi, phase = Z_vapour, phi_vapor, "Vapour=stable, Liquid=metastable"
            else:
                Z, phi, phase = Z_liquid, phi_liquid, "Liquid=stable, Vapour=metastable"

        else:
            raise ValueError(
                "Unexpected number of physical roots found. Check EOS parameters."
                "Number of physical roots: {}".format(len(physical_roots))
            )

        return Z, phi, phase
