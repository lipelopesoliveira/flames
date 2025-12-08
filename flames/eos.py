import numpy as np
from ase import units


class PengRobinsonEOS:
    def __init__(
        self,
        temperature: float,
        pressure: float,
        criticalTemperature: float,
        criticalPressure: float,
        acentricFactor: float,
        molarMass: float,
    ) -> None:
        """
        Peng-Robinson Equation of State

        Parameters:
        -----------
        temperature: float
            Temperature in Kelvin
        pressure: float
            Pressure in Pascals
        criticalTemperature: float
            Critical temperature in Kelvin
        criticalPressure: float
            Critical pressure in Pascals
        acentricFactor: float
            Acentric factor of the substance
        molarMass: float
            Molar mass of the substance in g/mol
        """

        self.T = temperature
        self.P = pressure
        self.Tc = criticalTemperature
        self.Pc = criticalPressure
        self.molar_mass = molarMass
        self.omega = acentricFactor
        self.reducedTemperature = temperature / criticalTemperature

        # Constants

        self.R = units.kB / units.J * units.mol  # J/(mol*K), universal gas constant

        nc = (1 + (4 - np.sqrt(8)) ** (1 / 3) + (4 + np.sqrt(8)) ** (1 / 3)) ** (-1)
        self.omega_a = (8 + 40 * nc) / (49 - 37 * nc)
        self.omega_b = nc / (3 + nc)

        self.a = self.omega_a * self.R**2 * self.Tc**2 / self.Pc
        self.b = self.omega_b * self.R * self.Tc / self.Pc

        self.kappa = 0.37464 + 1.54226 * self.omega - 0.26992 * self.omega**2
        self.alpha = (1 + self.kappa * (1 - np.sqrt(self.reducedTemperature))) ** 2

    def calculate_eos_parameters(self) -> tuple[float, float]:
        """
        Calculate the parameters A and B for the Peng-Robinson EOS.

        Returns:
        --------
        A: float
            Parameter A
        B: float
            Parameter B
        """

        A = self.a * self.alpha * self.P / (self.R**2 * self.T**2)
        B = self.b * self.P / (self.R * self.T)

        return A, B

    def get_compressibility(self) -> float:
        """
        Calculate the compressibility factor Z using the Peng-Robinson EOS.

        The compressibility factor Z is calculated by solving the cubic equation derived from the Peng-Robinson EOS:

        Z^3 - (1 - B) * Z^2 + (A - 2 * B - 3 * B^2) * Z - (A * B - B^2 - B^3) = 0

        Returns:
        --------
        Z: float
            Compressibility factor Z
        """
        A, B = self.calculate_eos_parameters()

        # Calculate the compressibility factor Z by solving the cubic equation
        coefficients = [1, -(1 - B), (A - 2 * B - 3 * B**2), -(A * B - B**2 - B**3)]
        roots = np.roots(coefficients)

        # Select the largest real root as the compressibility factor Z
        Z = np.max(roots).real

        return float(Z)

    def get_fugacity_coefficient(self) -> float:
        """
        Calculate the fugacity coefficient using the Peng-Robinson EOS.

        The fugacity coefficient is calculated using the compressibility factor Z and the parameters A and B:

        ln(phi) = (Z - 1) - log(Z - B) - A / (2 * sqrt(2) * B) * log((Z + (1 + sqrt(2)) * B) / (Z + (1 - sqrt(2)) * B))

        where:
        phi is the fugacity coefficient,
        Z is the compressibility factor,
        A and B are parameters calculated from the Peng-Robinson EOS.

        Returns:
        --------
        phi: float
            Fugacity coefficient phi
        """

        Z = self.get_compressibility()
        A, B = self.calculate_eos_parameters()

        ln_phi = (
            (Z - 1)
            - np.log(Z - B)
            - A
            / (2 * np.sqrt(2) * B)
            * np.log((Z + (1 + np.sqrt(2)) * B) / (Z + (1 - np.sqrt(2)) * B))
        )
        phi = np.exp(ln_phi)

        return phi

    def get_bulk_phase_density(self) -> float:
        """
        Calculate the bulk phase density using the Peng-Robinson EOS.

           rho = MM / Vm

        where:
        rho is the density in kg/m^3,
        MM is the molar mass in g/mol,
        Vm is the molar volume in m^3/mol.

        Returns:
        --------
        density: float
            Bulk phase density in kg/m^3
        """
        Z = self.get_compressibility()
        molar_volume = self.R * self.T * Z / self.P
        density = 1e-3 * self.molar_mass / molar_volume * units.mol  # Density in kg/m^3
        return density

    def get_bulk_phase_molar_density(self) -> float:
        """
        Calculate the equivalent bulk phase number of molecules per cubic meter.

        Returns:

        molar_density: float
            Bulk phase molar density in mol/m^3
        """

        Z = self.get_compressibility()
        molar_volume = self.R * self.T * Z / self.P

        molar_density = 1 / molar_volume * units.mol  # Density in mol/m^3

        return molar_density
