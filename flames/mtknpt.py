from __future__ import annotations

import ase
import numpy as np
from ase.atoms import Atoms
from ase.md.md import MolecularDynamics
from scipy.special import exprel

# Coefficients for the fourth-order Suzuki-Yoshida integration scheme
# Ref: H. Yoshida, Phys. Lett. A 150, 5-7, 262-268 (1990).
#      https://doi.org/10.1016/0375-9601(90)90092-3
FOURTH_ORDER_COEFFS = [
    1 / (2 - 2 ** (1 / 3)),
    -(2 ** (1 / 3)) / (2 - 2 ** (1 / 3)),
    1 / (2 - 2 ** (1 / 3)),
]


class MTKBarostat:
    """MTK barostat for volume-and-cell fluctuations.

    See `MTKNPT` for the references.

    Note for `mask` keyword argument:
        If `mask` is None, full 9 DoF cell fluctuations are allowed.
        If `mask` is a 3-tuple of booleans, only the specified axes are allowed
        to fluctuate, and its DoF is a number of the specified axes. Thus,
        `mask=(True, True, True)` gives 3 DoF cell fluctuations along a, b,
        and c axes, which is different from the full 9 DoF fluctuations.
        Other cell fluctuations such as partially allowed off-diagonal
        components are not supported in the current implementation.
    """

    def __init__(
        self,
        num_atoms_global: int,
        temperature_K: float,
        pdamp: float,
        pchain: int = 3,
        ploop: int = 1,
        mask: tuple[bool, bool, bool] | None = None,
        vol_constraint: bool = False,
    ):
        self._num_atoms_global = num_atoms_global
        self._pdamp = pdamp
        self._pchain = pchain
        self._ploop = ploop

        # A recent PR showed that the cell DoF is not correctly calculated in the original implementation.
        # The correct DoF is 6 for full cell fluctuations, and not 9 as in the original implementation.
        # For details see https://gitlab.com/ase/ase/-/merge_requests/4303
        self._cell_dof = 6 if mask is None else sum(mask)

        if vol_constraint:
            self._cell_dof -= 1  # Traceless constraint removes 1 DoF

        self._kT = ase.units.kB * temperature_K

        self._W = (self._num_atoms_global + 1) * self._kT * self._pdamp**2

        assert pchain >= 1
        # Virtual masses for barostat momenta. Q_j' in the original paper.
        self._R = np.zeros(self._pchain)
        self._R[0] = self._cell_dof * self._kT * self._pdamp**2
        self._R[1:] = self._kT * self._pdamp**2

        self._xi = np.zeros(self._pchain)  # barostat coordinates
        self._p_xi = np.zeros(self._pchain)

    @property
    def W(self) -> float:
        """Virtual mass for barostat momenta."""
        return self._W

    def get_barostat_energy(self) -> float:
        energy = (
            np.sum(self._p_xi**2 / self._R) / 2
            + self._cell_dof * self._kT * self._xi[0]
            + self._kT * np.sum(self._xi[1:])
        )
        return float(energy)

    def integrate_nhc_baro(self, p_cell: np.ndarray, delta: float) -> np.ndarray:
        """Integrate exp(i * L_NHC-baro * delta)"""
        for _ in range(self._ploop):
            for coeff in FOURTH_ORDER_COEFFS:
                p_cell = self._integrate_nhc_baro_loop(p_cell, coeff * delta / self._ploop)
        return p_cell

    def _integrate_nhc_baro_loop(self, p_cell: np.ndarray, delta: float) -> np.ndarray:
        delta2 = delta / 2
        delta4 = delta / 4

        for j in range(self._pchain):
            self._integrate_p_xi_j(p_cell, self._pchain - j - 1, delta2, delta4)
        self._integrate_xi(delta)
        self._integrate_nhc_p_cell(p_cell, delta)
        for j in range(self._pchain):
            self._integrate_p_xi_j(p_cell, j, delta2, delta4)

        return p_cell

    def _integrate_p_xi_j(self, p_cell: np.ndarray, j: int, delta2: float, delta4: float) -> None:
        if j < self._pchain - 1:
            self._p_xi[j] *= np.exp(-delta4 * self._p_xi[j + 1] / self._R[j + 1])

        if j == 0:
            g_j = np.sum(p_cell**2) / self._W - self._cell_dof * self._kT
        else:
            g_j = self._p_xi[j - 1] ** 2 / self._R[j - 1] - self._kT
        self._p_xi[j] += delta2 * g_j

        if j < self._pchain - 1:
            self._p_xi[j] *= np.exp(-delta4 * self._p_xi[j + 1] / self._R[j + 1])

    def _integrate_xi(self, delta: float) -> None:
        for j in range(self._pchain):
            self._xi[j] += delta * self._p_xi[j] / self._R[j]

    def _integrate_nhc_p_cell(self, p_cell: np.ndarray, delta: float) -> None:
        p_cell *= np.exp(-delta * self._p_xi[0] / self._R[0])


class NoseHooverChainThermostat:
    """Nose-Hoover chain style thermostats.

    See `NoseHooverChainNVT` for the references.
    """

    def __init__(
        self,
        *,
        ndof: int,
        masses: np.ndarray,
        temperature_K: float,
        tdamp: float,
        tchain: int = 3,
        tloop: int = 1,
    ):
        """See `NoseHooverChainNVT` for the parameters."""
        self._ndof = ndof
        self._masses = masses  # (len(atoms), 1)
        self._tdamp = tdamp
        self._tchain = tchain
        self._tloop = tloop

        self._kT = ase.units.kB * temperature_K

        assert tchain >= 1
        self._Q = np.zeros(tchain)
        self._Q[0] = self._ndof * self._kT * tdamp**2
        self._Q[1:] = self._kT * tdamp**2

        # The following variables are updated during self.step()
        self._eta = np.zeros(self._tchain)
        self._p_eta = np.zeros(self._tchain)

    def get_thermostat_energy(self) -> float:
        """Return energy-like contribution from the thermostat variables."""
        energy = (
            self._ndof * self._kT * self._eta[0]
            + self._kT * np.sum(self._eta[1:])
            + np.sum(0.5 * self._p_eta**2 / self._Q)
        )
        return float(energy)

    def integrate_nhc(self, p: np.ndarray, delta: float) -> np.ndarray:
        """Integrate exp(i * L_NHC * delta) and update momenta `p`."""
        for _ in range(self._tloop):
            for coeff in FOURTH_ORDER_COEFFS:
                p = self._integrate_nhc_loop(p, coeff * delta / self._tloop)

        return p

    def _integrate_p_eta_j(self, p: np.ndarray, j: int, delta2: float, delta4: float) -> None:
        if j < self._tchain - 1:
            self._p_eta[j] *= np.exp(-delta4 * self._p_eta[j + 1] / self._Q[j + 1])

        if j == 0:
            g_j = np.sum(p**2 / self._masses) - self._ndof * self._kT
        else:
            g_j = self._p_eta[j - 1] ** 2 / self._Q[j - 1] - self._kT
        self._p_eta[j] += delta2 * g_j

        if j < self._tchain - 1:
            self._p_eta[j] *= np.exp(-delta4 * self._p_eta[j + 1] / self._Q[j + 1])

    def _integrate_eta(self, delta: float) -> None:
        self._eta += delta * self._p_eta / self._Q

    def _integrate_nhc_p(self, p: np.ndarray, delta: float) -> None:
        p *= np.exp(-delta * self._p_eta[0] / self._Q[0])

    def _integrate_nhc_loop(self, p: np.ndarray, delta: float) -> np.ndarray:
        delta2 = delta / 2
        delta4 = delta / 4

        for j in range(self._tchain):
            self._integrate_p_eta_j(p, self._tchain - j - 1, delta2, delta4)
        self._integrate_eta(delta)
        self._integrate_nhc_p(p, delta)
        for j in range(self._tchain):
            self._integrate_p_eta_j(p, j, delta2, delta4)

        return p


class MTKNPT(MolecularDynamics):
    """Isothermal-isobaric molecular dynamics with volume-and-cell fluctuations
    by Martyna-Tobias-Klein (MTK) method [1].

    See also :class:`NoseHooverChainNVT` for the references.
    The factorization of the Liouville operator is the same as Reference [1].

    - [1] G. J. Martyna, D. J. Tobias, and M. L. Klein, J. Chem. Phys. 101,
          4177-4189 (1994). https://doi.org/10.1063/1.467468
    """

    def __init__(
        self,
        atoms: Atoms,
        timestep: float,
        temperature_K: float,
        pressure_au: float,
        tdamp: float,
        pdamp: float,
        tchain: int = 3,
        pchain: int = 3,
        tloop: int = 1,
        ploop: int = 1,
        vol_constraint: bool = True,
        **kwargs,
    ):
        """
        Parameters
        ----------
        atoms: ase.Atoms
            The atoms object.
        timestep: float
            The time step in ASE time units.
        temperature_K: float
            The target temperature in K.
        pressure_au: float
            The external pressure in eV/Ang^3.
        tdamp: float
            The characteristic time scale for the thermostat in ASE time units.
            Typically, it is set to 100 times of `timestep`.
        pdamp: float
            The characteristic time scale for the barostat in ASE time units.
            Typically, it is set to 1000 times of `timestep`.
        tchain: int
            The number of thermostat variables in the Nose-Hoover thermostat.
        pchain: int
            The number of barostat variables in the MTK barostat.
        tloop: int
            The number of sub-steps in thermostat integration.
        ploop: int
            The number of sub-steps in barostat integration.
        vol_constraint: bool
            If True, the (N, V, sigma_a = 0, T)-ensemble is sampled, which allows for full cell fluctuations
            while keeping the cell volume fixed.
            This ensemble was introduced in [doi:10.1021/acs.jctc.5b00748].
        **kwargs : dict, optional
            Additional arguments passed to :class:~ase.md.md.MolecularDynamics
            base class.
        """
        super().__init__(
            atoms=atoms,
            timestep=timestep,
            **kwargs,
        )
        assert self.masses.shape == (len(self.atoms), 1)

        self.vol_constraint = vol_constraint

        if len(atoms.constraints) > 0:
            raise NotImplementedError("Current implementation does not support constraints")

        self._num_atoms_global = self.atoms.get_global_number_of_atoms()
        self._thermostat = NoseHooverChainThermostat(
            ndof=self.atoms.get_number_of_degrees_of_freedom(),
            masses=self.masses,
            temperature_K=temperature_K,
            tdamp=tdamp,
            tchain=tchain,
            tloop=tloop,
        )

        self._barostat = MTKBarostat(
            num_atoms_global=self._num_atoms_global,
            temperature_K=temperature_K,
            pdamp=pdamp,
            pchain=pchain,
            ploop=ploop,
            vol_constraint=vol_constraint,
        )

        self._temperature_K = temperature_K
        self._pressure_au = pressure_au

        self._kT = ase.units.kB * self._temperature_K

        # The following variables are updated during self.step()
        self._q = self.atoms.get_positions()  # positions
        self._p = self.atoms.get_momenta()  # momenta
        self._h = np.array(self.atoms.get_cell())  # cell

        # ADDED: Store initial volume for strict conservation
        self._initial_volume = np.abs(np.linalg.det(self._h))

        self._init_cell_momenta()

    @property
    def mask(self) -> tuple[bool, bool, bool] | None:
        return None

    @mask.setter
    def mask(self, mask: tuple[bool, bool, bool]) -> None:
        raise AttributeError()

    def _init_cell_momenta(self) -> None:
        self._p_g = np.zeros((3, 3))  # cell momenta

    def step(self) -> None:
        dt2 = self.dt / 2

        self._integrate_p_cell_by_barostat(dt2)
        self._p = self._thermostat.integrate_nhc(self._p, dt2)
        self._integrate_p_cell(dt2)
        self._integrate_p(dt2)
        self._integrate_q(self.dt)
        self._integrate_q_cell(self.dt)
        self._integrate_p(dt2)
        self._integrate_p_cell(dt2)
        self._p = self._thermostat.integrate_nhc(self._p, dt2)
        self._integrate_p_cell_by_barostat(dt2)

        self._update_atoms()

    def get_conserved_energy(self) -> float:
        conserved_energy = (
            self.atoms.get_total_energy()
            + self._thermostat.get_thermostat_energy()
            + self._barostat.get_barostat_energy()
            + self._get_cell_kinetic_energy()
            + self._pressure_au * self._get_volume()
        )
        return float(conserved_energy)

    def _update_atoms(self) -> None:
        self.atoms.set_positions(self._q)
        self.atoms.set_momenta(self._p)
        self.atoms.set_cell(self._h, scale_atoms=False)

    def _get_volume(self) -> float:
        return np.abs(np.linalg.det(self._h))

    def _get_forces(self) -> np.ndarray:
        self._update_atoms()
        return self.atoms.get_forces(md=True)

    def _get_stress(self) -> np.ndarray:
        self._update_atoms()
        stress = self.atoms.get_stress(voigt=False, include_ideal_gas=True)
        return -stress

    def _get_cell_kinetic_energy(self) -> float:
        return float(np.sum(self._p_g**2) / (2 * self._barostat.W))

    def _integrate_q(self, delta: float) -> None:
        """Integrate exp(i * L_1 * delta)"""
        # eigvals: (3-eigvec), U: (3-xyz, 3-eigvec)
        eigvals, U = np.linalg.eigh(self._p_g)
        x = self._q @ U  # (num_atoms, 3-eigvec)
        y = self._p @ U  # (num_atoms, 3-eigvec)
        sol = (
            x * np.exp(eigvals * delta / self._barostat.W)[None, :]
            + delta * y / self.masses * exprel(eigvals * delta / self._barostat.W)[None, :]
        )  # (num_atoms, 3-eigvec)
        self._q = sol @ U.T

    def _integrate_p(self, delta: float) -> None:
        """Integrate exp(i * L_2 * delta)"""
        forces = self._get_forces()  # (num_atoms, 3-xyz)

        # eigvals: (3-eigvec), U: (3-xyz, 3-eigvec)
        eigvals, U = np.linalg.eigh(self._p_g)
        kappas = eigvals + np.trace(self._p_g) / (3 * self._num_atoms_global)  # (3-eigvec)
        y = self._p @ U  # (num_atoms, 3-eigvec)
        sol = (
            y * np.exp(-kappas * delta / self._barostat.W)[None, :]
            + delta * (forces @ U) * exprel(-kappas * delta / self._barostat.W)[None, :]
        )  # (num_atoms, 3-eigvec)
        self._p = sol @ U.T

    def _integrate_q_cell(self, delta: float) -> None:
        """Integrate exp(i * L_(g, 1) * delta)"""
        # U @ np.diag(eigvals) @ U.T = self._p_g
        # eigvals: (3-eigvec), U: (3-xyz, 3-eigvec)
        eigvals, U = np.linalg.eigh(self._p_g)
        n = self._h @ U  # (3-axis, 3-eigvec)
        sol = n * np.exp(eigvals * delta / self._barostat.W)[None, :]  # (3-axis, 3-eigvec)
        self._h = sol @ U.T

        # Rescale cell tensor to perfectly match initial volume
        if self.vol_constraint:
            current_vol = np.abs(np.linalg.det(self._h))
            scale_factor = (self._initial_volume / current_vol) ** (1.0 / 3.0)
            self._h *= scale_factor

    def _integrate_p_cell(self, delta: float) -> None:
        """Integrate exp(i * L_(g, 2) * delta)"""
        stress = self._get_stress()
        volume = self._get_volume()
        particle_dof = 3 * self._num_atoms_global
        kinetic_term = np.sum(self._p**2 / self.masses) / particle_dof
        pv_tensor = volume * (stress - self._pressure_au * np.eye(3))
        G = pv_tensor + kinetic_term * np.eye(3)

        # A traceless constraint is applied to the cell momenta to ensure that the cell volume remains constant.
        # Rogge, S. M. J. et al. Theory Comput. 11, 5583–5597 (2015) DOI: 10.1021/acs.jctc.5b00748
        if self.vol_constraint:
            G -= np.trace(G) / 3.0 * np.eye(3)

        self._p_g += delta * G

        # Strictly eliminate trace from the momenta tensor itself
        if self.vol_constraint:
            self._p_g -= np.trace(self._p_g) / 3.0 * np.eye(3)

    def _integrate_p_cell_by_barostat(self, delta: float) -> None:
        self._p_g = self._barostat.integrate_nhc_baro(self._p_g, delta)
