import math

import numpy as np
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from numba import njit, prange
from vesin import NeighborList


@njit(fastmath=True, parallel=False, cache=True)
def _compute_ewald_real_numba(distances, q_i, q_j, alpha):
    r"""
    Evaluates real space using the exact distances found by Vesin.

    The real space contribution to the electrostatic energy represents the interaction
    of the point charges screened by the local Gaussian clouds.
    Due to the properties of the complementary error function, this interaction decays rapidly with distance:

    .. math::

        E_{\mathrm{real}} = \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \sideset{}{'}\sum_{\mathbf{n}} \frac{q_i q_j \mathrm{erfc}(\alpha \vert{}\mathbf{r}_{ij} + \mathbf{n}\vert{})}{\vert{}\mathbf{r}_{ij} + \mathbf{n}\vert{}}

    The rapid decay ensures that the sum can be safely truncated at a predefined real-space cutoff radius :math:`r_c`, omitting contributions from image cells beyond this boundary.

    Parameters
    ----------
    distances : np.ndarray
        Array of distances between particle pairs.
    q_i : np.ndarray
        Array of charges for the first particle in each pair.
    q_j : np.ndarray
        Array of charges for the second particle in each pair.
    alpha : float
        Ewald splitting parameter.

    Returns
    -------
    float
        The computed real space energy contribution.
    """
    u_real = 0.0
    for k in range(len(distances)):
        r = distances[k]
        # Prevent division by zero mathematically
        if r > 1e-8:
            u_real += (q_i[k] * q_j[k]) * math.erfc(alpha * r) / r

    return u_real


@njit(fastmath=True, parallel=True, cache=True)
def _compute_ewald_recip_numba(positions, charges, recip_cell, nx, ny, nz, alpha, volume):
    """
    Evaluates reciprocal space using a parallelized Numba kernel.

    Parameters
    ----------
    positions : np.ndarray
        Array of particle positions.
    charges : np.ndarray
        Array of particle charges.
    recip_cell : np.ndarray
        The 3x3 reciprocal cell matrix.
    nx, ny, nz : int
        Grid limits for the reciprocal space.
    alpha : float
        Ewald splitting parameter.
    volume : float
        The volume of the simulation cell.
    """
    u_recip = 0.0
    alpha_sq_4 = 4.0 * alpha * alpha
    prefactor = 4.0 * np.pi / volume

    dim_x = 2 * nx + 1
    dim_y = 2 * ny + 1
    dim_z = 2 * nz + 1
    total_k_points = dim_x * dim_y * dim_z

    for idx in prange(total_k_points):
        h_idx = (idx // (dim_y * dim_z)) - nx
        rem = idx % (dim_y * dim_z)
        k_idx = (rem // dim_z) - ny
        l_idx = (rem % dim_z) - nz

        if h_idx == 0 and k_idx == 0 and l_idx == 0:
            continue

        kx = h_idx * recip_cell[0, 0] + k_idx * recip_cell[1, 0] + l_idx * recip_cell[2, 0]
        ky = h_idx * recip_cell[0, 1] + k_idx * recip_cell[1, 1] + l_idx * recip_cell[2, 1]
        kz = h_idx * recip_cell[0, 2] + k_idx * recip_cell[1, 2] + l_idx * recip_cell[2, 2]

        k_sq = kx * kx + ky * ky + kz * kz

        S_real = 0.0
        S_imag = 0.0

        for i in range(len(positions)):
            dot = kx * positions[i, 0] + ky * positions[i, 1] + kz * positions[i, 2]
            S_real += charges[i] * math.cos(dot)
            S_imag += charges[i] * math.sin(dot)

        S_sq = S_real * S_real + S_imag * S_imag

        term = prefactor * math.exp(-k_sq / alpha_sq_4) / k_sq * S_sq
        u_recip += term

    return u_recip


class CustomEwald(Calculator):
    r"""
    Custom Ewald calculator for periodic Coulomb interactions, optimized with Vesin and Numba.

    The Ewald summation method resolves this by partitioning the conditionally convergent series into two rapidly converging sums.
    This is achieved by introducing a screening parameter :math:`\alpha`, which effectively surrounds each point charge with a
    neutralizing Gaussian charge distribution of opposite sign.

    .. math::

       \frac{1}{r} = \frac{\mathrm{erfc}(\alpha r)}{r} + \frac{\mathrm{erf}(\alpha r)}{r}

    where :math:`\mathrm{erfc}` and :math:`\mathrm{erf}` are the complementary error function and the error function, respectively.
    This partitioning separates the total electrostatic energy into three distinct and convergent components: the real-space
    energy (:math:`E_{\mathrm{real}}`), the reciprocal-space energy (:math:`E_{\mathrm{recip}}`), and the self-energy correction
    (:math:`E_{\mathrm{self}}`).
    The real-space energy is computed using a cutoff, while the reciprocal-space energy is computed using a grid in reciprocal space.

    .. math::

       E_{total} = E_{real} + E_{recip} + E_{self}

    Parameters
    ----------
    cutoff : float, optional
        The cutoff distance for the real-space sum.
        Default is 12.0 Angstroms

    precision : float, optional
        The desired precision for the Ewald summation.
        This parameter influences the choice of the Ewald splitting parameter :math:`\alpha` and the grid limits in reciprocal space.
        Default is 1e-6.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.cutoff = self.parameters.get("cutoff", 12.0)  # type: ignore
        self.precision = self.parameters.get("precision", 1e-6)  # type: ignore

        self.implemented_properties = ["energy", "electrostatic_energy"]

        # Cache the tuning parameters so it only recalculate it if the cell volume/shape changes
        self._cached_cell = None
        self.alpha = None
        self.grid_limits = (1, 1, 1)
        self.recip_cell = None
        self.volume = None

        # Initialize Vesin once, update cell/positions during compute
        self.neighbor_calculator = NeighborList(cutoff=self.cutoff, full_list=False)

    def _tune_ewald_parameters(self, cell) -> None:
        """
        Tune Ewald parameters based on the current cell dimensions and desired precision.

        Parameters
        ----------
        cell : np.ndarray
            The 3x3 cell matrix.
        """
        self.volume = np.abs(np.linalg.det(cell))
        self.alpha = np.sqrt(-np.log(self.precision)) / self.cutoff
        k_max = 2.0 * self.alpha * np.sqrt(-np.log(self.precision))

        self.recip_cell = 2.0 * np.pi * np.linalg.inv(cell).T
        recip_lengths = np.linalg.norm(self.recip_cell, axis=1)

        nx = int(np.ceil(k_max / recip_lengths[0]))
        ny = int(np.ceil(k_max / recip_lengths[1]))
        nz = int(np.ceil(k_max / recip_lengths[2]))

        self.grid_limits = (nx, ny, nz)
        self._cached_cell = cell.copy()

    def calculate(self, atoms=None, properties=None, system_changes=all_changes) -> None:
        if properties is None:
            properties = self.implemented_properties

        super().calculate(atoms, properties, system_changes)

        positions = self.atoms.positions  # type: ignore
        cell = self.atoms.cell.array  # type: ignore
        charges = self.atoms.get_initial_charges()  # type: ignore

        # Calculate the Ewald parameters only if the cell shape/volume changes
        if self._cached_cell is None or not np.allclose(cell, self._cached_cell):
            self._tune_ewald_parameters(cell)

        nx, ny, nz = self.grid_limits

        # 1. Real Space: Vesin (C++) + Numba
        i_idx, j_idx, distances = self.neighbor_calculator.compute(
            points=positions,
            box=cell,
            periodic=True,
            quantities="ijd",
        )

        q_i = charges[i_idx]
        q_j = charges[j_idx]

        # Real space energy
        u_real = _compute_ewald_real_numba(distances, q_i, q_j, self.alpha)

        # 2. Reciprocal Space energy
        u_recip = _compute_ewald_recip_numba(
            positions, charges, self.recip_cell, nx, ny, nz, self.alpha, self.volume
        )

        # 3. Self Energy Correction
        u_self = -(self.alpha / np.sqrt(np.pi)) * np.sum(charges**2)

        # Total Energy in eV
        total_energy_ev = (u_real + u_recip + u_self) * units.Hartree * units.Bohr

        self.results["energy"] = total_energy_ev
        self.results["electrostatic_energy"] = total_energy_ev
