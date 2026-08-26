import math
import numpy as np
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from numba import njit, prange
from vesin import NeighborList

# --- 1. Numba Real Space Kernel ---
@njit(fastmath=True, parallel=False, cache=True)
def _compute_ewald_real_numba(distances, q_i, q_j, alpha):
    """
    Evaluates real space using the exact distances found by Vesin.
    Sequential loop is faster here because memory is contiguous and 
    avoids thread-spinning overhead for small arrays.
    """
    u_real = 0.0
    for k in range(len(distances)):
        r = distances[k]
        # Prevent division by zero mathematically
        if r > 1e-8:
            u_real += (q_i[k] * q_j[k]) * math.erfc(alpha * r) / r
            
    # Divide by 2 because Vesin full_list=True double-counts pairs
    return u_real / 2.0


# --- 2. Optimized Numba Reciprocal Space Kernel ---
@njit(fastmath=True, parallel=True, cache=True)
def _compute_ewald_recip_numba(positions, charges, recip_cell, nx, ny, nz, alpha, volume):
    """
    Your original k-space implementation, kept in Numba to avoid 
    JAX dispatch overhead.
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

    return u_recip / 2.0


# --- 3. ASE Calculator ---
class CustomEwald(Calculator):
    implemented_properties = ["energy", "free_energy"]
    default_parameters = {
        "cutoff": 12.0,
        "precision": 1e-6,
    }
    nolabel = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cutoff = self.parameters.get("cutoff", 12.0)
        self.precision = self.parameters.get("precision", 1e-6)

        self._cached_cell = None
        self.alpha = None
        self.grid_limits = (1, 1, 1)
        self.recip_cell = None
        self.volume = None
        
        # Initialize Vesin once, update cell/positions during compute
        self.neighbor_calculator = NeighborList(cutoff=self.cutoff, full_list=True)

    def _tune_ewald_parameters(self, cell):
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

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        if properties is None:
            properties = self.implemented_properties

        super().calculate(atoms, properties, system_changes)

        positions = self.atoms.positions
        cell = self.atoms.cell.array
        charges = self.atoms.get_initial_charges()

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
        
        u_real = _compute_ewald_real_numba(distances, q_i, q_j, self.alpha)

        # 2. Reciprocal Space: Parallel Numba
        u_recip = _compute_ewald_recip_numba(
            positions, charges, self.recip_cell, nx, ny, nz, self.alpha, self.volume
        )

        # 3. Self Energy Correction
        u_self = -(self.alpha / np.sqrt(np.pi)) * np.sum(charges**2)

        # 4. Total Energy
        total_energy_ev = (u_real + u_recip + u_self) * units.Hartree * units.Bohr

        self.results["energy"] = total_energy_ev
        self.results["free_energy"] = total_energy_ev