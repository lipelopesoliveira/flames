import math

import numpy as np
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from numba import njit, prange

# from flames.lennard_jones import numba_neighbor_list


@njit(fastmath=True, parallel=True)
def compute_ewald_real(i_idx, j_idx, distances, charges, alpha):
    """Computes the screened Coulomb interaction in real space in parallel."""
    u_real = 0.0

    # prange distributes the loop across all CPU cores.
    # Numba safely manages the u_real += reduction under the hood.
    for k in prange(len(i_idx)):
        i = i_idx[k]
        j = j_idx[k]
        r = distances[k]

        qiqj = charges[i] * charges[j]
        u_real += qiqj * math.erfc(alpha * r) / r

    return u_real / 2.0


@njit(fastmath=True, parallel=False, cache=True)
def compute_ewald_real_exact(positions, charges, cell, alpha, cutoff):
    """Brute-force real-space Ewald that checks all 27 periodic images.
    Immune to small unit cell MIC violations."""
    u_real = 0.0
    n = len(positions)
    cutoff_sq = cutoff * cutoff

    for i in prange(n):
        u_i = 0.0
        for j in range(n):
            dx_raw = positions[j, 0] - positions[i, 0]
            dy_raw = positions[j, 1] - positions[i, 1]
            dz_raw = positions[j, 2] - positions[i, 2]

            # Explicitly check the central cell and all 26 surrounding cells
            for sx in range(-1, 2):
                for sy in range(-1, 2):
                    for sz in range(-1, 2):
                        # Skip self-interaction ONLY at distance 0
                        if i == j and sx == 0 and sy == 0 and sz == 0:
                            continue

                        rx = dx_raw + sx * cell[0, 0] + sy * cell[1, 0] + sz * cell[2, 0]
                        ry = dy_raw + sx * cell[0, 1] + sy * cell[1, 1] + sz * cell[2, 1]
                        rz = dz_raw + sx * cell[0, 2] + sy * cell[1, 2] + sz * cell[2, 2]

                        r_sq = rx * rx + ry * ry + rz * rz

                        if r_sq <= cutoff_sq:
                            r = math.sqrt(r_sq)
                            u_i += (charges[i] * charges[j]) * math.erfc(alpha * r) / r
        u_real += u_i

    return u_real / 2.0


# --- 2. Reciprocal Space Kernel ---
@njit(fastmath=True, parallel=True)
def compute_ewald_recip(positions, charges, recip_cell, nx, ny, nz, alpha, volume):
    """Computes the long-range periodic Coulomb interaction in k-space in parallel."""
    u_recip = 0.0
    alpha_sq_4 = 4.0 * alpha * alpha
    prefactor = 4.0 * np.pi / volume

    # Calculate the total dimensions of the reciprocal grid
    dim_x = 2 * nx + 1
    dim_y = 2 * ny + 1
    dim_z = 2 * nz + 1
    total_k_points = dim_x * dim_y * dim_z

    # A single flattened parallel loop for maximum load-balancing
    for idx in prange(total_k_points):
        # Mathematically unpack the 1D index back into 3D (h, k, l) coordinates
        h_idx = (idx // (dim_y * dim_z)) - nx
        rem = idx % (dim_y * dim_z)
        k_idx = (rem // dim_z) - ny
        l_idx = (rem % dim_z) - nz

        # Skip the infinite self-interaction term at k=0
        if h_idx == 0 and k_idx == 0 and l_idx == 0:
            continue

        kx = h_idx * recip_cell[0, 0] + k_idx * recip_cell[1, 0] + l_idx * recip_cell[2, 0]
        ky = h_idx * recip_cell[0, 1] + k_idx * recip_cell[1, 1] + l_idx * recip_cell[2, 1]
        kz = h_idx * recip_cell[0, 2] + k_idx * recip_cell[1, 2] + l_idx * recip_cell[2, 2]

        k_sq = kx * kx + ky * ky + kz * kz

        S_real = 0.0
        S_imag = 0.0

        # This inner loop remains sequential per thread
        for i in range(len(positions)):
            dot = kx * positions[i, 0] + ky * positions[i, 1] + kz * positions[i, 2]
            S_real += charges[i] * math.cos(dot)
            S_imag += charges[i] * math.sin(dot)

        S_sq = S_real * S_real + S_imag * S_imag

        term = prefactor * math.exp(-k_sq / alpha_sq_4) / k_sq * S_sq
        u_recip += term

    return u_recip / 2.0


class CustomEwald(Calculator):
    implemented_properties = ["energy"]
    default_parameters = {
        "cutoff": 12.0,
        "precision": 1e-6,
    }
    nolabel = True

    def __init__(self, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.cutoff = kwargs.get("cutoff", 12.0)
        self.precision = kwargs.get("precision", 1e-6)

        # We will cache the tuning parameters so we only recalculate them
        # if the cell volume/shape changes (e.g., during NPT trial moves)
        self._cached_cell = None
        self.alpha = None
        self.grid_limits = (1, 1, 1)
        self.recip_cell = None
        self.volume = None

    def _tune_ewald_parameters(self, cell):
        """Calculates alpha and k-space grid based on target precision."""
        self.volume = np.abs(np.linalg.det(cell))

        # Real-space precision bound
        self.alpha = np.sqrt(-np.log(self.precision)) / self.cutoff

        # Reciprocal-space precision bound
        k_max = 2.0 * self.alpha * np.sqrt(-np.log(self.precision))

        # Reciprocal lattice vectors (b_i = 2*pi * inverse_transpose)
        self.recip_cell = 2.0 * np.pi * np.linalg.inv(cell).T
        recip_lengths = np.linalg.norm(self.recip_cell, axis=1)

        # Map k_max to integer grid limits
        nx = int(np.ceil(k_max / recip_lengths[0]))
        ny = int(np.ceil(k_max / recip_lengths[1]))
        nz = int(np.ceil(k_max / recip_lengths[2]))

        self.grid_limits = (nx, ny, nz)
        self._cached_cell = cell.copy()

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        if properties is None:
            properties = self.implemented_properties

        Calculator.calculate(self, atoms, properties, system_changes)

        positions = self.atoms.positions  # type: ignore
        cell = self.atoms.cell.array  # type: ignore

        # Ensure charges exist. Look in arrays first, fallback to get_initial_charges()
        charges = self.atoms.get_initial_charges()  # type: ignore

        if np.all(charges == 0):
            raise ValueError("All atomic charges are zero. Ewald cannot compute.")

        # 2. Dynamic Auto-Tuning (Only updates if cell shape/volume changes)
        if self._cached_cell is None or not np.allclose(cell, self._cached_cell):
            self._tune_ewald_parameters(cell)

        nx, ny, nz = self.grid_limits

        # 3. Real-Space Component
        # inv_cell = np.linalg.inv(cell)
        # i_idx, j_idx, distances = numba_neighbor_list(
        #    positions, cell, inv_cell, self.cutoff, use_robust_mic=True
        # )

        # u_real = compute_ewald_real(i_idx, j_idx, distances, charges, self.alpha)
        u_real = compute_ewald_real_exact(positions, charges, cell, self.alpha, self.cutoff)

        # 4. Reciprocal-Space Component
        u_recip = compute_ewald_recip(
            positions, charges, self.recip_cell, nx, ny, nz, self.alpha, self.volume
        )

        # 5. Self-Energy Correction (Analytic constant)
        u_self = -(self.alpha / np.sqrt(np.pi)) * np.sum(charges**2)

        # 6. Total Energy & Unit Conversion
        # ASE standardizes electrostatic energy using the Coulomb constant
        total_energy_ev = (u_real + u_recip + u_self) * units.Hartree * units.Bohr

        self.results["energy"] = total_energy_ev
        self.results["free_energy"] = total_energy_ev
