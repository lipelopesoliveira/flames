import numpy as np
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from numba import njit, set_num_threads
from vesin import NeighborList

NUM_THREADS_TO_USE = 1
set_num_threads(NUM_THREADS_TO_USE)


@njit(fastmath=True, parallel=False, cache=True)
def compute_lj_numba(
    i_idx, j_idx, distances, sigma_vec, epsilon_vec, cutoff, shifted
) -> tuple[float, np.ndarray]:
    total_energy = 0.0
    energies = np.zeros(len(sigma_vec))

    # Loop over all interacting pairs
    for k in range(len(i_idx)):
        i = i_idx[k]
        j = j_idx[k]
        r = distances[k]

        # Lorentz-Berthelot mixing
        sig = (sigma_vec[i] + sigma_vec[j]) / 2.0
        eps = np.sqrt(epsilon_vec[i] * epsilon_vec[j])

        # Optimization: Multiplication is faster than exponentiation (**6) in C/Numba
        s_over_r = sig / r
        s3 = s_over_r * s_over_r * s_over_r
        s6 = s3 * s3

        u = 4.0 * eps * (s6 * s6 - s6)

        if shifted:
            s_over_rc = sig / cutoff
            s_rc3 = s_over_rc * s_over_rc * s_over_rc
            s_rc6 = s_rc3 * s_rc3
            u_shift = 4.0 * eps * (s_rc6 * s_rc6 - s_rc6)
            u -= u_shift

        # Divide by 2 because the neighbor list includes both i->j and j->i
        u_half = u / 2.0
        total_energy += u_half
        energies[i] += u_half

    return total_energy, energies


@njit(fastmath=True, inline="always")
def fast_mic_sq(dx, dy, dz, cell, inv_cell) -> tuple[float, float, float]:
    """Fractional coordinate rounding. Fastest, but vulnerable to acute angles."""
    # Convert to fractional
    fx = dx * inv_cell[0, 0] + dy * inv_cell[1, 0] + dz * inv_cell[2, 0]
    fy = dx * inv_cell[0, 1] + dy * inv_cell[1, 1] + dz * inv_cell[2, 1]
    fz = dx * inv_cell[0, 2] + dy * inv_cell[1, 2] + dz * inv_cell[2, 2]

    # Nearest image shift
    fx -= np.round(fx)
    fy -= np.round(fy)
    fz -= np.round(fz)

    # Convert back to Cartesian
    rx = fx * cell[0, 0] + fy * cell[1, 0] + fz * cell[2, 0]
    ry = fx * cell[0, 1] + fy * cell[1, 1] + fz * cell[2, 1]
    rz = fx * cell[0, 2] + fy * cell[1, 2] + fz * cell[2, 2]

    return rx * rx + ry * ry + rz * rz


@njit(fastmath=True, inline="always")
def robust_mic_sq(dx, dy, dz, cell) -> float:
    """27-image explicit search. Slower, but perfect for skewed triclinic cells."""
    min_r_sq = 1e20

    for sx in range(-1, 2):
        for sy in range(-1, 2):
            for sz in range(-1, 2):
                rx = dx + sx * cell[0, 0] + sy * cell[1, 0] + sz * cell[2, 0]
                ry = dy + sx * cell[0, 1] + sy * cell[1, 1] + sz * cell[2, 1]
                rz = dz + sx * cell[0, 2] + sy * cell[1, 2] + sz * cell[2, 2]

                r_sq = rx * rx + ry * ry + rz * rz
                min_r_sq = min(min_r_sq, r_sq)

    return min_r_sq


@njit(
    fastmath=True,
)
def numba_neighbor_list(positions, cell, inv_cell, cutoff, use_robust_mic=True):
    n = positions.shape[0]
    cutoff_sq = cutoff * cutoff

    # Pre-allocate arrays
    max_pairs = n * 200
    i_out = np.empty(max_pairs, dtype=np.int32)
    j_out = np.empty(max_pairs, dtype=np.int32)
    d_out = np.empty(max_pairs, dtype=np.float64)

    idx = 0

    for i in range(n):
        for j in range(i + 1, n):

            # Raw distance vector
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            dz = positions[j, 2] - positions[i, 2]

            # --- The MIC Toggle ---
            if use_robust_mic:
                r_sq = robust_mic_sq(dx, dy, dz, cell)
            else:
                r_sq = fast_mic_sq(dx, dy, dz, cell, inv_cell)

            # --- Cutoff & Array Management ---
            if r_sq <= cutoff_sq:
                # Dynamic Reallocation (Safety Net)
                if idx + 2 > max_pairs:
                    max_pairs = max_pairs * 2
                    new_i = np.empty(max_pairs, dtype=np.int32)
                    new_j = np.empty(max_pairs, dtype=np.int32)
                    new_d = np.empty(max_pairs, dtype=np.float64)

                    new_i[:idx] = i_out[:idx]
                    new_j[:idx] = j_out[:idx]
                    new_d[:idx] = d_out[:idx]

                    i_out = new_i
                    j_out = new_j
                    d_out = new_d

                d = np.sqrt(r_sq)

                # i -> j
                i_out[idx] = i
                j_out[idx] = j
                d_out[idx] = d
                idx += 1

                # j -> i
                i_out[idx] = j
                j_out[idx] = i
                d_out[idx] = d
                idx += 1

    return i_out[:idx], j_out[:idx], d_out[:idx]


class CustomLennardJones(Calculator):
    """
    Custom Lennard Jones potential calculator based on the ASE calculator interface.
    This method is intended to be as close as possible to RASPA2 implementation.

    The fundamental definition of this potential is a pairwise energy:

    ``u_ij = 4 epsilon ( sigma^12/r_ij^12 - sigma^6/r_ij^6 )``

    For convenience, we'll use d_ij to refer to "distance vector" and
    ``r_ij`` to refer to "scalar distance". So, with position vectors `r_i`:

    ``r_ij = | r_j - r_i | = | d_ij |``


    We have to ensure that the potential goes to zero smoothly as an atom moves
    across the cutoff threshold, otherwise the potential is not continuous.
    In cases where the cutoff is so large that u_ij is very small at the cutoff
    this is automatically ensured, but in general, `u_ij(rc) != 0`.

    This implementation deal with this by shifting the pairwise energy

    ``u'_ij = u_ij - u_ij(rc)``

    which ensures that it is precisely zero at the cutoff. However, this means
    that the energy effectively depends on the cutoff, which might lead to
    unexpected results!
    """

    implemented_properties = ["energy", "energies"]
    default_parameters = {
        "epsilon": 1.0,
        "sigma": 1.0,
        "rc": None,
        "ro": None,
        "shifted": False,
    }
    nolabel = True

    def __init__(self, lj_parameters: dict, **kwargs):
        """
        Parameters
        ----------
        lj_parameters : dict
            Dictionary containing the Lennard-Jones parameters.
            The parameters should be in the form:
            "O": {
                "sigma": 3.03315,  # In Angstroms
                "epsilon": 48.1581 # In Kelvin
                }
        vdw_cutoff : float, optional
            Cutoff distance for the van der Waals interactions.
            Default is 12.0 Angstroms.
        shifted : bool, optional
            Whether to apply a shift to the potential
            to ensure it goes to zero at the cutoff.
        """

        Calculator.__init__(self, **kwargs)

        self.lj_params: dict = lj_parameters
        self.vdw_cutoff = kwargs.get("vdw_cutoff", 12.0)
        self.shifted = kwargs.get("shifted", False)

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
    ):

        if properties is None:
            properties = self.implemented_properties

        Calculator.calculate(self, atoms, properties, system_changes)

        np.seterr(invalid="ignore")

        # Use custom atom labels if available, otherwise fall back to chemical symbols
        if "labels" in self.atoms.arrays:  # type: ignore
            labels = [
                self.atoms.symbols[i] if str(label) == "0" else label  # type: ignore
                for i, label in enumerate(self.atoms.arrays["labels"])  # type: ignore
            ]
        else:
            labels = self.atoms.get_chemical_symbols()  # type: ignore

        sigma_vec = np.array([self.lj_params[s]["sigma"] for s in labels])
        epsilon_vec = np.array([self.lj_params[s]["epsilon"] * units.kB for s in labels])

        # Vesin Neighbor List (faster than Numba JIT, but requires vesin package)
        neighbor_calculator = NeighborList(cutoff=self.vdw_cutoff, full_list=True)
        i, j, d = neighbor_calculator.compute(
            points=self.atoms.positions,  # type: ignore
            box=self.atoms.cell.array,  # type: ignore
            periodic=True,
            quantities="ijd",
        )

        # Numba JIT Energy Math
        total_e_k, atomic_e_k = compute_lj_numba(
            i_idx=i,
            j_idx=j,
            distances=d,
            sigma_vec=sigma_vec,
            epsilon_vec=epsilon_vec,
            cutoff=self.vdw_cutoff,
            shifted=self.shifted,
        )

        self.results["energy"] = total_e_k
        self.results["energies"] = atomic_e_k
        self.results["free_energy"] = total_e_k
