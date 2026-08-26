import numpy as np
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from numba import njit, set_num_threads
from vesin import NeighborList

NUM_THREADS_TO_USE = 1
set_num_threads(NUM_THREADS_TO_USE)


@njit(fastmath=True, parallel=False, cache=True)
def compute_lj_numba(
    i_idx, j_idx, distances, atom_types, A_table, B_table, shift_table
) -> tuple[float, np.ndarray]:

    total_energy = 0.0
    energies = np.zeros(len(atom_types))

    # Loop over unique pairs (Vesin full_list=False halves the iterations)
    for k in range(len(i_idx)):
        i = i_idx[k]
        j = j_idx[k]
        r = distances[k]

        t_i = atom_types[i]
        t_j = atom_types[j]

        # Array lookups are nearly free since this tiny table stays in the L1 CPU cache
        A = A_table[t_i, t_j]
        B = B_table[t_i, t_j]
        shift = shift_table[t_i, t_j]

        # Fast 1/r^6 math (replaces division, powers, and branches)
        inv_r2 = 1.0 / (r * r)
        inv_r6 = inv_r2 * inv_r2 * inv_r2

        u = inv_r6 * (A * inv_r6 - B) - shift

        # Distribute the energy
        # Because we only see each pair once, we split the pair energy equally
        u_half = u * 0.5
        total_energy += u
        energies[i] += u_half
        energies[j] += u_half

    return total_energy, energies


class CustomLennardJones(Calculator):
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
        Calculator.__init__(self, **kwargs)

        self.lj_params: dict = lj_parameters
        self.vdw_cutoff = kwargs.get("vdw_cutoff", 12.0)
        self.shifted = kwargs.get("shifted", False)

        # 1. Cache the Vesin object so we don't rebuild it every MC step
        self.neighbor_calculator = NeighborList(cutoff=self.vdw_cutoff, full_list=False)

        # 2. Build Type Lookup Tables (A, B, and static shifts)
        self.unique_labels = list(self.lj_params.keys())
        self.label_to_type = {label: i for i, label in enumerate(self.unique_labels)}

        num_types = len(self.unique_labels)
        self.A_table = np.zeros((num_types, num_types), dtype=np.float64)
        self.B_table = np.zeros((num_types, num_types), dtype=np.float64)
        self.shift_table = np.zeros((num_types, num_types), dtype=np.float64)

        for i, l1 in enumerate(self.unique_labels):
            for j, l2 in enumerate(self.unique_labels):

                # Lorentz-Berthelot mixing calculated exactly ONCE per type pair
                sig = (self.lj_params[l1]["sigma"] + self.lj_params[l2]["sigma"]) / 2.0
                eps = (
                    np.sqrt(self.lj_params[l1]["epsilon"] * self.lj_params[l2]["epsilon"])
                    * units.kB
                )

                # Map to A and B coefficients
                A = 4.0 * eps * (sig**12)
                B = 4.0 * eps * (sig**6)

                self.A_table[i, j] = A
                self.B_table[i, j] = B

                if self.shifted:
                    inv_rc2 = 1.0 / (self.vdw_cutoff * self.vdw_cutoff)
                    inv_rc6 = inv_rc2 * inv_rc2 * inv_rc2
                    self.shift_table[i, j] = inv_rc6 * (A * inv_rc6 - B)

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

        # Map atoms to their integer type for the Numba lookup table
        if "labels" in self.atoms.arrays:  # type: ignore
            labels = [
                self.atoms.symbols[i] if str(label) == "0" else label  # type: ignore
                for i, label in enumerate(self.atoms.arrays["labels"])  # type: ignore
            ]
        else:
            labels = self.atoms.get_chemical_symbols()  # type: ignore

        atom_types = np.array([self.label_to_type[lbl] for lbl in labels], dtype=np.int32)

        # Vesin Neighbor List uses the cached C++ object
        i, j, d = self.neighbor_calculator.compute(
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
            atom_types=atom_types,
            A_table=self.A_table,
            B_table=self.B_table,
            shift_table=self.shift_table,
        )

        self.results["energy"] = total_e_k
        self.results["energies"] = atomic_e_k
        self.results["free_energy"] = total_e_k
