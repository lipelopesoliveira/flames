import numpy as np
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from numba import njit
from vesin import NeighborList
from dataclasses import dataclass

# --- User's Dataclass Definitions ---
@dataclass(slots=True)
class TypeParameters:
    s: float
    beta: float
    R: float
    C6: float

@dataclass(slots=True)
class GNPParameters:
    atom_type: dict[str, TypeParameters]

    @classmethod
    def from_dict(cls, parameters_dict: dict[str, dict[str, float]]) -> 'GNPParameters':
        atom_types = {element: TypeParameters(**params) for element, params in parameters_dict.items()}
        return cls(atom_type=atom_types)

# --- Numba JIT Kernel (Precomputed + Half-List) ---
@njit(fastmath=True, parallel=False, cache=True)
def compute_gnp_halflist(
    i_idx, j_idx, distances, type_indices,
    s_mix_mat, beta_mix_mat, r_mix_mat, c6_mix_mat, u_shift_mat,
    shifted
) -> tuple[float, np.ndarray]:
    
    total_energy = 0.0
    n_atoms = len(type_indices)
    energies = np.zeros(n_atoms, dtype=np.float64)

    # Loop over unique interacting pairs (half-list)
    for k in range(len(i_idx)):
        i = i_idx[k]
        j = j_idx[k]
        r = distances[k]

        t_i = type_indices[i]
        t_j = type_indices[j]

        # Instant lookup of precomputed mixing rules
        s_mix = s_mix_mat[t_i, t_j]
        beta_mix = beta_mix_mat[t_i, t_j]
        R_mix = r_mix_mat[t_i, t_j]
        C6_mix = c6_mix_mat[t_i, t_j]

        # Fast math for R^6 and r^6
        r3 = r * r * r
        r6 = r3 * r3
        R3 = R_mix * R_mix * R_mix
        R6 = R3 * R3

        # Energy calculation
        e_pr = np.exp(-(r - beta_mix) / s_mix)
        e_ld = -C6_mix / (R6 + r6)
        u = e_pr + e_ld

        # Instant lookup of precomputed shift energy
        if shifted:
            u -= u_shift_mat[t_i, t_j]

        # --- No Double Counting Logic ---
        # Add the full pairwise energy to the total system energy
        total_energy += u
        
        # Split the pairwise energy equally between the two participating atoms
        u_half = u / 2.0
        energies[i] += u_half
        energies[j] += u_half

    return total_energy, energies

# --- ASE Calculator ---
class CustomGNP(Calculator):
    implemented_properties = ["energy", "energies", "free_energy"]
    default_parameters = {
        "vdw_cutoff": 12.0,
        "shifted": True,
    }
    nolabel = True

    def __init__(self, gnp_parameters: GNPParameters, **kwargs):
        Calculator.__init__(self, **kwargs)
        
        self.gnp_params: GNPParameters = gnp_parameters
        self.vdw_cutoff = kwargs.get("vdw_cutoff", 12.0)
        self.shifted = kwargs.get("shifted", True)
        
        # Cache for atom type indices
        self._cached_labels = None
        self._type_indices = None

        self._precompute_matrices()

    def _precompute_matrices(self):
        """Builds 2D arrays of pre-mixed parameters for instant Numba lookup."""
        labels = list(self.gnp_params.atom_type.keys())
        n_types = len(labels)
        
        self.label_to_id = {label: idx for idx, label in enumerate(labels)}

        self.s_mix_mat = np.zeros((n_types, n_types), dtype=np.float64)
        self.beta_mix_mat = np.zeros((n_types, n_types), dtype=np.float64)
        self.R_mix_mat = np.zeros((n_types, n_types), dtype=np.float64)
        self.C6_mix_mat = np.zeros((n_types, n_types), dtype=np.float64)
        self.u_shift_mat = np.zeros((n_types, n_types), dtype=np.float64)

        cutoff = self.vdw_cutoff
        rc3 = cutoff * cutoff * cutoff
        rc6 = rc3 * rc3

        for i, lab_i in enumerate(labels):
            for j, lab_j in enumerate(labels):
                pi = self.gnp_params.atom_type[lab_i]
                pj = self.gnp_params.atom_type[lab_j]

                # Mixing Rules
                s_m = (pi.s + pj.s) / 2.0
                beta_m = (pi.beta + pj.beta) / 2.0
                r_m = np.sqrt(pi.R * pj.R)
                c6_m = np.sqrt(pi.C6 * pj.C6)

                self.s_mix_mat[i, j] = s_m
                self.beta_mix_mat[i, j] = beta_m
                self.R_mix_mat[i, j] = r_m
                self.C6_mix_mat[i, j] = c6_m

                # Precalculate the shift energy exactly at the cutoff distance
                R3 = r_m * r_m * r_m
                R6 = R3 * R3
                e_pr_shift = np.exp(-(cutoff - beta_m) / s_m)
                e_ld_shift = -c6_m / (R6 + rc6)
                
                self.u_shift_mat[i, j] = e_pr_shift + e_ld_shift

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        if properties is None:
            properties = self.implemented_properties

        Calculator.calculate(self, atoms, properties, system_changes)
        np.seterr(invalid="ignore")

        if "labels" in self.atoms.arrays:  # type: ignore
            labels = [
                self.atoms.symbols[i] if str(label) == "0" else label  # type: ignore
                for i, label in enumerate(self.atoms.arrays["labels"])  # type: ignore
            ]
        else:
            labels = self.atoms.get_chemical_symbols()  # type: ignore

        # Convert labels to integer array only if they change
        if self._cached_labels != labels:
            self._cached_labels = labels
            self._type_indices = np.array(
                [self.label_to_id[sym] for sym in labels], dtype=np.int32
            )

        positions = self.atoms.positions  # type: ignore
        cell = self.atoms.cell.array  # type: ignore

        # --- IMPORTANT CHANGE: full_list=False ---
        # Vesin will now only return unique i-j pairs (e.g., i < j)
        calculator = NeighborList(cutoff=self.vdw_cutoff, full_list=False)
        i, j, d = calculator.compute(points=positions, box=cell, periodic=True, quantities="ijd")

        # Numba JIT Energy Math
        total_e_kcal, atomic_e_kcal = compute_gnp_halflist(
            i_idx=i,
            j_idx=j,
            distances=d,
            type_indices=self._type_indices,
            s_mix_mat=self.s_mix_mat,
            beta_mix_mat=self.beta_mix_mat,
            r_mix_mat=self.R_mix_mat,
            c6_mix_mat=self.C6_mix_mat,
            u_shift_mat=self.u_shift_mat,
            shifted=self.shifted,
        )

        # Convert kcal/mol -> eV
        kcal_to_eV = units.kcal / units.mol
        
        self.results["energy"] = total_e_kcal * kcal_to_eV
        self.results["energies"] = atomic_e_kcal * kcal_to_eV
        self.results["free_energy"] = self.results["energy"]