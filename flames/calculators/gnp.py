from dataclasses import dataclass

import numpy as np
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from numba import njit
from vesin import NeighborList


# --- User's Dataclass Definitions ---
@dataclass(slots=True)
class TypeParameters:
    """Class for storing element parameters."""

    s: float
    beta: float
    R: float
    C6: float


@dataclass(slots=True)
class GNPParameters:
    """Class for storing GNP parameters for multiple elements."""

    atom_type: dict[str, TypeParameters]

    @classmethod
    def from_dict(cls, parameters_dict: dict[str, dict[str, float]]) -> "GNPParameters":
        """Create GNPParameters from a dictionary."""
        atom_types = {
            element: TypeParameters(**params) for element, params in parameters_dict.items()
        }
        return cls(atom_type=atom_types)


# --- Numba JIT Kernel ---
@njit(fastmath=True, parallel=False, cache=True)
def compute_gnp_numba(
    i_idx, j_idx, distances, s_vec, beta_vec, r_vec, c6_vec, cutoff, shifted
) -> tuple[float, np.ndarray]:
    total_energy = 0.0
    energies = np.zeros(len(s_vec))

    # Loop over all interacting pairs
    for k in range(len(i_idx)):
        i = i_idx[k]
        j = j_idx[k]
        r = distances[k]

        # Mixing rules (Arithmetic for s/beta, Geometric for R/C6)
        s_mix = (s_vec[i] + s_vec[j]) / 2.0
        beta_mix = (beta_vec[i] + beta_vec[j]) / 2.0
        r_mix = np.sqrt(r_vec[i] * r_vec[j])
        c6_mix = np.sqrt(c6_vec[i] * c6_vec[j])

        # Fast math for R^6 and r^6
        r3 = r * r * r
        r6 = r3 * r3
        R3 = r_mix * r_mix * r_mix
        R6 = R3 * R3

        # Energy calculation
        e_pr = np.exp(-(r - beta_mix) / s_mix)
        e_ld = -c6_mix / (R6 + r6)
        u = e_pr + e_ld

        if shifted:
            # Calculate the potential exactly at the cutoff distance
            e_pr_shift = np.exp(-(cutoff - beta_mix) / s_mix)
            rc3 = cutoff * cutoff * cutoff
            rc6 = rc3 * rc3
            e_ld_shift = -c6_mix / (R6 + rc6)

            u_shift = e_pr_shift + e_ld_shift
            u -= u_shift

        # Divide by 2 because the neighbor list includes both i->j and j->i
        u_half = u / 2.0
        total_energy += u_half
        energies[i] += u_half

    return total_energy, energies


# --- ASE Calculator ---
class CustomGNP(Calculator):
    """
    Custom Generalized Nonbonded Potential (GNP) calculator based on the ASE interface, based
    on the work of Luo and Goddard III, J. Chem. Theory Comput. 2025, 21, 1, 499-515.
    DOI: 10.1021/acs.jctc.4c01435

    Energy is evaluated as:
    E = exp(-(r - beta) / s) - C6 / (R^6 + r^6)
    """

    implemented_properties = ["energy", "energies", "free_energy"]
    default_parameters = {
        "vdw_cutoff": 12.0,
        "shifted": False,
    }
    nolabel = True

    def __init__(self, gnp_parameters: GNPParameters, **kwargs):
        """
        Parameters
        ----------
        gnp_parameters : GNPParameters
            Dataclass instance containing the GNP parameters for elements.
        vdw_cutoff : float, optional
            Cutoff distance for the nonbonded interactions. Default is 12.0 Angstroms.
        shifted : bool, optional
            Whether to shift the potential to zero at the cutoff. Default is False.
        """
        Calculator.__init__(self, **kwargs)

        self.gnp_params: GNPParameters = gnp_parameters
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
            labels = self.atoms.info.get("labels", self.atoms.get_chemical_symbols())  # type: ignore

        # Dataclass Unpacking for Numba
        s_vec = np.array([self.gnp_params.atom_type[sym].s for sym in labels], dtype=np.float64)
        beta_vec = np.array(
            [self.gnp_params.atom_type[sym].beta for sym in labels], dtype=np.float64
        )
        r_vec = np.array([self.gnp_params.atom_type[sym].R for sym in labels], dtype=np.float64)
        c6_vec = np.array([self.gnp_params.atom_type[sym].C6 for sym in labels], dtype=np.float64)

        # Extract positions and cell data as raw NumPy arrays
        positions = self.atoms.positions  # type: ignore
        cell = self.atoms.cell.array  # type: ignore

        # Vesin Neighbor List calculation
        calculator = NeighborList(cutoff=self.vdw_cutoff, full_list=True)
        i, j, d = calculator.compute(points=positions, box=cell, periodic=True, quantities="ijd")

        # Numba JIT Energy Math (returns kcal/mol)
        total_e_kcal, atomic_e_kcal = compute_gnp_numba(
            i_idx=i,
            j_idx=j,
            distances=d,
            s_vec=s_vec,
            beta_vec=beta_vec,
            r_vec=r_vec,
            c6_vec=c6_vec,
            cutoff=self.vdw_cutoff,
            shifted=self.shifted,
        )

        # Convert kcal/mol -> eV
        kcal_to_eV = units.kcal / units.mol

        total_e_eV = total_e_kcal * kcal_to_eV
        atomic_e_eV = atomic_e_kcal * kcal_to_eV

        # Store in ASE results dictionary
        self.results["energy"] = total_e_eV
        self.results["energies"] = atomic_e_eV
        self.results["free_energy"] = total_e_eV
