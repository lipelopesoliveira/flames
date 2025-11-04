from typing import Union

import ase
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from scipy import special  # For erfc


class EwaldSum(Calculator):
    """
    A generalized, vectorized Ewald summation calculator for ASE.

    This class calculates the electrostatic energy of a periodic structure
    using the Ewald summation method. It is generalized for any unit cell
    and uses numpy vectorization for speed.

    It calculates per-atom energies, and the total energy is the sum of these.

    The energy is E = E_real + E_reciprocal + E_self.
    """

    implemented_properties = [
        "energy",
        "free_energy",
        "energies",
        # 'forces', 'stress', # to be implemented
    ]

    def __init__(self, R_cutoff: float, G_cutoff_N: float, alpha: float, **kwargs) -> None:
        """
        Initializes the EwaldSum calculator.

        Args:
            R_cutoff: The cutoff radius for the real-space sum (Angstroms).
            G_cutoff_N: The *integer* cutoff for the reciprocal-space sum.
                        (sum over nx^2 + ny^2 + nz^2 <= G_cutoff_N^2).
            alpha: The Ewald splitting parameter (in 1/Angstroms).
                   A common choice is 5.0 / L (for a cell of length L).
        """

        Calculator.__init__(self, **kwargs)

        self.R_cutoff = R_cutoff
        self.G_cutoff_N = G_cutoff_N
        self.alpha = alpha

    def _getNMax(self, cell, volume) -> tuple[int, int, int]:
        """
        Calculates the number of maximum unit cells based on the "height"
        of the cell perpendicular to each pair of vectors and the Real
        space cutoff (R_cutoff)

        Returns
        -------
        N_max: tuple[int, int, int]
            The nx_max, ny_max, and nz_max numbers of unit cells

        """
        a, b, c = cell
        cross_bc = np.cross(b, c)
        cross_ac = np.cross(a, c)
        cross_ab = np.cross(a, b)

        # Add safety checks for 2D/1D systems
        norm_cross_bc = np.linalg.norm(cross_bc)
        norm_cross_ac = np.linalg.norm(cross_ac)
        norm_cross_ab = np.linalg.norm(cross_ab)

        h_a = volume / norm_cross_bc if norm_cross_bc > 1e-9 else np.inf
        h_b = volume / norm_cross_ac if norm_cross_ac > 1e-9 else np.inf
        h_c = volume / norm_cross_ab if norm_cross_ab > 1e-9 else np.inf

        N_max = np.ceil(self.R_cutoff / np.array([h_a, h_b, h_c])).astype(int).tolist()

        return N_max

    def _realspaceEnergy(self, structure: ase.Atoms) -> np.ndarray:
        """
        Calculates the real-space energy (vectorized) per atom.
        E_real_i = 0.5 * sum_j,n' [ q_i*q_j * erfc(alpha*|r_ij + n|) / |r_ij + n| ]

        Returns:
            np.ndarray: Array of size (N_atoms) with raw real-space energy
                        per atom (in e^2/A).
        """
        n_atoms = len(structure)
        cell = structure.cell.array
        positions = structure.get_positions(wrap=True)
        charges = structure.get_initial_charges()
        real_energies = np.zeros(n_atoms)

        # Find max translation vectors (n_x, n_y, n_z) needed
        Nx_max, Ny_max, Nz_max = self._getNMax(cell=cell, volume=structure.cell.volume)

        # Create grid of n-vectors [nx, ny, nz]
        n_x_range = np.arange(-Nx_max, Nx_max + 1, dtype=int)
        n_y_range = np.arange(-Ny_max, Ny_max + 1, dtype=int)
        n_z_range = np.arange(-Nz_max, Nz_max + 1, dtype=int)
        nx_grid, ny_grid, nz_grid = np.meshgrid(n_x_range, n_y_range, n_z_range, indexing="ij")

        # (N_cells, 3) array of n-vectors
        n_vectors_flat = np.stack([nx_grid.ravel(), ny_grid.ravel(), nz_grid.ravel()], axis=-1)

        # (N_cells, 3) array of Cartesian translation vectors
        nv_vectors_cart = np.dot(n_vectors_flat, cell)

        # (N_cells) boolean array identifying n=0 vector
        n_is_zero = ~np.any(n_vectors_flat, axis=1)

        for i in range(n_atoms):
            i_energy_raw = 0.0
            for j in range(n_atoms):
                qi_qj = charges[i] * charges[j]
                rij_vec = positions[i] - positions[j]

                # (N_cells, 3) array of all translated r_ij vectors
                rv_vectors = rij_vec + nv_vectors_cart

                # (N_cells) array of distances
                r_norms = np.linalg.norm(rv_vectors, axis=1)

                if i == j:
                    # For self-interaction, only consider n != 0
                    r_norms = r_norms[~n_is_zero]

                # Apply cutoff
                r_norms_inside_cutoff = r_norms[r_norms <= self.R_cutoff]

                # Filter out r=0 (e.g., overlapping atoms at n=0)
                r_norms_valid = r_norms_inside_cutoff[r_norms_inside_cutoff > 1e-9]

                if r_norms_valid.size > 0:
                    with np.errstate(divide="ignore"):
                        terms = special.erfc(self.alpha * r_norms_valid) / r_norms_valid
                        i_energy_raw += qi_qj * np.sum(terms)

            # The per-atom energy is 1/2 of its sum with all other atoms
            real_energies[i] = 0.5 * i_energy_raw

        return real_energies

    def _reciprocalEnergy(self, structure: ase.Atoms) -> np.ndarray:
        """
        Calculates the reciprocal-space energy (vectorized) per atom.
        E_recip_i = q_i * (2*pi/V) * sum_k!=0 [ A(k) * Re( exp(i*k.r_i) * Q(k)* ) ]

        Returns:
            np.ndarray: Array of size (N_atoms) with raw recip. energy
                        per atom (in e^2/A).
        """
        n_atoms = len(structure)
        volume = structure.cell.volume
        reciprocal_cell_matrix = 2.0 * np.pi * np.linalg.inv(structure.cell.array).T
        positions = structure.get_positions(wrap=True)
        charges = structure.get_initial_charges()

        Nmax = int(np.ceil(self.G_cutoff_N))
        G_cutoff_N_sq = self.G_cutoff_N**2

        # Create grid of n-vectors [nx, ny, nz]
        n_range = np.arange(-Nmax, Nmax + 1, dtype=int)
        nx_grid, ny_grid, nz_grid = np.meshgrid(n_range, n_range, n_range, indexing="ij")
        n_vectors_flat = np.stack([nx_grid.ravel(), ny_grid.ravel(), nz_grid.ravel()], axis=-1)

        # --- Filter n-vectors ---
        # 1. Remove n=0 term
        n_is_not_zero = np.any(n_vectors_flat, axis=1)
        n_vectors_nonzero = n_vectors_flat[n_is_not_zero]

        # 2. Apply spherical cutoff in index space
        n_norm_sq = np.sum(n_vectors_nonzero**2, axis=1)
        n_vectors_valid = n_vectors_nonzero[n_norm_sq <= G_cutoff_N_sq]

        if n_vectors_valid.shape[0] == 0:
            return np.zeros(n_atoms)  # No k-vectors in cutoff

        # --- Calculate k-vectors and A(k) term ---
        # (N_k_vectors, 3) array of Cartesian k-vectors
        kv_cartesian = np.dot(n_vectors_valid, reciprocal_cell_matrix)

        # (N_k_vectors) array of k-magnitudes squared
        k_norm_sq = np.sum(kv_cartesian**2, axis=1)

        # A(k) term = (1/k^2) * exp(-k^2 / (4*alpha^2))
        # Note: This is A(k) *without* the (2*pi/V) prefactor
        Ak_terms = (1.0 / k_norm_sq) * np.exp(-k_norm_sq / (4.0 * self.alpha**2))

        # --- Calculate Per-Atom Term ---
        # We need: q_i * Re( exp(i*k.r_i) * Q(k)* )

        # (N_atoms, N_k_vectors) array of exp(i * k.r_i)
        k_dot_r_matrix = np.dot(positions, kv_cartesian.T)
        exp_k_dot_r = np.exp(1j * k_dot_r_matrix)

        # (N_k_vectors) array of Q(k) = sum_j q_j * exp(i * k.r_j)
        Q_vector = np.dot(charges, exp_k_dot_r)

        # (N_k_vectors) array of Q(k)* (conjugate)
        Q_conj_vector = np.conjugate(Q_vector)

        # (N_atoms, N_k_vectors) array of exp(i*k.r_i) * Q(k)*
        term_in_brackets = exp_k_dot_r * Q_conj_vector

        # (N_atoms, N_k_vectors) array of Re[ ... ]
        real_term = np.real(term_in_brackets)

        # (N_atoms) array: sum_k [ Ak * Re(...) ]
        sum_over_k = np.dot(real_term, Ak_terms)

        # (N_atoms) array: q_i * sum_k [ ... ]
        recip_energies_raw = charges * sum_over_k

        # Apply prefactor
        rec_energy_constant = 2 * np.pi / volume

        return rec_energy_constant * recip_energies_raw

    def _selfEnergy(self, charges: np.ndarray) -> np.ndarray:
        """
        Calculates the self-energy correction term per atom.
        E_self_i = - (alpha / sqrt(pi)) * q_i^2

        Returns:
            np.ndarray: Array of size (N_atoms) with raw self-energy
                        per atom (in e^2/A).
        """
        # E_self_i = - (alpha / sqrt(pi)) * q_i^2
        return -(self.alpha / np.sqrt(np.pi)) * (charges**2)

    def calculate(
        self,
        atoms: Union[ase.Atoms, None] = None,
        properties: list[str] = ["energy", "energies"],
        system_changes=all_changes,
    ):

        Calculator.calculate(self, atoms, properties, system_changes)

        COULOMB_CONSTANT_eV_A = 14.399645353  # This converts energy from (e^2 / A) to (eV)

        charges = self.atoms.get_initial_charges()  # type: ignore

        # --- Calculate all per-atom components (raw, in e^2/A) ---
        real_energies_raw = self._realspaceEnergy(self.atoms)  # type: ignore
        recip_energies_raw = self._reciprocalEnergy(self.atoms)  # type: ignore
        self_energies_raw = self._selfEnergy(charges)

        # --- Sum them to get the total per-atom array (raw) ---
        total_energies_raw = real_energies_raw + recip_energies_raw + self_energies_raw

        # --- Convert to eV ---
        total_energies_ev = total_energies_raw * COULOMB_CONSTANT_eV_A

        # --- Store results ---
        if "energies" in properties:
            self.results["energies"] = total_energies_ev

        if "energy" in properties:
            total_energy_ev = np.sum(total_energies_ev)
            self.results["energy"] = total_energy_ev
            self.results["free_energy"] = total_energy_ev
