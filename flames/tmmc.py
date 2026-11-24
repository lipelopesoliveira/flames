from __future__ import annotations

import datetime
import json
import os
from copy import deepcopy

import ase
import numpy as np
from ase import units
from ase.calculators import calculator
from ase.io.trajectory import TrajectoryReader
from tqdm import tqdm

from flames import VERSION
from flames.adsorbate import Adsorbate
from flames.base_simulator import BaseSimulator
from flames.logger import TMMCLogger
from flames.operations import (
    check_overlap_vesin,
    random_mol_insertion,
)


class TMMC(BaseSimulator):
    """
    Base class for transition matrix Monte Carlo (TMMC) simulations using ASE.

    This class implements TMMC deletion/insertion moves, recording the
    deletion/insertion energies of the adsorbate.

    :param calculator:
        The calculator to use for energy calculations. Can be any ASE-compatible calculator.
        The output of the calculator should be in eV.
    :type calculator: ase.calculators.calculator.Calculator

    :param adsorbate_calculator:
            The calculator to use for adsorbate energy calculations. Can be any ASE-compatible calculator.
            The output of the calculator should be in eV.
    :type adsorbate_calculator: ase.calculators.calculator.Calculator | None

    :param framework_atoms:
        The framework structure as an ASE Atoms object.
    :type framework_atoms: ase.Atoms

    :param adsorbates:
        The adsorbate structure(s) as an Adsorbate object or list of Adsorbate objects.
    :type adsorbates: Adsorbate | list[Adsorbate]

    :param temperature:
        Temperature of the ideal reservoir in Kelvin.
    :type temperature: float

    :param pressure:
        Pressure of the ideal reservoir in Pascal.
    :type pressure: float

    :param device:
        Device to run the simulation on, e.g., ``'cpu'`` or ``'cuda'``.
    :type device: str

    :param vdw_radii:
        Van der Waals radii for the atoms in the framework and adsorbate.
        Should be an array of the same length as the number of atomic numbers in ASE.
    :type vdw_radii: np.ndarray

    :param vdw_factor:
        Factor to scale the Van der Waals radii. Default is ``0.6``.
    :type vdw_factor: float, optional

    :param save_frequency:
        Frequency at which to save the simulation state and results. Default is ``100``.
    :type save_frequency: int, optional

    :param output_to_file:
        If ``True``, writes the output to a file named ``output_{temperature}_{pressure}.out`` in the ``results`` directory. Default is ``True``.
    :type output_to_file: bool, optional

    :param output_folder:
        Folder to save the output files. If ``None``, a folder named ``results_<T>_<P>`` will be created.
    :type output_folder: str or None, optional

    :param debug:
        If ``True``, prints detailed debug information during the simulation. Default is ``False``.
    :type debug: bool, optional

    :param random_seed:
        Random seed for reproducibility. Default is ``None`` and will generate a random seed automatically if not provided.
    :type random_seed: int or None, optional

    :param cutoff_radius:
        Interaction potential cut-off radius used to estimate the minimum unit cell. Default is ``6.0``.
    :type cutoff_radius: float, optional
    """

    def __init__(
        self,
        calculator: calculator.Calculator,
        framework_atoms: ase.Atoms,
        adsorbates: Adsorbate | list[Adsorbate],
        temperature: float,
        pressure: float,
        device: str,
        vdw_radii: np.ndarray,
        adsorbate_calculator: calculator.Calculator | None = None,
        vdw_factor: float = 0.6,
        framework_energy: float | None = None,
        adsorbate_energy: float | None = None,
        max_deltaE: float = 1.555,
        save_frequency: int = 100,
        output_to_file: bool = True,
        output_folder: str | None = None,
        debug: bool = False,
        random_seed: int | None = None,
        cutoff_radius: float = 6.0,
        automatic_supercell: bool = False,
    ) -> None:
        """
        Initialize the transition matrix Monte Carlo (TMMC) simulation.
        """

        super().__init__(
            calculator=calculator,
            adsorbate_calculator=adsorbate_calculator,
            framework_atoms=framework_atoms,
            adsorbates=adsorbates,
            temperature=temperature,
            pressure=pressure,
            device=device,
            vdw_radii=vdw_radii,
            vdw_factor=vdw_factor,
            framework_energy=framework_energy,
            adsorbate_energy=adsorbate_energy,
            max_deltaE=max_deltaE,
            save_frequency=save_frequency,
            save_rejected=False,
            output_to_file=output_to_file,
            output_folder=output_folder,
            debug=debug,
            fugacity_coeff=0.0,
            random_seed=random_seed,
            cutoff_radius=cutoff_radius,
            automatic_supercell=automatic_supercell,
        )

        self.logger: TMMCLogger = TMMCLogger(simulation=self, output_file=self.out_file)

        self.start_time = datetime.datetime.now()

        # Tracking number of adsorbates per species
        self.n_adsorbates: dict[str, int] = {adsorbate.name: 0 for adsorbate in self.adsorbates}

        # Parameters for storing the main results during the simulation (partitioned by adsorbate)
        self.total_ins_energy_list: dict[str, list[float]] = {
            ads.name: [] for ads in self.adsorbates
        }
        self.total_del_energy_list: dict[str, list[float]] = {
            ads.name: [] for ads in self.adsorbates
        }
        self.total_energy_list: dict[str, list[list[float]]] = {
            ads.name: [] for ads in self.adsorbates
        }
        self.volume_list: list[float] = []

        # Base iteration for restarting the simulation. This is for tracking the iteration count only
        self._base_iteration: int = 0

        # Pre-computed fugacity-independent Boltzmann weights for insertion and deletion moves
        self.insertion_weights: dict[int, list[np.ndarray]] = {}
        self.deletion_weights: dict[int, list[np.ndarray]] = {}

    @property
    def base_iteration(self) -> int:
        """
        Get the base iteration for the TMMC simulation.

        Returns
        -------
        int
            The base iteration count.
        """
        return self._base_iteration

    @base_iteration.setter
    def base_iteration(self, iteration: int) -> None:
        """
        Set the base iteration for the TMMC simulation.

        Parameters
        ----------
        iteration : int
            The base iteration count to set.
        """
        self._base_iteration = iteration

    def get_number_of_adsorbates(self, system: ase.Atoms | None = None) -> dict[str, int]:
        """
        Get the number of adsorbates in the current system.
        It considers the possibility of having multiple adsorbate species in the simulation.
        """
        if not system:
            system = self.current_system

        n_adsorbate_by_type = {}

        for adsorbate in self.adsorbates:
            adsorbate_indices = np.where(system.get_tags() == adsorbate.tag)[0]
            n_adsorbate_by_type[adsorbate.name] = int(
                len(adsorbate_indices) / len(adsorbate.structure)
            )

        return n_adsorbate_by_type

    def get_macrostate_str(self) -> str:
        """
        Return a string representation of the current multi-component macrostate.
        Example: For a binary mixture with 1 CO2 and 2 H2O, returns "0001_0002".
        """
        return "_".join([f"{self.n_adsorbates[ads.name]:04d}" for ads in self.adsorbates])

    def save_results(
        self,
        file_name: str | None = None,
    ) -> None:
        """
        Save a json file with the main results of the simulation.

        Parameters
        ----------
        file_name : str
            Name of the output file. Default uses macrostate string format.
        """
        macrostate_str = self.get_macrostate_str()

        if file_name is None:
            file_name = f"results_{self.T}_{macrostate_str}.json"

        results = {
            "simulation": {
                "code_version": VERSION,
                "random_seed": self.random_seed,
                "temperature_K": self.T,
                "macrostate": self.n_adsorbates,
                "n_steps": len(self.volume_list),
                "enlapsed_time_hours": (datetime.datetime.now() - self.start_time).total_seconds()
                / 3600,
            },
        }

        with open(os.path.join(self.out_folder, file_name), "w") as f:
            json.dump(results, f, indent=4)

    def set_state(self, state: ase.Atoms) -> None:
        """
        Set the current state of the simulation to a given ASE Atoms object.
        This method updates the current system, total energy, and number of adsorbates.

        Parameters
        ----------
        state : ase.Atoms
            The ASE Atoms object representing the new state of the system.
        """
        self.current_system = deepcopy(state)
        self.current_system.calc = self.calculator
        self.current_total_energy = self.current_system.get_potential_energy()
        self.n_adsorbates = self.get_number_of_adsorbates(system=state)

    def _save_state(self) -> None:
        """
        Save the simulation trajectory and generated energies.
        """

        macrostate_str = self.get_macrostate_str()

        for ads in self.adsorbates:
            np.save(
                os.path.join(self.out_folder, f"ins_energy_{ads.name}_{macrostate_str}.npy"),
                np.array(self.total_ins_energy_list[ads.name]),
            )
            np.save(
                os.path.join(self.out_folder, f"del_energy_{ads.name}_{macrostate_str}.npy"),
                np.array(self.total_del_energy_list[ads.name]),
            )
            np.save(
                os.path.join(self.out_folder, f"total_energy_{ads.name}_{macrostate_str}.npy"),
                np.array(self.total_energy_list[ads.name]),
            )

        np.save(
            os.path.join(self.out_folder, f"volume_{macrostate_str}.npy"),
            np.array(self.volume_list),
        )

    def try_virtual_insertion(self, adsorbate_tag: int) -> tuple[float, float]:
        """
        Try to insert a new adsorbate molecule into the framework.
        This method randomly places the adsorbate in the framework and checks for van der Waals overlap.
        If there is no overlap, it calculates the new potential energy and decides whether to accept the insertion
        based on the acceptance criteria.

        Parameters
        ----------
        adsorbate_tag : int
            The tag of the adsorbate molecule being inserted.

        Returns
        -------
        deltaE: float
            Insertion energy.
        e_new: float
            New total energy.
        """
        adsorbate = next((ads for ads in self.adsorbates if ads.tag == adsorbate_tag), None)

        if adsorbate is None:
            raise ValueError(f"Adsorbate with tag {adsorbate_tag} not found.")

        # Get the structure to be safe in case adsorbate has several conformers or structures
        adsorbate_structure = adsorbate.structure

        atoms_trial = random_mol_insertion(
            self.current_system, adsorbate_structure, self.rnd_generator
        )

        overlaped = check_overlap_vesin(
            atoms=atoms_trial,
            group1_indices=np.arange(len(self.current_system)),
            group2_indices=np.arange(len(self.current_system), len(atoms_trial)),
            vdw_radii=self.vdw,
        )
        if overlaped:
            return self.max_deltaE * 100, self.current_total_energy

        atoms_trial.calc = self.calculator
        e_new = atoms_trial.get_potential_energy()

        # deltaE = E_new - E_old = E(AB) - (E(A) + E(B))
        deltaE = e_new - (self.current_total_energy + adsorbate_structure.get_potential_energy())

        if deltaE < -self.max_deltaE:
            self.logger._print_warning(
                f"WARNING: Energy difference {deltaE:.4f} eV exceeds the maximum allowed {self.max_deltaE:.4f} eV."
            )

        return deltaE, e_new

    def try_virtual_deletion(
        self, adsorbate_tag: int, ads_index: list | None = None
    ) -> tuple[float, float]:
        """
        Try to delete an adsorbate molecule from the framework.
        This method randomly selects an adsorbate molecule and attempts deletion.

        Parameters
        ----------
        adsorbate_tag : int
            The tag of the adsorbate molecule being deleted.

        Returns
        -------
        deltaE
            Deletion energy.
        e_new
            New total energy.
        """
        ads_name = next((ads.name for ads in self.adsorbates if ads.tag == adsorbate_tag), None)

        if ads_name is None:
            raise ValueError(f"Adsorbate with tag {adsorbate_tag} not found.")

        if self.n_adsorbates[ads_name] == 0:
            return self.max_deltaE * 100, self.current_total_energy

        # Randomly select an adsorbate molecule of this specific tag to delete if ads_index is not provided
        if not ads_index:
            ads_indices_list = self.get_adsorbates_index(tag=adsorbate_tag)
            ads_indices = self.rnd_generator.choice(ads_indices_list, axis=0)

        else:
            ads_indices = ads_index

        atoms_trial = self.current_system.copy()
        atoms_trial.calc = self.calculator  # type: ignore

        to_delete = atoms_trial[ads_indices[0] : ads_indices[-1] + 1]  # type: ignore
        to_delete.calc = self.calculator  # type: ignore
        to_delete_energy = to_delete.get_potential_energy()  # type: ignore

        # Delete the selected adsorbate atoms from the trial structure
        del atoms_trial[ads_indices[0] : ads_indices[-1] + 1]  # type: ignore

        # deltaE = E_new - E_old = (E(A) + E(B)) - E(AB)
        e_new = atoms_trial.get_potential_energy()  # type: ignore
        deltaE = (e_new + to_delete_energy) - self.current_total_energy

        return deltaE, e_new

    def check_tmmc_files_exist(
        self, file_path: str, max_loading: int, replicas: list | range = range(1, 5)
    ) -> None:
        """
        Check if the required TMMC simulation data files exist for the specified replicas and loading.

        Parameters
        ----------
        file_path : str
            Path to the directory containing the TMMC simulation data.
        max_loading : int
            Maximum number of adsorbate molecules to consider for loading.
        replicas : list | range, optional
            List or range of replica indices to check. Default is range(1, 5).

        Raises
        ------
        FileNotFoundError
            If the specified directory or any required files do not exist.
        """
        for replica in replicas:
            prefix = f"{file_path}_{replica}"
            if not os.path.exists(prefix):
                raise FileNotFoundError(
                    f"Directory {prefix} does not exist. Please check the path and replica index."
                )
            not_found_files = []

            for n_mol in range(max_loading + 1):
                ins_file = f"{prefix}/ins_energy_{self.adsorbates[0].name}_{n_mol:04d}.npy"
                del_file = f"{prefix}/del_energy_{self.adsorbates[0].name}_{n_mol:04d}.npy"
                vol_file = f"{prefix}/volume_{n_mol:04d}.npy"

                for file in [ins_file, del_file, vol_file]:
                    if not os.path.exists(file):
                        not_found_files.append(file)
            if len(not_found_files) > 0:
                raise FileNotFoundError(
                    f"The following required files are missing for replica {replica}:\n"
                    + "\n".join(not_found_files)
                )

    def load_tmmc_weights(
        self,
        file_path: str,
        max_loading: int,
        replicas: list | range = range(1, 5),
        show_progress: bool = True,
        folder_name: str = "tmmc_weights",
    ) -> tuple[dict[int, list[np.ndarray]], dict[int, list[np.ndarray]]]:
        """
        Loads TMMC simulation data and pre-computes the fugacity-independent Boltzmann weights
        for a given temperature, returning them separated by replica.

        Parameters
        ----------
        file_path : str
            Path to the directory containing the TMMC simulation data.
        max_loading : int
            Maximum number of adsorbate molecules to consider for loading.
        temperature_K : float
            Temperature in Kelvin for which to compute the Boltzmann weights.
        replicas : list | range, optional
            List or range of replica indices to load. Default is range(1, 5).
        show_progress : bool, optional
            If True, shows a progress bar during the loading of weights. Default is True.
        folder_name : str, optional
            Name of the folder to save the pre-computed weights. Default is "tmmc_weights".

        Returns
        -------
        ins_weights: dict[int, list[np.ndarray]]
            Dictionary containing the pre-computed insertion weights for each replica.
        del_weights: dict[int, list[np.ndarray]]
            Dictionary containing the pre-computed deletion weights for each replica.

        Raises
        ------
        FileNotFoundError
            If the specified directory or any required files do not exist.
        """

        # Checks if the weights directory exists
        # self.check_tmmc_files_exist(file_path, max_loading, replicas)

        ins_weights: dict[int, list[np.ndarray]] = {replica: [] for replica in replicas}
        del_weights: dict[int, list[np.ndarray]] = {replica: [] for replica in replicas}

        os.makedirs(folder_name, exist_ok=True)

        for replica in replicas:
            prefix = f"{folder_name}/{replica}"

            # Read the saved wheighs if exists
            ins_file = f"{prefix}_{self.T}_ins_weights.npz"
            del_file = f"{prefix}_{self.T}_del_weights.npz"

            if os.path.exists(ins_file) and os.path.exists(del_file):
                print(f"Loading precomputed weights for replica {replica} from disk...")
                ins_data = np.load(ins_file)
                del_data = np.load(del_file)
                ins_weights[replica] = [ins_data[f"arr_{i}"] for i in range(len(ins_data.files))]
                del_weights[replica] = [del_data[f"arr_{i}"] for i in range(len(del_data.files))]

            else:
                prefix = f"{file_path}_{replica}"
                for n_mol in tqdm(
                    range(max_loading + 1),
                    desc=f"Loading weights (Replica {replica})",
                    disable=not show_progress,
                ):
                    # Load and mask insertion energies
                    rep_ins_energies = np.load(
                        f"{prefix}/ins_energy_{self.adsorbates[0].name}_{n_mol:04d}.npy"
                    )

                    # Load and mask deletion energies
                    rep_del_energies = np.load(
                        f"{prefix}/del_energy_{self.adsorbates[0].name}_{n_mol:04d}.npy"
                    )

                    # Load volumes
                    volumes = np.load(f"{prefix}/volume_{n_mol:04d}.npy")

                    # Precompute fugacity-independent part of insertion acceptance
                    w_ins = (volumes * self.beta / (n_mol + 1)) * np.exp(
                        -self.beta * rep_ins_energies
                    )
                    ins_weights[replica].append(w_ins)

                    # Precompute fugacity-independent part of deletion acceptance
                    w_del = (n_mol / (volumes * self.beta)) * np.exp(-self.beta * rep_del_energies)
                    del_weights[replica].append(w_del)

                # Save weights to disk for future use as compressed numpy arrays
                np.savez_compressed(
                    f"{folder_name}/{replica}_{self.T}_ins_weights.npz", *ins_weights[replica]
                )
                np.savez_compressed(
                    f"{folder_name}/{replica}_{self.T}_del_weights.npz", *del_weights[replica]
                )

        self.insertion_weights = ins_weights
        self.deletion_weights = del_weights

        return ins_weights, del_weights

    def compute_mpd(
        self,
        ins_weights: list[np.ndarray],
        del_weights: list[np.ndarray],
        fugacities_Pa: np.ndarray,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Computes the Macrostate Probability Distribution (MPD or Π) for a given set of fugacities.

        At equilibrium, the system obeys the principle of detailed balance, meaning the
        probability flux between adjacent macrostates is equal:

        Π(N) * P(N → N + 1) = Π(N + 1) * P(N + 1 → N)

        By rearranging this equality, the true physical MPD (Π) can be constructed
        iteratively from the transition probabilities (P):

        ln Π(N + 1) = ln Π(N) + ln( P(N → N + 1) / P(N + 1 → N) )

        This function calculates those transition probabilities using the unbiased
        acceptance weights scaled by fugacity, and applies a Log-Sum-Exp (LSE) trick
        during normalization to guarantee numerical stability.

        Parameters
        ----------
        ins_weights : list[np.ndarray]
            List of pre-computed insertion weights for each loading.
        del_weights : list[np.ndarray]
            List of pre-computed deletion weights for each loading.
        fugacities_Pa : np.ndarray
            Array of fugacities in Pascal.
        show_progress : bool, optional
            If True, shows a progress bar. Default is True.

        Returns
        -------
        pi : np.ndarray
            2D array of the MPD for each loading and fugacity. Shape: (max_loading, n_press).
        """
        # Convert fugacities from Pascal to ASE units (eV/Å³) for consistency with the Boltzmann weights
        fugacity_array = np.asarray(fugacities_Pa, dtype=float) * units.Pascal
        n_press = len(fugacity_array)
        n_states = len(ins_weights)

        # Initialize arrays for transition probabilities P(N -> N+1) and P(N -> N-1)
        probs_plus = np.zeros((n_states, n_press))
        probs_minus = np.zeros((n_states, n_press))

        # Calculate transition probabilities P(N -> N+1) and P(N -> N-1)
        for n_mol in tqdm(range(n_states), desc="Computing MPD (Π)", disable=not show_progress):
            # Broadcast: (n_samples, 1) * (1, n_press) -> (n_samples, n_press)
            # P_ins(N -> N+1) = min(1, w_ins(N) * fugacity)
            ins_acc = np.minimum(1.0, ins_weights[n_mol][:, None] * fugacity_array[None, :])
            probs_plus[n_mol] = np.mean(ins_acc, axis=0)

            # Broadcast: (n_samples, 1) / (1, n_press) -> (n_samples, n_press)
            # P_del(N -> N-1) = min(1, w_del(N) / fugacity)
            del_acc = np.minimum(1.0, del_weights[n_mol][:, None] / fugacity_array[None, :])
            probs_minus[n_mol] = np.mean(del_acc, axis=0)

        # Numerical safeguard to prevent log(0) which causes NaNs in the cumsum
        epsilon = 1e-300
        safe_probs_plus = np.clip(probs_plus[:-1], a_min=epsilon, a_max=None)
        safe_probs_minus = np.clip(probs_minus[1:], a_min=epsilon, a_max=None)

        # Calculate ln(Pi) using detailed balance and vectorized cumulative sum
        lnpi = np.zeros((n_states, n_press))
        log_ratios = np.log(safe_probs_plus / safe_probs_minus)
        lnpi[1:] = np.cumsum(log_ratios, axis=0)

        # Numerically stable normalization using Log-Sum-Exp prevents overflow/underflow
        lnpi -= np.max(lnpi, axis=0, keepdims=True)
        pi = np.exp(lnpi)
        pi /= np.sum(pi, axis=0, keepdims=True)

        return pi

    def compute_uptake_from_mpd(self, pi: np.ndarray) -> np.ndarray:
        """
        Calculates the expected number of molecules (uptake) from the Macrostate
        Probability Distribution (MPD).

        The MPD (Π) acts as the centerpiece of the ensemble thus the ensemble
        average of any physical property A can be derived directly from it via
        histogram reweighting:

        ⟨A⟩ = [ Σ_N A(N) * Π(N) ] / [ Σ_N Π(N) ]

        For the adsorption isotherm (uptake), the physical property A(N) is simply
        the number of molecules N. Since Π is already normalized (Σ_N Π(N) = 1),
        the formula simplifies to the expected value:

        ⟨N⟩ = Σ_N (N * Π(N))

        Parameters
        ----------
        pi : np.ndarray
            2D array of the MPD. Expected shape: (max_loading, n_press).

        Returns
        -------
        uptakes : np.ndarray
            1D array of expected number of molecules (uptake) for each fugacity.
            Shape: (n_press,).
        """
        n_states = pi.shape[0]
        n_grid = np.arange(n_states)[:, None]

        # ⟨N⟩ = Σ (N * Π)
        uptakes = np.sum(n_grid * pi, axis=0)

        return uptakes

    def compute_isotherm_replica(
        self,
        ins_weights: list[np.ndarray],
        del_weights: list[np.ndarray],
        fugacities_Pa: np.ndarray,
        show_progress: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes uptakes and macrostate probability distributions (pi) for a given
        set of fugacities using pre-computed insertion and deletion weights.

        This is a convenience wrapper combining `compute_mpd` and `compute_uptake_from_mpd`.
        """
        # 1. Compute the Macrostate Probability Distribution
        pi = self.compute_mpd(
            ins_weights=ins_weights,
            del_weights=del_weights,
            fugacities_Pa=fugacities_Pa,
            show_progress=show_progress,
        )

        # 2. Derive the macroscopic uptake from the MPD
        uptakes = self.compute_uptake_from_mpd(pi)

        # Transpose pi so shape is (n_press, max_loading + 1) to match original API expectation
        return uptakes, pi.T

    def compute_isotherm(
        self, fugacities_Pa: np.ndarray, uptake_units: str = "nmol", show_progress: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes uptakes and macrostate probability distributions (pi) for a list of fugacities.

        Parameters
        ----------
        fugacities_Pa : np.ndarray
            Array of fugacities in Pascal for which to compute the isotherm.
        uptake_units : str, optional
            The units in which to report the uptake. Default is "nmol".
        show_progress : bool, optional
            If True, shows a progress bar during the computation. Default is True.

        Returns
        -------
        uptakes_average : np.ndarray
            Array of expected uptake (in the specified units) averaged across replicas for each fugacity.
        uptakes_sd : np.ndarray
            Array of standard error of the mean (SEM) for the uptake across replicas for each fugacity.
        pi_average : np.ndarray
            2D array of macrostate probability distributions averaged across replicas for each fugacity and loading.
        """

        if not self.insertion_weights or not self.deletion_weights:
            raise ValueError(
                "Insertion and deletion weights must be loaded before computing the isotherm. Please run `load_tmmc_weights` first."
            )

        if uptake_units not in self.conv_factors:
            raise ValueError(
                f"Invalid uptake_units '{uptake_units}'. Valid options are: {list(self.conv_factors.keys())}"
            )

        uptakes_list = []
        pi_list = []

        for replica in range(len(self.insertion_weights)):
            uptakes, pi = self.compute_isotherm_replica(
                ins_weights=self.insertion_weights[replica + 1],
                del_weights=self.deletion_weights[replica + 1],
                fugacities_Pa=fugacities_Pa,
                show_progress=show_progress,
            )

            uptakes_list.append(uptakes)
            pi_list.append(pi)

        # Calculates the average uptake across replicas
        uptakes_average = np.mean(uptakes_list, axis=0)

        # Calculates the standard error of the mean (SEM) for the uptake across replicas
        uptakes_sd = np.std(uptakes_list, axis=0, ddof=1) / np.sqrt(len(self.insertion_weights))

        # Convert uptake to the specified units
        uptakes_average *= self.conv_factors[uptake_units][self.adsorbates[0].name]
        uptakes_sd *= self.conv_factors[uptake_units][self.adsorbates[0].name]

        pi_average = np.mean(pi_list, axis=0)

        return uptakes_average, uptakes_sd, pi_average

    def load_tmmc_weights_interpolated(
        self,
        file_path: str,
        loadings: list[int] | np.ndarray,
        replicas: list | range = range(1, 5),
        show_progress: bool = True,
        folder_name: str = "tmmc_weights",
    ) -> tuple[dict[int, dict[int, np.ndarray]], dict[int, dict[int, np.ndarray]]]:
        """
        Loads TMMC simulation data ONLY for a specified subset of macrostates (loadings)
        and pre-computes the fugacity-independent Boltzmann weights for each replica.

        In the interpolated single-macrostate (NVT + ghost swap) GC-TMMC approach
        (Mazur et al., ACS Appl. Mater. Interfaces 2024, 16, 25559-25567), direct
        simulations are performed only for a sparse subset of particle numbers N
        spanning [0, N_max], reducing computational cost by up to 1-2 orders of
        magnitude. For each explicitly simulated macrostate N, the fugacity-independent
        insertion and deletion weights are pre-computed as:

            w_ins(N) = (V * beta / (N + 1)) * exp(-beta * deltaE_ins)
            w_del(N) = (N / (V * beta))     * exp(-beta * deltaE_del)

        such that at any fugacity f, the unbiased Metropolis acceptance probabilities
        are evaluated as:

            p_acc(N -> N + 1) = min(1, w_ins(N) * f)
            p_acc(N -> N - 1) = min(1, w_del(N) / f)

        The resulting weights are stored as nested dictionaries mapping
        ``replica -> {n_mol: weight_array}`` so that the exact simulated macrostate
        indices are preserved for subsequent interpolation.

        Parameters
        ----------
        file_path : str
            Path prefix to the directories containing the TMMC simulation data
            (expects ``{file_path}_{replica}/``).
        loadings : list[int] | np.ndarray
            Subset of macrostate loadings (number of adsorbate molecules N) to load.
            Must include ``0`` and at least one non-zero loading (typically ``N_max``).
            Including ``1`` in the subset is recommended to avoid interpolation error
            in the deletion probability near the empty-pore limit.
        replicas : list | range, optional
            List or range of replica indices to load. Default is ``range(1, 5)``.
        show_progress : bool, optional
            If True, shows a progress bar during weight loading. Default is True.
        folder_name : str, optional
            Directory where compressed pre-computed weight caches (``.npz``) are
            stored and reused. Default is ``"tmmc_weights"``.

        Returns
        -------
        ins_weights : dict[int, dict[int, np.ndarray]]
            Pre-computed insertion weights mapped as ``ins_weights[replica][n_mol]``.
        del_weights : dict[int, dict[int, np.ndarray]]
            Pre-computed deletion weights mapped as ``del_weights[replica][n_mol]``.

        Raises
        ------
        ValueError
            If ``loadings`` contains fewer than two unique macrostates or does not
            include ``0``.
        """
        loadings_arr = np.array(sorted({int(n) for n in loadings}), dtype=int)
        if len(loadings_arr) < 2:
            raise ValueError("`loadings` must contain at least two distinct macrostates.")
        if loadings_arr[0] != 0:
            raise ValueError("`loadings` must include N = 0.")

        ins_weights: dict[int, dict[int, np.ndarray]] = {replica: {} for replica in replicas}
        del_weights: dict[int, dict[int, np.ndarray]] = {replica: {} for replica in replicas}

        os.makedirs(folder_name, exist_ok=True)
        ads_name = self.adsorbates[0].name

        for replica in replicas:
            cache_tag = f"{replica}_{self.T}_interp_{len(loadings_arr)}states"
            ins_file = os.path.join(folder_name, f"{cache_tag}_ins_weights.npz")
            del_file = os.path.join(folder_name, f"{cache_tag}_del_weights.npz")

            loaded_from_cache = False
            if os.path.exists(ins_file) and os.path.exists(del_file):
                ins_data = np.load(ins_file)
                del_data = np.load(del_file)
                if "loadings" in ins_data and np.array_equal(ins_data["loadings"], loadings_arr):
                    print(
                        f"Loading precomputed interpolated weights for replica {replica} from disk..."
                    )
                    ins_weights[replica] = {int(n): ins_data[f"n_{n:04d}"] for n in loadings_arr}
                    del_weights[replica] = {int(n): del_data[f"n_{n:04d}"] for n in loadings_arr}
                    loaded_from_cache = True

            if not loaded_from_cache:
                prefix = f"{file_path}_{replica}"
                for n_mol in tqdm(
                    loadings_arr,
                    desc=f"Loading subset weights (Replica {replica})",
                    disable=not show_progress,
                ):
                    rep_ins_energies = np.load(f"{prefix}/ins_energy_{ads_name}_{n_mol:04d}.npy")
                    rep_del_energies = np.load(f"{prefix}/del_energy_{ads_name}_{n_mol:04d}.npy")
                    volumes = np.load(f"{prefix}/volume_{n_mol:04d}.npy")

                    w_ins = (volumes * self.beta / (n_mol + 1)) * np.exp(
                        -self.beta * rep_ins_energies
                    )
                    w_del = (n_mol / (volumes * self.beta)) * np.exp(-self.beta * rep_del_energies)

                    ins_weights[replica][int(n_mol)] = w_ins
                    del_weights[replica][int(n_mol)] = w_del

                np.savez_compressed(
                    ins_file,
                    loadings=loadings_arr,
                    **{f"n_{n:04d}": ins_weights[replica][int(n)] for n in loadings_arr},
                    allow_pickle=True,
                )
                np.savez_compressed(
                    del_file,
                    loadings=loadings_arr,
                    **{f"n_{n:04d}": del_weights[replica][int(n)] for n in loadings_arr},
                    allow_pickle=True,
                )

        self.interpolated_insertion_weights = ins_weights
        self.interpolated_deletion_weights = del_weights
        self.sampled_loadings = loadings_arr

        return ins_weights, del_weights

    @staticmethod
    def _linear_interp_2d(
        x_new: np.ndarray,
        x_known: np.ndarray,
        y_known: np.ndarray,
    ) -> np.ndarray:
        """
        Vectorized piecewise linear interpolation along axis 0 of a 2D array.

        If ``x_new`` contains points outside ``[x_known[0], x_known[-1]]``,
        linearly extrapolates using the nearest boundary interval.

        Parameters
        ----------
        x_new : np.ndarray
            1D array of new x-coordinates where interpolation is desired.
        x_known : np.ndarray
            1D array of known x-coordinates corresponding to the rows of ``y_known``
        y_known : np.ndarray
            2D array of known y-values, where each row corresponds to a known x-coordinate

        Returns
        -------
        np.ndarray
            2D array of interpolated y-values at ``x_new``, with the same number of columns as ``y_known``.
        """
        if len(x_known) == 1:
            return np.repeat(y_known, len(x_new), axis=0)

        # Find the indices into a sorted array `x_known` such that, if the corresponding elements
        # in `x_new` were inserted before the indices, the order of `x_known` would be preserved.
        idx = np.searchsorted(x_known, x_new, side="right") - 1
        idx = np.clip(idx, 0, len(x_known) - 2)

        x_left = x_known[idx]
        x_right = x_known[idx + 1]
        alpha = ((x_new - x_left) / (x_right - x_left))[:, None]

        return (1.0 - alpha) * y_known[idx] + alpha * y_known[idx + 1]

    def compute_mpd_interpolated(
        self,
        ins_weights: dict[int, np.ndarray],
        del_weights: dict[int, np.ndarray],
        fugacities_Pa: np.ndarray,
        interpolation_space: str = "log",
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Computes the full Macrostate Probability Distribution (MPD or Π) across all
        integer loadings N ∈ [0, N_max] from a sparse subset of simulated macrostates
        by interpolating the transition probabilities.

        In single-macrostate (NVT + ghost swap) GC-TMMC, each independent canonical
        simulation at fixed loading N yields the unbiased transition probabilities for
        trial insertion and deletion:

        P(N -> N + 1; f) = ⟨ min(1, w_ins(N) * f) ⟩_NVT
        P(N -> N - 1; f) = ⟨ min(1, w_del(N) / f) ⟩_NVT

        Instead of simulating every integer N ∈ [0, N_max], transition probabilities
        are evaluated only at a few sampled macrostates and reconstructed across all
        unsampled integers N lying between adjacent sampled states N_a < N < N_b
        with α = (N - N_a) / (N_b - N_a):

        1. **Log-space interpolation** (``interpolation_space="log"``, default):
        Linearly interpolates the natural logarithms of the transition probabilities
        directly:

        ln P(N -> N + 1; f) = (1 - α) ln P(N_a -> N_a + 1; f) + α ln P(N_b -> N_b + 1; f)

        ln P(N -> N - 1; f) = (1 - α) ln P(N_a -> N_a - 1; f) + α ln P(N_b -> N_b - 1; f)

        Because detailed balance integrates the *logarithm* of the transition probability
        ratio—corresponding to the discrete derivative of the free energy landscape
        ∂(ln Π)/∂N—interpolating in log-space is exact when P(N) varies exponentially
        with N (e.g., near pore saturation or across free-energy barriers), prevents 
        overestimation across steep probability drops, and strictly guarantees P > 0 everywhere.

        2. **Linear-space interpolation** (``interpolation_space="linear"``):
        Linearly interpolates P(N -> N ± 1; f) directly before taking the logarithm,
        following Mazur et al. (ACS Appl. Mater. Interfaces 2024, 16, 25559-25567).

        Once the complete transition probability profiles are obtained for all
        N ∈ [0, N_max], equilibrium detailed balance between adjacent macrostates:

        Π(N) * P(N -> N + 1) = Π(N + 1) * P(N + 1 -> N)

        is integrated sequentially starting from an arbitrary reference ln Π(0) = 0:

        ln Π(N + 1) = ln Π(N) + ln P(N -> N + 1) - ln P(N + 1 -> N)

        Finally, Π(N) is normalized at each fugacity using the Log-Sum-Exp (LSE)
        trick to prevent numerical overflow or underflow.

        Notes
        -----
        At N = 0, deletion is impossible so P(0 -> -1) = 0 and ln P(0 -> -1) = -∞.
        In log-space mode, N = 0 is therefore excluded from the deletion anchors and
        ln P(N -> N - 1) is interpolated using only sampled macrostates with N ≥ 1.
        **Including N = 1 in the sampled subset is strongly recommended** so that
        all deletion probabilities on N ∈ [1, N_max] are purely interpolated. If
        N = 1 is absent, ln P(N -> N - 1) is linearly extrapolated backward from
        the first two non-zero sampled macrostates and clipped to [ln(1e-300), 0].

        Parameters
        ----------
        ins_weights : dict[int, np.ndarray]
            Dictionary mapping simulated macrostate ``n_mol`` to its pre-computed
            insertion weight array.
        del_weights : dict[int, np.ndarray]
            Dictionary mapping simulated macrostate ``n_mol`` to its pre-computed
            deletion weight array.
        fugacities_Pa : np.ndarray
            1D array of fugacities in Pascal.
        interpolation_space : str, optional
            Space in which piecewise linear interpolation is performed:
            ``"log"`` (interpolates ln P) or ``"linear"`` (interpolates P).
            Default is ``"log"``.
        show_progress : bool, optional
            If True, shows a progress bar over the simulated macrostates.
            Default is True.

        Returns
        -------
        pi : np.ndarray
            2D array of the normalized MPD for every integer loading N ∈ [0, N_max]
            and each fugacity. Shape: ``(N_max + 1, n_press)``.
        """
        if interpolation_space not in ("log", "linear"):
            raise ValueError(
                f"Invalid interpolation_space '{interpolation_space}'. "
                "Expected 'log' or 'linear'."
            )

        simulated_N = np.array(sorted(ins_weights.keys()), dtype=int)
        if len(simulated_N) < 2 or simulated_N[0] != 0:
            raise ValueError(
                "Simulated macrostates must include N = 0 and at least one non-zero loading."
            )
        if set(simulated_N) != set(del_weights.keys()):
            raise ValueError(
                "`ins_weights` and `del_weights` must contain the same macrostate keys."
            )

        # Convert fugacities from Pascal to ASE units (eV/Å³)
        fugacity_array = np.asarray(fugacities_Pa, dtype=float) * units.Pascal
        n_press = len(fugacity_array)
        n_sampled = len(simulated_N)
        max_loading = int(simulated_N[-1])
        full_N = np.arange(0, max_loading + 1, dtype=int)

        # Compute P(N -> N+1) and P(N -> N-1) ONLY at the simulated macrostates
        p_plus_sparse = np.zeros((n_sampled, n_press), dtype=float)
        p_minus_sparse = np.zeros((n_sampled, n_press), dtype=float)

        for idx, n_mol in enumerate(
            tqdm(simulated_N, desc="Computing sparse transition probs", disable=not show_progress)
        ):
            ins_acc = np.minimum(1.0, ins_weights[int(n_mol)][:, None] * fugacity_array[None, :])
            p_plus_sparse[idx] = np.mean(ins_acc, axis=0)

            if n_mol > 0:
                del_acc = np.minimum(
                    1.0, del_weights[int(n_mol)][:, None] / fugacity_array[None, :]
                )
                p_minus_sparse[idx] = np.mean(del_acc, axis=0)

        epsilon = 1e-300
        log_eps = np.log(epsilon)

        # Interpolate onto the full integer grid N ∈ [0, max_loading]
        if interpolation_space == "log":
            # Insertion: interpolate ln P(N -> N+1) on N ∈ [0, max_loading - 1]
            log_p_plus_sparse = np.log(np.clip(p_plus_sparse, a_min=epsilon, a_max=1.0))
            log_probs_plus = self._linear_interp_2d(full_N[:-1], simulated_N, log_p_plus_sparse)

            # Deletion: exclude N = 0 (where P_del = 0) and interpolate ln P(N -> N-1) on N ∈ [1, max_loading]
            simulated_N_del = simulated_N[1:]
            if simulated_N_del[0] > 1:
                self.logger._print_warning(
                    f"WARNING: N = 1 is not in the sampled macrostates (lowest non-zero N is "
                    f"{simulated_N_del[0]}). Linearly extrapolating ln P_del backward to N = 1; "
                    "include N = 1 in `loadings` for higher low-pressure accuracy."
                )

            log_p_minus_sparse = np.log(np.clip(p_minus_sparse[1:], a_min=epsilon, a_max=1.0))
            log_probs_minus = self._linear_interp_2d(
                full_N[1:], simulated_N_del, log_p_minus_sparse
            )
            # Enforce physical probability bounds [epsilon, 1.0] in log-space after any backward extrapolation
            log_probs_minus = np.clip(log_probs_minus, a_min=log_eps, a_max=0.0)

            log_ratios = log_probs_plus - log_probs_minus

        else:
            probs_plus = self._linear_interp_2d(full_N[:-1], simulated_N, p_plus_sparse)
            probs_minus = self._linear_interp_2d(full_N[1:], simulated_N, p_minus_sparse)

            safe_probs_plus = np.clip(probs_plus, a_min=epsilon, a_max=None)
            safe_probs_minus = np.clip(probs_minus, a_min=epsilon, a_max=None)
            log_ratios = np.log(safe_probs_plus / safe_probs_minus)

        # Integrate ln Π(N) via detailed balance
        lnpi = np.zeros((max_loading + 1, n_press), dtype=float)
        lnpi[1:] = np.cumsum(log_ratios, axis=0)

        # Numerically stable normalization using Log-Sum-Exp
        lnpi -= np.max(lnpi, axis=0, keepdims=True)
        pi = np.exp(lnpi)
        pi /= np.sum(pi, axis=0, keepdims=True)

        return pi

    def compute_isotherm_replica_interpolated(
        self,
        ins_weights: dict[int, np.ndarray],
        del_weights: dict[int, np.ndarray],
        fugacities_Pa: np.ndarray,
        interpolation_space: str = "log",
        show_progress: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the adsorption isotherm (expected uptake) and interpolated Macrostate
        Probability Distribution (MPD) for a single replica from a sparse subset of
        pre-computed insertion and deletion weights.

        Convenience wrapper combining ``compute_mpd_interpolated`` and
        ``compute_uptake_from_mpd``.

        Parameters
        ----------
        ins_weights : dict[int, np.ndarray]
            Dictionary mapping simulated macrostate ``n_mol`` to its insertion weight array.
        del_weights : dict[int, np.ndarray]
            Dictionary mapping simulated macrostate ``n_mol`` to its deletion weight array.
        fugacities_Pa : np.ndarray
            1D array of fugacities in Pascal.
        interpolation_space : str, optional
            Space in which transition probabilities are interpolated: ``"log"``
            (default) or ``"linear"``.
        show_progress : bool, optional
            If True, shows a progress bar. Default is True.

        Returns
        -------
        uptakes : np.ndarray
            1D array of expected uptake (molecules per simulation cell) for each fugacity.
            Shape: ``(n_press,)``.
        pi_T : np.ndarray
            2D array of the interpolated MPD, transposed to shape ``(n_press, N_max + 1)``.
        """
        pi = self.compute_mpd_interpolated(
            ins_weights=ins_weights,
            del_weights=del_weights,
            fugacities_Pa=fugacities_Pa,
            interpolation_space=interpolation_space,
            show_progress=show_progress,
        )
        uptakes = self.compute_uptake_from_mpd(pi)

        return uptakes, pi.T

    def compute_isotherm_interpolated(
        self,
        fugacities_Pa: np.ndarray,
        loadings: list[int] | np.ndarray | None = None,
        interpolation_space: str = "log",
        uptake_units: str = "nmol",
        show_progress: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes replica-averaged adsorption uptakes and interpolated Macrostate
        Probability Distributions (MPD) across a list of fugacities from a reduced
        set of macrostates.

        Supports two workflows:
        1. **Direct sparse workflow**: Call ``load_tmmc_weights_interpolated`` first,
           then call this method with ``loadings=None`` to use the loaded subset weights.
        2. **Subsampling/benchmarking workflow**: If full-macrostate weights were
           already loaded via ``load_tmmc_weights``, pass a ``loadings`` subset here
           to slice those weights in memory and benchmark interpolation accuracy
           without reloading from disk.

        Parameters
        ----------
        fugacities_Pa : np.ndarray
            Array of fugacities in Pascal for which to compute the isotherm.
        loadings : list[int] | np.ndarray | None, optional
            Optional subset of macrostates to slice from full weights loaded via
            ``load_tmmc_weights`` (or to further subsample already-loaded interpolated
            weights). If ``None``, uses all macrostates loaded by
            ``load_tmmc_weights_interpolated``. Default is ``None``.
        interpolation_space : str, optional
            Space in which transition probabilities are interpolated: ``"log"``
            (interpolates ln P, default) or ``"linear"`` (interpolates P directly).
        uptake_units : str, optional
            Units in which to report the uptake (must be a key in ``self.conv_factors``).
            Default is ``"nmol"``.
        show_progress : bool, optional
            If True, shows a progress bar during computation. Default is True.

        Returns
        -------
        uptakes_average : np.ndarray
            Expected uptake (in ``uptake_units``) averaged across replicas for each fugacity.
        uptakes_sd : np.ndarray
            Standard error of the mean (SEM) for the uptake across replicas.
        pi_average : np.ndarray
            2D array of the interpolated MPD averaged across replicas.
            Shape: ``(n_press, N_max + 1)``.
        """
        if uptake_units not in self.conv_factors:
            raise ValueError(
                f"Invalid uptake_units '{uptake_units}'. Valid options are: {list(self.conv_factors.keys())}"
            )

        interp_ins = getattr(self, "interpolated_insertion_weights", None)
        interp_del = getattr(self, "interpolated_deletion_weights", None)

        if interp_ins and interp_del:
            if loadings is not None:
                selected = sorted(set(int(n) for n in loadings))
                ins_source = {r: {n: interp_ins[r][n] for n in selected} for r in interp_ins}
                del_source = {r: {n: interp_del[r][n] for n in selected} for r in interp_del}
            else:
                ins_source = interp_ins
                del_source = interp_del
        elif self.insertion_weights and self.deletion_weights and loadings is not None:
            selected = sorted(set(int(n) for n in loadings))
            ins_source = {
                r: {n: self.insertion_weights[r][n] for n in selected}
                for r in self.insertion_weights
            }
            del_source = {
                r: {n: self.deletion_weights[r][n] for n in selected}
                for r in self.deletion_weights
            }
        else:
            raise ValueError(
                "No interpolated weights available. Either run `load_tmmc_weights_interpolated` "
                "first, or pass `loadings` after running `load_tmmc_weights`."
            )

        uptakes_list = []
        pi_list = []

        for replica in sorted(ins_source.keys()):
            uptakes, pi = self.compute_isotherm_replica_interpolated(
                ins_weights=ins_source[replica],
                del_weights=del_source[replica],
                fugacities_Pa=fugacities_Pa,
                interpolation_space=interpolation_space,
                show_progress=show_progress,
            )
            uptakes_list.append(uptakes)
            pi_list.append(pi)

        n_replicas = len(uptakes_list)
        uptakes_average = np.mean(uptakes_list, axis=0)
        ddof = 1 if n_replicas > 1 else 0
        uptakes_sd = np.std(uptakes_list, axis=0, ddof=ddof) / np.sqrt(n_replicas)

        conv = self.conv_factors[uptake_units][self.adsorbates[0].name]
        uptakes_average *= conv
        uptakes_sd *= conv

        pi_average = np.mean(pi_list, axis=0)

        return uptakes_average, uptakes_sd, pi_average

    def run_tmmc(self, N_per_frame: int, trajectory_path: str) -> None:
        """
        Run the transition matrix Monte Carlo simulation for N iterations in
        a given trajectory.

        The trajectory is read from a file and the simulation is performed on
        each frame of the trajectory for N iterations.

        For each iteration, no real move are actually performed on the system,
        but virtual insertion and deletion are performed and the corresponding
        energies and the unit cell volume are recorded.

        Parameters
        ----------
        N_per_frame : int
            The number of iteration for each frame in the trajectory.
        trajectory_path : str
            The path to the trajectory file where the simulation will be executed.
        """

        self.logger.print_tmmc_run_header()

        trajectory = TrajectoryReader(trajectory_path)

        # Iterate over each frame in the trajectory and perform N_per_frame virtual insertions and deletions
        for frame_number, frame in enumerate(trajectory):
            self.set_state(frame)  # type: ignore
            frame_volume = self.current_system.get_volume()

            ads_indices_list = self.get_adsorbates_index(tag=self.adsorbates[0].tag)

            # Pre-compute the deletion energies for all adsorbates in the current frame
            del_energy_temp_list = []
            total_energy_temp_list = []

            for ads_idx in tqdm(
                ads_indices_list,
                disable=(self.out_file is None),
                desc=f"TMMC Deletions on frame  {frame_number:05d}",
            ):
                del_energy, e_new = self.try_virtual_deletion(
                    self.adsorbates[0].tag, ads_index=ads_idx
                )
                del_energy_temp_list.append(del_energy)
                total_energy_temp_list.append(e_new)

            # If there are no adsorbates in the current frame, set a default deletion energy to avoid empty lists
            if not del_energy_temp_list:
                del_energy_temp_list = [self.max_deltaE * 100]
                total_energy_temp_list = [self.current_total_energy]

            # Perform N virtual insertions of the adsorbate
            for iteration in tqdm(
                range(1, N_per_frame + 1),
                disable=(self.out_file is None),
                desc=f"TMMC Insertions on frame {frame_number:05d}",
            ):
                step_time_start = datetime.datetime.now()

                # Perform a virtual insertion of the adsorbate and record the insertion energy
                ins_energy, e_new_ins = self.try_virtual_insertion(self.adsorbates[0].tag)
                self.total_ins_energy_list[self.adsorbates[0].name].append(ins_energy)

                # Perform a virtual deletion of a randomly selected adsorbate and record the deletion energy
                del_energy_index = self.rnd_generator.choice(range(len(del_energy_temp_list)))
                del_energy = del_energy_temp_list[del_energy_index]
                e_new_del = total_energy_temp_list[del_energy_index]

                self.total_del_energy_list[self.adsorbates[0].name].append(del_energy)
                self.total_energy_list[self.adsorbates[0].name].append([e_new_ins, e_new_del])

                # Record the volume of the current system for this iteration
                self.volume_list.append(frame_volume)

                # Save the current state and trajectory
                self._save_state()

                # Print the current step information to the output file
                self.logger.print_tmmc_step_info(
                    step=(iteration + frame_number * N_per_frame),
                    del_energy={self.adsorbates[0].name: del_energy},
                    ins_energy={self.adsorbates[0].name: ins_energy},
                    step_time=(datetime.datetime.now() - step_time_start).total_seconds(),
                )
