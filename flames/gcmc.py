import datetime
import json
import os

import ase
import numpy as np
import pymser
from ase import units
from ase.calculators import calculator
from ase.io import Trajectory, read
from tqdm import tqdm

from flames import VERSION
from flames.adsorbate import Adsorbate
from flames.base_simulator import BaseSimulator
from flames.logger import GCMCLogger
from flames.operations import (
    check_overlap_vesin,
    random_mol_insertion,
    random_rotation_limited,
    random_translation,
    swap_positions,
)


class GCMC(BaseSimulator):
    """
    Base class for Grand Canonical Monte Carlo (GCMC) simulations using ASE.

    This class employs Monte Carlo simulations under the grand canonical ensemble (:math:`μVT`) to study the adsorption of molecules in a framework material.
    It allows for movements such as insertion, deletion, translation, and rotation of adsorbate molecules within the framework.

    Currently, it supports any ASE-compatible calculator for energy calculations.

    :param model:
        The calculator to use for energy calculations. Can be any ASE-compatible calculator.
        The output of the calculator should be in eV.
    :type model: ase.calculators.calculator.Calculator

    :param framework_atoms:
        The framework structure as an ASE Atoms object.
    :type framework_atoms: ase.Atoms

    :param adsorbates:
        The adsorbate structure as an Adsorbate object.
    :type adsorbates: Adsorbate

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

    :param max_deltaE:
        Maximum energy difference (in eV) to consider for acceptance criteria.
        This is used to avoid overflow due to problematic calculations. Default is ``1.555`` eV (approx. 150 kJ/mol).
    :type max_deltaE: float, optional

    :param vdw_factor:
        Factor to scale the Van der Waals radii. Default is ``0.6``.
    :type vdw_factor: float, optional

    :param framework_energy:
        Pre-calculated potential energy of the empty framework in eV. If not provided, it will be calculated during initialization.
    :type framework_energy: float or None, optional

    :param adsorbate_energy:
        Pre-calculated potential energy of the adsorbate molecule in eV. If not provided, it will be calculated during initialization.
    :type adsorbate_energy: float or None, optional

    :param max_translation:
        Maximum translation distance. Default is ``1.5``.
    :type max_translation: float, optional

    :param max_rotation:
        Maximum rotation angle (in radians). Default is ``90`` degrees (converted to radians).
    :type max_rotation: float, optional

    :param save_frequency:
        Frequency at which to save the simulation state and results. Default is ``100``.
    :type save_frequency: int, optional

    :param save_rejected:
        If ``True``, saves the rejected moves in a trajectory file. Default is ``False``.
    :type save_rejected: bool, optional

    :param output_to_file:
        If ``True``, writes the output to a file named ``output_{temperature}_{pressure}.out`` in the ``results`` directory. Default is ``True``.
    :type output_to_file: bool, optional

    :param output_folder:
        Folder to save the output files. If ``None``, a folder named ``results_<T>_<P>`` will be created.
    :type output_folder: str or None, optional

    :param debug:
        If ``True``, prints detailed debug information during the simulation. Default is ``False``.
    :type debug: bool, optional

    :param fugacity_coeff:
        Fugacity coefficient to correct the pressure. Default is ``1.0``.
        Only used if ``criticalTemperature``, ``criticalPressure``, and ``acentricFactor`` are not provided.
    :type fugacity_coeff: float, optional

    :param random_seed:
        Random seed for reproducibility. Default is ``None`` and will generate a random seed automatically if not provided.
    :type random_seed: int or None, optional

    :param cutoff_radius:
        Interaction potential cut-off radius used to estimate the minimum unit cell. Default is ``6.0``.
    :type cutoff_radius: float, optional

    :param automatic_supercell:
        If ``True``, automatically creates a supercell based on the cutoff radius. Default is ``True``.
    :type automatic_supercell: bool, optional

    :param max_length:
        Maximum length (in Angstroms) for any side of the supercell. If ``None``, no maximum length is enforced.
        This can be used to limit the size of the supercell for computational efficiency. Default is ``None``.
    :type max_length: float or None, optional

    :param criticalTemperature:
        Critical temperature of the adsorbate in Kelvin.
    :type criticalTemperature: float, optional

    :param criticalPressure:
        Critical pressure of the adsorbate in Pascal.
    :type criticalPressure: float, optional

    :param acentricFactor:
        Acentric factor of the adsorbate.
    :type acentricFactor: float, optional

    :param void_fraction:
        Void fraction of the adsorbate.
    :type void_fraction: float, optional

    """

    def __init__(
        self,
        model: calculator.Calculator,
        framework_atoms: ase.Atoms,
        adsorbates: Adsorbate | list[Adsorbate],
        temperature: float,
        pressure: float,
        device: str,
        vdw_radii: np.ndarray,
        vdw_factor: float = 0.6,
        framework_energy: float | None = None,
        adsorbate_energy: float | None = None,
        max_translation: float = 1.5,
        max_rotation: float = np.radians(90),
        max_deltaE: float = 1.555,
        save_frequency: int = 100,
        save_rejected: bool = False,
        output_to_file: bool = True,
        output_folder: str | None = None,
        debug: bool = False,
        fugacity_coeff: float = 1.0,
        random_seed: int | None = None,
        cutoff_radius: float = 6.0,
        automatic_supercell: bool = True,
        void_fraction: float | list[float] = 0.0,
    ) -> None:
        """
        Initialize the Grand Canonical Monte Carlo (GCMC) simulation.
        """

        super().__init__(
            model=model,
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
            save_rejected=save_rejected,
            output_to_file=output_to_file,
            output_folder=output_folder,
            debug=debug,
            fugacity_coeff=fugacity_coeff,
            random_seed=random_seed,
            cutoff_radius=cutoff_radius,
            automatic_supercell=automatic_supercell,
        )

        self.logger = GCMCLogger(simulation=self, output_file=self.out_file)

        self.start_time = datetime.datetime.now()

        # Parameters for calculateing the Peng-Robinson equation of state
        self.void_fraction = void_fraction
        self.excess_nmol = {adsorbate.name: 0.0 for adsorbate in self.adsorbates}

        # Parameters for storing the main results during the simulation
        self.n_adsorbates: dict[str, int] = {adsorbate.name: 0 for adsorbate in self.adsorbates}
        self.uptake_list: np.ndarray = np.zeros((1, len(self.adsorbates)), dtype=int)
        self.total_energy_list: list[float] = [0]
        self.total_ads_list: list[float] = [0]

        self.max_translation = max_translation
        self.max_rotation = max_rotation

        self.movements: dict = {
            "insertion": self.try_insertion,
            "deletion": self.try_deletion,
            "rotation": self.try_rotation,
            "translation": self.try_translation,
            "reinsertion": self.try_reinsertion,
            "identity_swap": self.try_identity_swap,
            "nve_md": self.try_nve_md,
            "nvt_md": self.try_nvt_md,
            "npt_md": self.try_npt_md
        }

        self.n_movements = {move: [] for move in self.movements.keys()}

        # Base iteration for restarting the simulation. This is for tracking the iteration count only
        self._base_iteration: int = 0

        # Dictionary to store the equilibrated results by pyMSER
        self.equilibrated_results: dict = {}

    @property
    def base_iteration(self) -> int:
        """
        Get the base iteration for the GCMC simulation.

        Returns
        -------
        int
            The base iteration count.
        """
        return self._base_iteration

    @base_iteration.setter
    def base_iteration(self, iteration: int) -> None:
        """
        Set the base iteration for the GCMC simulation.

        Parameters
        ----------
        iteration : int
            The base iteration count to set.
        """
        self._base_iteration = iteration

    def restart(self) -> None:
        """
        Restart the simulation from the last state.

        This method loads the last saved state from the trajectory file and restores the simulation to that state.
        It also loads the uptake, total energy, and total adsorbates lists from the saved files if they exist.
        """

        print("Restarting simulation...")
        uptake_restart, total_energy_restart, total_ads_restart = [], [], []

        if os.path.exists(os.path.join(self.out_folder, f"uptake_{self.P:.5f}.npy")):
            uptake_restart = np.load(os.path.join(self.out_folder, f"uptake_{self.P:.5f}.npy"))

        if os.path.exists(os.path.join(self.out_folder, f"total_energy_{self.P:.5f}.npy")):
            total_energy_restart = np.load(
                os.path.join(self.out_folder, f"total_energy_{self.P:.5f}.npy")
            ).tolist()

        if os.path.exists(os.path.join(self.out_folder, f"total_ads_{self.P:.5f}.npy")):
            total_ads_restart = np.load(
                os.path.join(self.out_folder, f"total_ads_{self.P:.5f}.npy")
            ).tolist()

        # Check if the len of all restart elements are the same:
        if not (len(uptake_restart) == len(total_energy_restart) == len(total_ads_restart)):
            raise ValueError(f"""
            The lengths of uptake, total energy, and total adsorbates lists do not match.
            Please check the saved files.
            Found lengths: {len(uptake_restart)}, {len(total_energy_restart)}, {len(total_ads_restart)}
            for uptake, total energy, and total ads respectively.""")

        self.uptake_list = np.array(uptake_restart)
        self.total_energy_list = total_energy_restart
        self.total_ads_list = total_ads_restart

        # Set the base iteration to the length of the uptake list
        self.base_iteration = len(self.uptake_list)

        self.logger.print_restart_info()

        if os.path.exists(os.path.join(self.out_folder, "Movies", "Trajectory.traj")):
            try:
                self.load_state(os.path.join(self.out_folder, "Movies", "Trajectory.traj"))
            except Exception as e:
                self.logger._print(
                    "=" * 76
                    + "\n"
                    + "WARNING: Error occurred while loading trajectory file:\n"
                    + str(e)
                    + "\n"
                    + "Cannot load the last state of the simulation.\n"
                    + "This is likely due to empty or corrupted trajectory file.\n"
                    + "Simulation will start from scratch.\n"
                    + "=" * 76
                    + "\n"
                )
        else:
            raise FileNotFoundError(
                f"ERROR: Trajectory file '{os.path.join(self.out_folder, 'Movies', 'Trajectory.traj')}' does not exist. "
                + "Cannot load the last state of the simulation."
            )

    def load_state(self, state_file: str) -> None:
        """
        Load the state of the simulation from a file.

        Parameters
        ----------
        state_file : str
            Path to the file containing the saved state of the simulation.
        """
        print(f"Loading state from {state_file}...")

        if not os.path.exists(state_file):
            raise FileNotFoundError(f"State file '{state_file}' does not exist.")

        if state_file.endswith(".traj"):
            state = Trajectory(state_file, "r")[-1]  # type: ignore
        else:
            state: ase.Atoms = read(state_file)  # type: ignore

        # Workaround to load the labels from Trajectory.info since ASE's Trajectory does not support custom arrays
        if "labels" in state.info.keys():
            state.set_array("labels", state.info["labels"])

        self.set_state(state)

        self.n_adsorbates = self.get_number_of_adsorbates(state)

        average_binding_energy = self.current_total_energy - self.framework_energy

        for ads_name in self.n_adsorbates.keys():
            average_binding_energy -= self.n_adsorbates[ads_name] * self.adsorbate_energy[ads_name]

        average_binding_energy /= units.kJ / units.mol

        self.logger.print_load_state_info(
            n_atoms=len(state), average_ads_energy=average_binding_energy
        )

    def get_number_of_adsorbates(self, system: ase.Atoms | None = None) -> dict[str, int]:
        """
        Get the number of adsorbates in the current system.
        It considers the possibility of having multiple adsorbate species in the simulation.

        Parameters
        ----------
        system : ase.Atoms | None
            The ASE Atoms object representing the current system.

        Returns
        -------
        n_adsorbate: dict[str, int]
            A dictionary mapping adsorbate names to their respective counts.
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

    def insert_adsorbates(self, n_adsorbates: int, ads_name: str, max_attempts: int = 1000) -> None:
        """
        Insert a given number of adsorbate molecules into the framework at random positions without overlap.

        Parameters
        ----------
        n_molecules : int
            Number of adsorbate molecules to insert.
        ads_name : str
            Name of the adsorbate type to insert.
        max_attempts : int
            Maximum number of attempts to insert the molecules without overlap. Default is 1000.
        """
        temp_system = self.current_system.copy()

        adsorvate = next((ads for ads in self.adsorbates if ads.name == ads_name), None)

        if adsorvate is None:
            raise ValueError(f"Adsorbate with name '{ads_name}' not found in the adsorbates list.")

        n_attempts = 0
        inserted_adsorbates = 0
        while inserted_adsorbates < n_adsorbates and n_attempts < max_attempts:
            n_attempts += 1
            atoms_trial = random_mol_insertion(temp_system, adsorvate.structure, self.rnd_generator)

            overlaped = check_overlap_vesin(
                atoms=atoms_trial,
                group1_indices=np.arange(len(temp_system)),
                group2_indices=np.arange(len(temp_system), len(atoms_trial)),
                vdw_radii=self.vdw,
            )

            if not overlaped:
                temp_system = atoms_trial.copy()
                inserted_adsorbates += 1

        if n_attempts == max_attempts:
            raise Warning(
                f"Maximum number of attempts ({max_attempts}) reached. "
                + f"Could not insert all {n_adsorbates} adsorbates without overlap. "
                + f"Max inserted adsorbates: {inserted_adsorbates}."
            )

        # Update the current system and total energy
        temp_system.calc = self.model
        self.current_system = temp_system.copy()
        self.current_total_energy = temp_system.get_potential_energy()
        self.n_adsorbates[ads_name] += inserted_adsorbates

    def get_adsorbates_index(self, tag: int | None = None) -> list[list]:
        """
        Get a list of index of the adsorbate molecules in the current system.
        If a tag is provided, only returns the indices for adsorbates matching that tag.

        Returns
        -------
        list[list]
            A list of lists containing the indices of the adsorbate molecules.
        """
        adsorbates_list = []

        for adsorbate in self.adsorbates:
            # If a tag is provided and doesn't match this adsorbate, skip to the next one
            if tag is not None and adsorbate.tag != tag:
                continue

            # Look for the current adsorbate's tag in the system
            indices = np.where(self.current_system.get_tags() == adsorbate.tag)[0]

            if len(indices) > 0:
                adsorbates_list.extend(indices.reshape(-1, len(adsorbate.structure)).tolist())

        return adsorbates_list

    def equilibrate(
        self,
        equilibration_steps: int = 0,
        LLM: bool = False,
        batch_size: int | bool = False,
        run_ADF: bool = False,
        uncertainty: str = "uSD",
    ) -> None:
        """
        Use pyMSER to get the equilibrated statistics of the simulation.

        Parameters
        ----------
        equilibration_steps : int
            Number of steps to use for equilibration. Default is 0.
        LLM : bool
            If True, use the Leftmost Local Minima (LLM) on the determination of the equilibration point
            by `pyMSER <https://github.com/lipelopesoliveira/pyMSER>`_.
            this can underestimate the equilibration point in some situations,
            but generate good averages for well-behaved scenarios.
            Default is False.
        batch_size : int
            Batch size to use for speedup the equilibration process. Default is 100.
        run_ADF : bool
            If True, run the Augmented Dickey-Fuller (ADF) test to confirm for stationarity.
            Default is False.
        uncertainty : str
            The type of uncertainty to use for the equilibration process. Default is "uSD".
            Options are:
            - "uSD": uncorrelated Standard Deviation
            - "uSE": uncorrelated Standard Error
            - "SD": Standard Deviation
            - "SE": Standard Error
        """

        assert uncertainty in ["uSD", "uSE", "SD", "SE"], (
            f"Invalid uncertainty type: {uncertainty}. "
            + "Valid options are: 'uSD', 'uSE', 'SD', 'SE'."
        )

        assert isinstance(equilibration_steps, int) and equilibration_steps >= 0, (
            f"Invalid type for equilibration_steps: {type(equilibration_steps)}. "
            + "Expected a non-negative integer."
        )

        # Equilibration uses the total uptake for determining the equilibration point.
        # The total uptake is only equilibrated if all components are equilibrated.
        total_uptake = np.array(self.uptake_list).sum(axis=-1)

        eq_results = pymser.equilibrate(
            total_uptake[equilibration_steps:],
            LLM=LLM,
            batch_size=(
                int(len(total_uptake[equilibration_steps:]) / 50)
                if batch_size is False
                else batch_size
            ),
            ADF_test=run_ADF,
            uncertainty=uncertainty,
            print_results=False,
        )

        for i, ads in enumerate(self.adsorbates):
            uptake = np.array(self.uptake_list)[:, i]

            average, avg_uncertainty = pymser.calc_equilibrated_average(
                uptake, eq_results["t0"], uncertainty, int(eq_results["ac_time"])
            )

            eq_results[f"average_{ads.name}"] = average
            eq_results[f"uncertainty_{ads.name}"] = avg_uncertainty  # type: ignore

            enthalpy, enthalpy_sd = pymser.calc_equilibrated_enthalpy(
                energy=np.array(self.total_ads_list[equilibration_steps:])
                / units.kB,  # Convert to K
                number_of_molecules=uptake[equilibration_steps:],
                temperature=self.T,
                eq_index=eq_results["t0"],
                uncertainty="SD",
                ac_time=int(eq_results["ac_time"]),
            )

            eq_results[f"enthalpy_{ads.name}_kJ_per_mol"] = float(enthalpy)
            eq_results[f"enthalpy_{ads.name}_sd_kJ_per_mol"] = float(enthalpy_sd)

        eq_results["LLM"] = LLM
        eq_results["average"] = float(eq_results["average"])
        eq_results["uncertainty"] = float(eq_results["uncertainty"])
        eq_results["ac_time"] = int(eq_results["ac_time"])
        eq_results["uncorr_samples"] = int(eq_results["uncorr_samples"])

        eq_results["equilibrated"] = eq_results["t0"] < 0.75 * len(
            total_uptake[equilibration_steps:]
        )

        self.equilibrated_results = eq_results

    def save_results(
        self,
        file_name: str | None = None,
        batch_size: int | bool = False,
        run_ADF: bool = False,
        uncertainty: str = "uSD",
        LLM: bool = False,
    ) -> None:
        """
        Save a json file with the main results of the simulation.

        Parameters
        ----------
        file_name : str
            Name of the output file. Default is 'GCMC_Results.json'.
        LLM : bool
            If True, use the Leftmost-Local Minima (LLM) method to determine the equilibration time.
            This is only recommended for high-throughput simulations, and sometimes can underestimate
            the true equilibration point.
            Default is True.
        batch_size : int
            Batch size to use for speedup the equilibration process.
            Default is False, which means 2% of the total number of steps.
        run_ADF : bool
            If True, run the Augmented Dickey-Fuller (ADF) test to confirm for stationarity.
            Default is False.
        uncertainty : str
            The type of uncertainty to use for the equilibration process. Default is "uSD".
            Options are:
            - "uSD": uncorrelated Standard Deviation
            - "uSE": uncorrelated Standard Error
            - "SD": Standard Deviation
            - "SE": Standard Error
        LLM : bool
            If True, use the Leftmost-Local Minima (LLM) method to determine the equilibration time.
            This is only recommended for high-throughput simulations, and sometimes can underestimate
            the true equilibration point.
            Default is False.

        """

        if file_name is None:
            file_name = f"results_{self.T}_{self.P}.json"

        self.equilibrate(batch_size=batch_size, run_ADF=run_ADF, uncertainty=uncertainty, LLM=LLM)

        results = {
            "simulation": {
                "code_version": VERSION,
                "random_seed": self.random_seed,
                "temperature_K": self.T,
                "pressure_Pa": self.P,
                "fugacity_coefficient": self.fugacity_coeff,
                "fugacity_Pa": self.fugacity_coeff * self.P,
                "move_weights": {ads.name: ads.weights.__dict__ for ads in self.adsorbates},
                "mol_fractions": self.mol_fractions,
                "n_steps": len(self.uptake_list),
                "enlapsed_time_hours": (datetime.datetime.now() - self.start_time).total_seconds()
                / 3600,
            },
            "equilibration": {
                "LLM": self.equilibrated_results.get("LLM", False),
                "t0": int(self.equilibrated_results.get("t0", 0)),
                "average": self.equilibrated_results.get("average", None),
                "uncertainty": self.equilibrated_results.get("uncertainty", None),
                "equilibrated": bool(self.equilibrated_results.get("equilibrated", False)),
                "ac_time": self.equilibrated_results.get("ac_time", None),
                "uncorr_samples": self.equilibrated_results.get("uncorr_samples", None),
            },
            "results": {},
        }

        for ads in self.adsorbates:
            results["results"][ads.name] = {}

            avrg = float(self.equilibrated_results.get(f"average_{ads.name}", 0))
            stdv = float(self.equilibrated_results.get(f"uncertainty_{ads.name}", 0))

            # --- Uptake data (computed from conversion factors) ---
            results["results"][ads.name]["absolute_uptake"] = {
                unit: {
                    "mean": (avrg) * factor[ads.name],
                    "sd": stdv * factor[ads.name],
                }
                for unit, factor in self.conv_factors.items()
            }

            results["results"][ads.name]["excess_uptake"] = {
                unit: {
                    "mean": (avrg - self.excess_nmol[ads.name]) * factor[ads.name],
                    "sd": stdv * factor[ads.name],
                }
                for unit, factor in self.conv_factors.items()
            }

            results["results"][ads.name]["enthalpy"] = {
                "kJ_mol": {
                    "mean": self.equilibrated_results.get(f"enthalpy_{ads.name}_kJ_per_mol", None),
                    "sd": self.equilibrated_results.get(f"enthalpy_{ads.name}_sd_kJ_per_mol", None),
                }
            }

        with open(os.path.join(self.out_folder, file_name), "w") as f:
            json.dump(results, f, indent=4)

    def _insertion_acceptance(self, deltaE, adsorbate_tag) -> bool:
        """
        Calculate the acceptance probability for insertion of an adsorbate molecule as

        P_acc (N -> N + 1) = min(1, β * V * f * exp(-β ΔE) / (N + 1))

        Parameters
        ----------
        deltaE : float
            Energy difference between the new and old configuration in eV.
        adsorbate_tag : int
            Tag identifying the adsorbate molecule.
        """

        exp_value = np.exp(-self.beta * deltaE)

        ads_name = next((ads.name for ads in self.adsorbates if ads.tag == adsorbate_tag), None)

        n_ads = self.n_adsorbates[ads_name] if ads_name else 0

        pre_factor = self.V * self.beta * self.fugacity / (n_ads + 1)

        acc = min(1, pre_factor * exp_value)

        rnd_number = self.rnd_generator.random()

        if self.debug:
            self.logger.print_debug_movement(
                movement="Insertion",
                deltaE=deltaE,
                prefactor=pre_factor,
                acc=acc,
                rnd_number=rnd_number,
                adsorbate_name=ads_name,
            )

        # Apply Metropolis acceptance/rejection rule
        return rnd_number < acc

    def _deletion_acceptance(self, deltaE, adsorbate_tag) -> bool:
        """
        Calculate the acceptance probability for deletion of an adsorbate molecule as

        P_del (N -> N - 1 ) = min(1, N / (β * V * f) * exp(-β ΔE) )

        Parameters
        ----------
        deltaE : float
            Energy difference between the new and old configuration in eV.
        adsorbate_tag : int
            Tag identifying the adsorbate molecule.
            Current number of adsorbate molecules in the system.
        """

        exp_value = np.exp(-self.beta * deltaE)

        ads_name = next((ads.name for ads in self.adsorbates if ads.tag == adsorbate_tag), None)
        n_ads = self.n_adsorbates[ads_name] if ads_name else 0

        pre_factor = n_ads / (self.V * self.beta * self.fugacity)

        acc = min(1, pre_factor * exp_value)

        rnd_number = self.rnd_generator.random()

        if self.debug:
            self.logger.print_debug_movement(
                movement="Deletion",
                deltaE=deltaE,
                prefactor=pre_factor,
                acc=acc,
                rnd_number=rnd_number,
                adsorbate_name=ads_name,
            )

        # Apply Metropolis acceptance/rejection rule
        return rnd_number < acc

    def _reinsertion_acceptance(self, deltaE, adsorbate_tag) -> bool:
        """
        Calculate the acceptance probability for reinsertion of an adsorbate molecule as

        P_reins (N -> N ) = min(1, exp(-β ΔE) )

        Parameters
        ----------
        deltaE : float
            Energy difference between the new and old configuration in eV.
        adsorbate_tag : int
            Tag identifying the adsorbate molecule.
        """

        exp_value = np.exp(-self.beta * deltaE)
        acc = min(1, exp_value)

        ads_name = next((ads.name for ads in self.adsorbates if ads.tag == adsorbate_tag), None)

        rnd_number = self.rnd_generator.random()

        if self.debug:
            self.logger.print_debug_movement(
                movement="Reinsertion",
                deltaE=deltaE,
                prefactor=1,
                acc=acc,
                rnd_number=rnd_number,
                adsorbate_name=ads_name,
            )

        # Apply Metropolis acceptance/rejection rule
        return rnd_number < acc

    def _swap_acceptance(self, deltaE: float, adsorbate_tags: list[int]) -> bool:
        """
        Calculate the acceptance probability for translation or rotation of an adsorbate molecule as

        P_move = min(1, N_a / (N_b + 1) . f_b / f_a . exp(-β ΔE))

        Parameters
        ----------
        deltaE : float
            Energy difference between the new and old configuration in eV.
        adsorbate_tags : list[int], optional
            Tags identifying the adsorbate molecules. Default is None.
        """

        ads_names = [
                    next((ads.name for ads in self.adsorbates if ads.tag == tag), None)
                    for tag in adsorbate_tags
                ]

        N_a = self.n_adsorbates[ads_names[0]] if ads_names[0] else 0
        N_b = self.n_adsorbates[ads_names[1]] if ads_names[1] else 0

        f_a = next((ads.eos.get_fugacity_coefficient(self.T, self.P) for ads in self.adsorbates if ads.name == ads_names[0]), 1.0)  # type: ignore
        f_b = next((ads.eos.get_fugacity_coefficient(self.T, self.P) for ads in self.adsorbates if ads.name == ads_names[1]), 1.0)  # type: ignore

        exp_value = np.exp(-self.beta * deltaE)
        pre_factor = (N_a / (N_b + 1)) * (f_b / f_a)
        acc = min(1, pre_factor * exp_value)

        rnd_number = self.rnd_generator.random()

        if self.debug:
            self.logger.print_debug_movement(
                movement="Identity Swap",
                deltaE=deltaE,
                prefactor=pre_factor,
                acc=acc,
                rnd_number=rnd_number,
                adsorbate_name=ads_names,
            )

        # Apply Metropolis acceptance/rejection rule
        return rnd_number < acc

    def _move_acceptance(self, deltaE: float, movement_name: str, adsorbate_tag: int) -> bool:
        """
        Calculate the acceptance probability for translation or rotation of an adsorbate molecule as

        P_move = min(1, exp(-β ΔE))

        Parameters
        ----------
        deltaE : float
            Energy difference between the new and old configuration in eV.
        movement_name : str
            Name of the movement being performed (e.g., "translation" or "rotation").
        adsorbate_tag : int, optional
            Tag identifying the adsorbate molecule. Default is None.
        """

        exp_value = np.exp(-self.beta * deltaE)
        acc = min(1, exp_value)

        ads_name = next((ads.name for ads in self.adsorbates if ads.tag == adsorbate_tag), None)

        rnd_number = self.rnd_generator.random()

        if self.debug:
            self.logger.print_debug_movement(
                movement=movement_name,
                deltaE=deltaE,
                prefactor=1,
                acc=acc,
                rnd_number=rnd_number,
                adsorbate_name=ads_name,
            )

        # Apply Metropolis acceptance/rejection rule
        return rnd_number < acc

    def _nvemd_acceptance(self, deltaU: float, deltaK: float) -> bool:
        """
        Calculate the acceptance probability for a NVE-MD move of the system.

        Parameters
        ----------
        deltaU : float
            Change in potential energy.
        deltaK : float
            Change in kinetic energy.
        """

        deltaE = deltaU + deltaK
        exp_value = np.exp(-self.beta * deltaE)
        acc = min(1, exp_value)

        rnd_number = self.rnd_generator.random()

        if self.debug:
            self.logger.print_debug_movement(
                movement="NVE-MD",
                deltaE=deltaE,
                prefactor=1,
                acc=acc,
                rnd_number=rnd_number,
                adsorbate_name="System",
            )

        # Apply Metropolis acceptance/rejection rule
        return rnd_number < acc

    def _nvtmd_acceptance(self, deltaE: float) -> bool:
        """
        Calculate the acceptance probability for a NVT-MD move of the system.

        Parameters
        ----------
        deltaE : float
            Change in total energy.
        """

        exp_value = np.exp(-self.beta * deltaE)
        acc = min(1, exp_value)

        rnd_number = self.rnd_generator.random()

        if self.debug:
            self.logger.print_debug_movement(
                movement="NVT-MD",
                deltaE=deltaE,
                prefactor=1,
                acc=acc,
                rnd_number=rnd_number,
                adsorbate_name="System",
            )

        # Apply Metropolis acceptance/rejection rule
        return rnd_number < acc

    def _nptmd_acceptance(self, deltaE: float, v_old: float, v_new: float) -> bool:
        """
        Calculate the acceptance probability for a NPT-MD move of the system.

        Parameters
        ----------
        deltaE : float
            Change in total energy.
        v_old : float
            Old volume.
        v_new : float
            New volume.
        """

        detaH = deltaE + self.P * (v_new - v_old) - self.n_adsorbates * np.log(v_new / v_old) / self.beta

        exp_value = np.exp(-self.beta * detaH)
        acc = min(1, exp_value)

        rnd_number = self.rnd_generator.random()

        if self.debug:
            self.logger.print_debug_movement(
                movement="NPT-MD",
                deltaE=deltaE,
                prefactor=1,
                acc=acc,
                rnd_number=rnd_number,
                adsorbate_name='System',
            )

        # Apply Metropolis acceptance/rejection rule
        return rnd_number < acc

    def _save_state(self, actual_iteration: int) -> None:
        """
        Save the current state of the simulation to a file if
        the current iteration is a multiple of the save frequency.

        Parameters
        ----------
        actual_iteration : int
            The current iteration number of the simulation.
        """

        if actual_iteration % self.save_every == 0:

            np.save(
                os.path.join(self.out_folder, f"uptake_{self.P:.5f}.npy"),
                np.array(self.uptake_list),
            )

            np.save(
                os.path.join(self.out_folder, f"total_energy_{self.P:.5f}.npy"),
                np.array(self.total_energy_list),
            )

            np.save(
                os.path.join(self.out_folder, f"total_ads_{self.P:.5f}.npy"),
                np.array(self.total_ads_list),
            )

    def try_insertion(self, adsorbate_tag: int) -> bool:
        """
        Try to insert a new adsorbate molecule into the framework.
        This method randomly places the adsorbate in the framework and checks for van der Waals overlap.
        If there is no overlap, it calculates the new potential energy and decides whether to accept the insertion
        based on the acceptance criteria.

        Parameters
        ----------
        adsorbate_tag : int
            The tag of the adsorbate molecule to be inserted.

        Returns
        -------
        bool
            True if the insertion was accepted, False otherwise.
        """

        # Get the adsorbate index based on the provided tag
        adsorbate = next(
            (i for i, ads in enumerate(self.adsorbates) if ads.tag == adsorbate_tag), None
        )
        if adsorbate is None:
            raise ValueError(f"Adsorbate with tag {adsorbate_tag} not found.")

        atoms_trial = random_mol_insertion(
            framework=self.current_system,
            molecule=self.adsorbates[adsorbate].structure,
            rnd_generator=self.rnd_generator,
        )

        overlaped = check_overlap_vesin(
            atoms=atoms_trial,
            group1_indices=np.arange(len(self.current_system)),
            group2_indices=np.arange(len(self.current_system), len(atoms_trial)),
            vdw_radii=self.vdw,
        )

        if overlaped:
            return False

        # Energy calculation
        atoms_trial.calc = self.model
        e_new = atoms_trial.get_potential_energy()

        deltaE = (
            e_new
            - self.current_total_energy
            - self.adsorbate_energy[self.adsorbates[adsorbate].name]
        )

        if deltaE < -self.max_deltaE:
            self.logger._print_warning(
                f"WARNING: Energy difference {deltaE:.4f} eV exceeds the maximum allowed {self.max_deltaE:.4f} eV."
            )

        # Apply the acceptance criteria for insertion
        if self._insertion_acceptance(deltaE=deltaE, adsorbate_tag=adsorbate_tag):
            self.current_system = atoms_trial.copy()
            self.current_total_energy = e_new
            self.n_adsorbates[self.adsorbates[adsorbate].name] += 1
            return True

        self._save_rejected(atoms_trial)
        return False

    def try_deletion(self, adsorbate_tag: int) -> bool:
        """
        Try to delete an adsorbate molecule from the framework.
        This method randomly selects an adsorbate molecule and try to apply the deletion.

        If there are no adsorbates, it returns False.

        Parameters
        ----------
        adsorbate_tag : int
            The tag of the adsorbate molecule to be deleted.

        Returns
        -------
        bool
            True if the deletion was accepted, False otherwise.
        """
        ads_tags = list(set(self.current_system.get_tags()))

        if adsorbate_tag not in ads_tags:
            return False

        # Randomly select an adsorbate molecule to delete
        ads_indices = self.rnd_generator.choice(
            self.get_adsorbates_index(tag=adsorbate_tag), axis=0
        )

        mol_name = [
            adsorbate.name for adsorbate in self.adsorbates if adsorbate.tag == adsorbate_tag
        ][0]

        # Create a trial system for the deletion
        atoms_trial = self.current_system.copy()
        atoms_trial.calc = self.model  # type: ignore

        # Delete the adsorbate atoms from the trial structure
        del atoms_trial[ads_indices[0] : ads_indices[-1] + 1]

        # Calculate the new potential energy of the trial structure
        e_new = atoms_trial.get_potential_energy()  # type: ignore

        deltaE = e_new + self.adsorbate_energy[mol_name] - self.current_total_energy

        if deltaE < -self.max_deltaE:
            self.logger._print_warning(
                f"WARNING: Energy difference {deltaE:.4f} eV exceeds the maximum allowed {self.max_deltaE:.4f} eV."
            )

        # Apply the acceptance criteria for deletion
        if self._deletion_acceptance(deltaE=deltaE, adsorbate_tag=adsorbate_tag):

            self.current_system = atoms_trial.copy()
            self.current_total_energy = e_new
            self.n_adsorbates[mol_name] -= 1

            return True

        self._save_rejected(atoms_trial)
        return False

    def try_reinsertion(self, adsorbate_tag: int) -> bool:
        """
        Try to delete and reinsert an adsorbate molecule.
        This method randomly selects an adsorbate molecule, deletes it, and tries to reinsert it
        at a new random position within the framework.

        If there are no adsorbates, it returns False.

        Parameters
        ----------
        adsorbate_tag : int
            The tag of the adsorbate molecule to be reinserted.

        Returns
        -------
        bool
            True if the reinsertion was accepted, False otherwise.
        """

        ads_tags = list(set(self.current_system.get_tags()))

        if adsorbate_tag not in ads_tags:
            return False

        # Randomly select an adsorbate molecule to reinsertion
        ads_indices = self.rnd_generator.choice(
            self.get_adsorbates_index(tag=adsorbate_tag), axis=0
        )

        # Create a trial system for the deletion
        atoms_trial = self.current_system.copy()

        # Delete the adsorbate atoms from the trial structure
        del atoms_trial[ads_indices[0] : ads_indices[-1] + 1]

        to_reinsert = [ads.structure for ads in self.adsorbates if ads.tag == adsorbate_tag][0]

        temp = random_mol_insertion(atoms_trial, to_reinsert, self.rnd_generator)

        overlaped = check_overlap_vesin(
            atoms=temp,
            group1_indices=np.arange(len(atoms_trial)),
            group2_indices=np.arange(len(atoms_trial), stop=len(temp)),
            vdw_radii=self.vdw,
        )

        if overlaped:
            return False

        temp.calc = self.model  # type: ignore
        e_new = temp.get_potential_energy()

        deltaE = e_new - self.current_total_energy

        if deltaE < -self.max_deltaE:
            self.logger._print_warning(
                f"WARNING: Energy difference {deltaE:.4f} eV exceeds the maximum allowed {self.max_deltaE:.4f} eV."
            )

        # Apply the acceptance criteria for deletion
        if self._reinsertion_acceptance(deltaE=deltaE, adsorbate_tag=adsorbate_tag):

            self.current_system = temp.copy()
            self.current_total_energy = e_new

            return True

        self._save_rejected(temp)
        return False

    def try_translation(self, adsorbate_tag: int) -> bool:
        """
        Try to translate an adsorbate molecule within the framework.
        This method randomly selects an adsorbate molecule and applies a random translation.
        It checks for van der Waals overlap and calculates the new potential energy.

        Parameters
        ----------
        adsorbate_tag : int
            The tag of the adsorbate molecule to be translated.

        Returns
        -------
        bool
            True if the translation was accepted, False otherwise.
        """

        ads_tags = list(set(self.current_system.get_tags()))

        if adsorbate_tag not in ads_tags:
            return False

        ads_indices = self.rnd_generator.choice(
            self.get_adsorbates_index(tag=adsorbate_tag), axis=0
        )
        atoms_trial = self.current_system.copy()

        pos = atoms_trial.get_positions()  # type: ignore

        pos[ads_indices[0] : ads_indices[-1] + 1] = random_translation(
            original_position=pos[ads_indices[0] : ads_indices[-1] + 1],
            cell=self.current_system.cell.array,
            max_translation=self.max_translation,
            rnd_generator=self.rnd_generator,
        )

        atoms_trial.set_positions(pos)  # type: ignore

        overlaped = check_overlap_vesin(
            atoms=atoms_trial,
            group1_indices=np.concatenate(
                [np.arange(0, ads_indices[0]), np.arange(ads_indices[-1] + 1, len(atoms_trial))]
            ),
            group2_indices=np.arange(ads_indices[0], ads_indices[-1] + 1),
            vdw_radii=self.vdw,
        )

        if overlaped:
            return False

        atoms_trial.calc = self.model  # type: ignore
        e_trial = atoms_trial.get_potential_energy()  # type: ignore

        deltaE = e_trial - self.current_total_energy

        if deltaE < -self.max_deltaE:
            self.logger._print_warning(
                f"WARNING: Energy difference {deltaE:.4f} eV exceeds the maximum allowed {self.max_deltaE:.4f} eV."
            )

        if self._move_acceptance(
            deltaE=deltaE, movement_name="Translation", adsorbate_tag=adsorbate_tag
        ):
            self.current_system = atoms_trial.copy()
            self.current_total_energy = e_trial
            return True

        self._save_rejected(atoms_trial)
        return False

    def try_rotation(self, adsorbate_tag: int) -> bool:
        """
        Try to rotate an adsorbate molecule within the framework.
        This method randomly selects an adsorbate molecule and applies a random rotation.
        It checks for van der Waals overlap and calculates the new potential energy.

        Parameters
        ----------
        adsorbate_tag : int
            The tag of the adsorbate molecule to be rotated.

        Returns
        -------
        bool
            True if the rotation was accepted, False otherwise.
        """

        ads_tags = list(set(self.current_system.get_tags()))

        if adsorbate_tag not in ads_tags:
            return False

        atoms_trial = self.current_system.copy()
        pos = atoms_trial.get_positions()  # type: ignore

        # Randomly select an adsorbate molecule to rotate
        ads_indices = self.rnd_generator.choice(
            self.get_adsorbates_index(tag=adsorbate_tag), axis=0
        )

        pos[ads_indices[0] : ads_indices[-1] + 1] = random_rotation_limited(
            original_position=pos[ads_indices[0] : ads_indices[-1] + 1],
            cell=self.current_system.cell.array,
            rnd_generator=self.rnd_generator,
            theta_max=self.max_rotation,
        )
        atoms_trial.set_positions(pos)  # type: ignore

        overlaped = check_overlap_vesin(
            atoms=atoms_trial,
            group1_indices=np.concatenate(
                [np.arange(0, ads_indices[0]), np.arange(ads_indices[-1] + 1, len(atoms_trial))]
            ),
            group2_indices=np.arange(ads_indices[0], ads_indices[-1] + 1),
            vdw_radii=self.vdw,
        )

        if overlaped:
            return False

        atoms_trial.calc = self.model  # type: ignore
        e_trial = atoms_trial.get_potential_energy()  # type: ignore

        deltaE = e_trial - self.current_total_energy

        if deltaE < -self.max_deltaE:
            self.logger._print_warning(
                f"WARNING: Energy difference {deltaE:.4f} eV exceeds the maximum allowed {self.max_deltaE:.4f} eV."
            )

        if self._move_acceptance(
            deltaE=deltaE, movement_name="Rotation", adsorbate_tag=adsorbate_tag
        ):
            self.current_system = atoms_trial.copy()
            self.current_total_energy = e_trial
            return True

        self._save_rejected(atoms_trial)
        return False

    def try_identity_swap(self, adsorbate_tag: int) -> bool:
        """
        Try to swap the identity of two adsorbate molecules in the system.
        This method randomly selects two adsorbate molecules and attempts to swap their
        identities.

        Parameters
        ----------
        adsorbate_tag : int
            The tag of the first adsorbate molecule to be swapped.

        Returns
        -------
        bool
            True if the swap was accepted, False otherwise.
        """

        adsorbate1_tag = adsorbate_tag
        adsorbate2_tag = self.rnd_generator.choice(
            [ads.tag for ads in self.adsorbates if ads.tag != adsorbate1_tag]
        )

        ads1_name = next((ads.name for ads in self.adsorbates if ads.tag == adsorbate1_tag), 'None')
        ads2_name = next((ads.name for ads in self.adsorbates if ads.tag == adsorbate2_tag), 'None')

        # Check if both adsorbate tags are present in the system
        ads_tags = list(set(self.current_system.get_tags()))

        if adsorbate1_tag not in ads_tags:
            return False

        ads1_indices = self.rnd_generator.choice(
            self.get_adsorbates_index(tag=adsorbate1_tag), axis=0
        )

        atoms_trial = self.current_system.copy()
        atoms_trial.calc = self.model  # type: ignore

        cm_position = atoms_trial[ads1_indices[0] : ads1_indices[-1] + 1].get_center_of_mass() 

        to_insert = [ads.structure.copy() for ads in self.adsorbates if ads.tag == adsorbate2_tag][0]
        to_insert.set_positions(to_insert.get_positions() - to_insert.get_center_of_mass() + cm_position)  # type: ignore

        # Delete the adsorbate atoms from the trial structure
        del atoms_trial[ads1_indices[0] : ads1_indices[-1] + 1]

        atoms_trial += to_insert

        overlaped = check_overlap_vesin(
                    atoms=atoms_trial,
                    group1_indices=np.arange(len(atoms_trial) - len(to_insert)),
                    group2_indices=np.arange(len(atoms_trial) - len(to_insert), len(atoms_trial)),
                    vdw_radii=self.vdw,
                )

        if overlaped:
            return False


        atoms_trial.calc = self.model  # type: ignore
        e_trial = atoms_trial.get_potential_energy()  # type: ignore

        deltaE = e_trial - self.current_total_energy - self.adsorbate_energy[ads2_name] + self.adsorbate_energy[ads1_name]

        if deltaE < -self.max_deltaE:
            self.logger._print_warning(
                f"WARNING: Energy difference {deltaE:.4f} eV exceeds the maximum allowed {self.max_deltaE:.4f} eV."
            )

        if self._swap_acceptance(deltaE=deltaE, adsorbate_tags=[adsorbate1_tag, adsorbate2_tag]):
            self.current_system = atoms_trial.copy()
            self.current_total_energy = e_trial
            self.n_adsorbates[ads1_name] -= 1
            self.n_adsorbates[ads2_name] += 1
            return True

        self._save_rejected(atoms_trial)
        return False

    def try_nve_md(self, n_teps: int = 500, **kwargs) -> bool:
        raise NotImplementedError("NVE-MD move is not implemented yet.")

    def try_nvt_md(self, n_teps: int = 100, **kwargs) -> bool:
        """
        Try to perform a NVT-MD move of the system using the Nose-Hoover thermostat.

        Parameters
        ----------
        n_teps : int
            Number of time steps to run the NVT-MD simulation.

        Returns
        -------
        bool
            True if the NVT-MD move was accepted, False otherwise.
        """

        print(f"Attempting NVT-MD move with {n_teps} time steps...")

        temp_trajectory = Trajectory(
            os.path.join(self.out_folder, f"temp_nvtmd_{self.P:.5f}.traj"), "w"
        )

        atoms_trial = self.md(
            nsteps=n_teps,
            time_step=0.5,
            ensemble="NVT",
            thermostat="NoseHoover",
            calculator=self.model,
            update_state=False,
            trajectory_file=temp_trajectory,
        )

        if self._nvtmd_acceptance(deltaE=atoms_trial.get_potential_energy() - self.current_total_energy): # type: ignore
            self.current_system = atoms_trial.copy() # type: ignore
            self.current_total_energy = atoms_trial.get_potential_energy() # type: ignore
            print('NVT-MD move accepted.')

            print(temp_trajectory)

            # Append the temp_trajectory to the main trajectory file
            for frame in temp_trajectory:  # type: ignore
                self.trajectory.write(frame)  # type: ignore
            return True

        print('NVT-MD move rejected.')
        self._save_rejected(atoms_trial) # type: ignore
        return False

    def try_npt_md(self) -> bool:
        raise NotImplementedError("NPT-MD move is not implemented yet.")

    def _pick_random_move(self) -> tuple[int, str]:
        """
        Randomly select a move from the `move_weights` dict.
        If there is no molecule on the system, always return insertion.

        Returns
        -------
        adsorbate_tag: int
        move: str
        """

        # Pick a adsorbate based on the mol fractions
        ads = self.rnd_generator.choice(
            np.array(self.adsorbates), p=[ads.mol_fraction for ads in self.adsorbates]
        )

        move = ads.pick_random_move(self.rnd_generator)

        return ads.tag, move

    def get_total_ads_energy(self) -> float:
        """
        Compute the total adsorption energy of the system.

        The adsorption energy is calculated as:
            E_ads_total = E_total - sum_i(N_ads_i * E_adsorbate_i) - E_framework

        Returns
        -------
        total_adsorption_energy: float
            The total adsorption energy of the system in eV.
            Returns 0.0 if no molecules are adsorbed (N_ads == 0).
        """

        if sum(self.n_adsorbates.values()) == 0:
            return 0.0

        # Total adsorption energy (system - framework - isolated adsorbates * n adsorbates)
        total_adsorption_energy = self.current_total_energy - self.framework_energy

        for ads_name, n_ads in self.n_adsorbates.items():
            total_adsorption_energy -= n_ads * self.adsorbate_energy[ads_name]

        return total_adsorption_energy

    def get_average_ads_energy(self) -> float:
        """
        Compute the average adsorption energy per adsorbed molecule.

        The adsorption energy is calculated as:
            E_ads_avg = [E_total - sum_i(N_ads_i * E_adsorbate_i) - E_framework] / N_ads

        where:
            - E_total: current total energy of the system (simulation)
            - N_ads: number of adsorbed molecules
            - E_adsorbate: energy of an isolated adsorbate molecule
            - E_framework: energy of the empty framework

        The result is converted from simulation units to kJ/mol per adsorbate.

        Returns
        -------
        average_adsorption_energy: float
            The average adsorption energy per molecule in kJ/mol.
            Returns 0.0 if no molecules are adsorbed (N_ads == 0).
        """

        if sum(self.n_adsorbates.values()) == 0:
            return 0.0

        # Total adsorption energy (system - framework - isolated adsorbates * n adsorbates)
        total_adsorption_energy = self.get_total_ads_energy()

        # Convert to kJ/mol and normalize per adsorbate
        average_adsorption_energy = (
            total_adsorption_energy / (units.kJ / units.mol) / sum(self.n_adsorbates.values())
        )

        return average_adsorption_energy

    def step(self, iteration: int) -> None:
        """
        Perform a single Grand Canonical Monte Carlo step.
        It will randomly select a move based on the move weights and attempt to perform it.
        The uptake, total energy, and total adsorbates lists are updated accordingly.

        Parameters
        ----------
        iteration : int
            The current iteration number.
        """

        actual_iteration = iteration + self.base_iteration

        step_time_start = datetime.datetime.now()

        # Randomly select a move based on the move weights
        ads_tag, move = self._pick_random_move()

        ads_name = [ads.name for ads in self.adsorbates if ads.tag == ads_tag][0]

        accepted = self.movements[move](adsorbate_tag=ads_tag)
        self.n_movements[move].append(1 if accepted else 0)

        self.uptake_list = np.append(self.uptake_list, [list(self.n_adsorbates.values())], axis=0)
        self.total_energy_list.append(self.current_total_energy)
        self.total_ads_list.append(self.get_total_ads_energy())

        average_ads_energy = self.get_average_ads_energy()

        self.logger.print_step_info(
            step=actual_iteration,
            average_ads_energy=average_ads_energy,
            step_time=(datetime.datetime.now() - step_time_start).total_seconds(),
            adsorbate_name=ads_name,
        )

        # Save the current state and trajectory
        self._save_trajectory(actual_iteration)
        self._save_state(actual_iteration)

    def run(self, N) -> None:
        """Run the Grand Canonical Monte Carlo simulation for N iterations."""

        self.logger.print_run_header()

        for iteration in tqdm(range(1, N + 1), disable=(self.out_file is None), desc="GCMC Step"):
            self.step(iteration)
