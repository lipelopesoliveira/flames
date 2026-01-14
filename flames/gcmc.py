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
from flames.base_simulator import BaseSimulator
from flames.eos import PengRobinsonEOS
from flames.logger import GCMCLogger
from flames.operations import (
    check_overlap,
    random_mol_insertion,
    random_rotation_limited,
    random_translation,
)
from flames.utilities import check_weights


class GCMC(BaseSimulator):
    """
    Base class for Grand Canonical Monte Carlo (GCMC) simulations using ASE.

    This class employs Monte Carlo simulations under the grand canonical ensemble (:math:`\mu VT`) to study the adsorption of molecules in a framework material.
    It allows for movements such as insertion, deletion, translation, and rotation of adsorbate molecules within the framework.

    Currently, it supports any ASE-compatible calculator for energy calculations.

    :param model:
        The calculator to use for energy calculations. Can be any ASE-compatible calculator.
        The output of the calculator should be in eV.
    :type model: ase.calculators.calculator.Calculator

    :param framework_atoms:
        The framework structure as an ASE Atoms object.
    :type framework_atoms: ase.Atoms

    :param adsorbate_atoms:
        The adsorbate structure as an ASE Atoms object.
    :type adsorbate_atoms: ase.Atoms

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

    :param LLM:
        Use Leftmost Local Minima (LLM) on the determination of the equilibration point by `pyMSER <https://github.com/lipelopesoliveira/pyMSER>`_.
        This can underestimate the equilibration point in some situations, but generate good averages for well-behaved scenarios.
    :type LLM: bool, optional

    :param move_weights:
        A dictionary containing the move weights for ``'insertion'``, ``'deletion'``, ``'translation'``, ``'rotation'``, and ``'reinsertion'``.
        Default is equal weights for all moves.
        Example: ``{'insertion': 0.3, 'deletion': 0.3, 'translation': 0.2, 'rotation': 0.2, 'reinsertion': 0.0}``
    :type move_weights: dict, optional

    """

    def __init__(
        self,
        model: calculator.Calculator,
        framework_atoms: ase.Atoms,
        adsorbate_atoms: ase.Atoms,
        temperature: float,
        pressure: float,
        device: str,
        vdw_radii: np.ndarray,
        vdw_factor: float = 0.6,
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
        criticalTemperature: float | None = None,
        criticalPressure: float | None = None,
        acentricFactor: float | None = None,
        void_fraction: float = 0.0,
        LLM: bool = False,
        move_weights: dict = {
            "insertion": 0.20,
            "deletion": 0.20,
            "translation": 0.20,
            "rotation": 0.20,
            "reinsertion": 0.20,
        },
    ) -> None:
        """
        Initialize the Grand Canonical Monte Carlo (GCMC) simulation.
        """

        super().__init__(
            model=model,
            framework_atoms=framework_atoms,
            adsorbate_atoms=adsorbate_atoms,
            temperature=temperature,
            pressure=pressure,
            device=device,
            vdw_radii=vdw_radii,
            vdw_factor=vdw_factor,
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
        self.criticalTemperature = criticalTemperature
        self.criticalPressure = criticalPressure
        self.acentricFactor = acentricFactor
        self.void_fraction = void_fraction
        self.excess_nmol = 0.0

        # Check if any critical parameters are not None
        if all([self.criticalTemperature, self.criticalPressure, self.acentricFactor]):
            self.eos = PengRobinsonEOS(
                temperature=self.T,
                pressure=self.P,
                criticalTemperature=self.criticalTemperature,  # type: ignore
                criticalPressure=self.criticalPressure,  # type: ignore
                acentricFactor=self.acentricFactor,  # type: ignore
                molarMass=self.adsorbate_mass * 1e3,  # convert kg/mol to g/mol
            )
            self.fugacity_coeff = self.eos.get_fugacity_coefficient()

            self.excess_nmol = self.eos.get_bulk_phase_molar_density() * self.V * self.void_fraction

        # Parameters for storing the main results during the simulation
        self._n_adsorbates: int = 0
        self.uptake_list: list[int] = []
        self.total_energy_list: list[float] = []
        self.total_ads_list: list[float] = []

        self._move_weights = check_weights(move_weights)

        self.max_translation = max_translation
        self.max_rotation = max_rotation

        self.mov_dict: dict = {
            "insertion": [],
            "deletion": [],
            "translation": [],
            "rotation": [],
            "reinsertion": [],
        }

        self.movements: dict = {
            "insertion": self.try_insertion,
            "deletion": self.try_deletion,
            "rotation": self.try_rotation,
            "translation": self.try_translation,
            "reinsertion": self.try_reinsertion,
        }

        # Base iteration for restarting the simulation. This is for tracking the iteration count only
        self._base_iteration: int = 0

        # Dictionary to store the equilibrated results by pyMSER
        self.equilibrated_results: dict = {}

        # Uses Leftmost Local Minima
        self.LLM = LLM

    @property
    def move_weights(self) -> dict:
        """
        Get the move weights for the GCMC simulation.

        Returns
        -------
        dict
            A dictionary containing the move weights for each type of movement.
        """
        return self._move_weights

    @move_weights.setter
    def move_weights(self, weights: dict) -> None:
        """
        Set the move weights for the GCMC simulation.

        Parameters
        ----------
        weights : dict
            A dictionary containing the move weights for each type of movement.
        """
        self._move_weights = check_weights(weights)

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

    @property
    def n_adsorbates(self) -> int:
        """
        Get the number of adsorbates in the current system.

        Returns
        -------
        int
            The number of adsorbates.
        """
        return self._n_adsorbates

    @n_adsorbates.setter
    def n_adsorbates(self, n: int) -> None:
        """
        Set the number of adsorbates in the current system.

        Parameters
        ----------
        n : int
            The number of adsorbates to set.
        """

        # Check if the number is a valid integer
        if not isinstance(n, int) or n < 0:
            raise ValueError("Number of adsorbates must be a non-negative integer.")

        n_adsorbate_atoms = len(self.current_system) - self.n_atoms_framework

        if n != int(n_adsorbate_atoms / self.n_adsorbate_atoms):
            raise ValueError(
                f"Number of adsorbates ({n}) is different from the number of adsorbate atoms in the system."
                f" Currently there are {int(n_adsorbate_atoms / self.n_adsorbate_atoms)} adsorbates."
            )

        self._n_adsorbates = n

    def _save_rejected(self, atoms_trial: ase.Atoms) -> None:
        """
        Helper to conditionally write the rejected configuration to the trajectory.

        Parameters
        ----------
        atoms_trial : ase.Atoms
            The trial configuration that was rejected.
        """
        if self.save_rejected:
            self.rejected_trajectory.write(atoms_trial)  # type: ignore

    def restart(self) -> None:
        """
        Restart the simulation from the last state.

        This method loads the last saved state from the trajectory file and restores the simulation to that state.
        It also loads the uptake, total energy, and total adsorbates lists from the saved files if they exist.
        """

        print("Restarting simulation...")
        uptake_restart, total_energy_restart, total_ads_restart = [], [], []

        if os.path.exists(os.path.join(self.out_folder, f"uptake_{self.P:.5f}.npy")):
            uptake_restart = np.load(
                os.path.join(self.out_folder, f"uptake_{self.P:.5f}.npy")
            ).tolist()

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
            raise ValueError(
                f"""
            The lengths of uptake, total energy, and total adsorbates lists do not match.
            Please check the saved files.
            Found lengths: {len(uptake_restart)}, {len(total_energy_restart)}, {len(total_ads_restart)}
            for uptake, total energy, and total ads respectively."""
            )

        self.uptake_list = uptake_restart
        self.total_energy_list = total_energy_restart
        self.total_ads_list = total_ads_restart

        # Set the base iteration to the length of the uptake list
        self.base_iteration = len(self.uptake_list)

        self.logger.print_restart_info()

        self.load_state(os.path.join(self.out_folder, "Movies", "Trajectory.traj"))

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

        self.set_state(state)

        self.n_adsorbates = int((len(state) - self.n_atoms_framework) / len(self.adsorbate))
        average_binding_energy = (
            (
                self.current_total_energy
                - self.framework_energy
                - self.n_adsorbates * self.adsorbate_energy
            )
            / (units.kJ / units.mol)
            / self.n_adsorbates
            if self.n_adsorbates > 0
            else 0
        )

        self.logger.print_load_state_info(
            n_atoms=len(state), average_ads_energy=average_binding_energy
        )

    def insert_adsorbates(self, n_adsorbates: int, max_attempts: int = 1000) -> None:
        """
        Insert a given number of adsorbate molecules into the framework at random positions without overlap.

        Parameters
        ----------
        n_molecules : int
            Number of adsorbate molecules to insert.
        max_attempts : int
            Maximum number of attempts to insert the molecules without overlap. Default is 1000.
        """
        temp_system = self.current_system.copy()

        n_attempts = 0
        inserted_adsorbates = 0
        while inserted_adsorbates < n_adsorbates and n_attempts < max_attempts:
            n_attempts += 1
            atoms_trial = random_mol_insertion(temp_system, self.adsorbate, self.rnd_generator)

            overlaped = check_overlap(
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
        self.n_adsorbates += inserted_adsorbates

    def equilibrate(
        self,
        batch_size: int | bool = False,
        run_ADF: bool = False,
        uncertainty: str = "uSD",
        production_start: int = 0,
    ) -> None:
        """
        Use pyMSER to get the equilibrated statistics of the simulation.

        Parameters
        ----------
        LLM : bool
            If True, use the Leftmost-Local Minima (LLM) method to determine the equilibration time.
            This is only recommended for high-throughput simulations, and sometimes can underestimate
            the true equilibration point.
            Default is True.
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
        production_start : int
            The step to start the production analysis from. Default is 0.

        """

        eq_results = pymser.equilibrate(
            self.uptake_list[production_start:],
            LLM=self.LLM,
            batch_size=(
                int(len(self.uptake_list[production_start:]) / 50)
                if batch_size is False
                else batch_size
            ),
            ADF_test=run_ADF,
            uncertainty=uncertainty,
            print_results=False,
        )

        enthalpy, enthalpy_sd = pymser.calc_equilibrated_enthalpy(
            energy=np.array(self.total_ads_list[production_start:]) / units.kB,  # Convert to K
            number_of_molecules=self.uptake_list[production_start:],
            temperature=self.T,
            eq_index=eq_results["t0"],
            uncertainty="SD",
            ac_time=int(eq_results["ac_time"]),
        )

        eq_results["average"] = float(eq_results["average"])
        eq_results["uncertainty"] = float(eq_results["uncertainty"])
        eq_results["ac_time"] = int(eq_results["ac_time"])
        eq_results["uncorr_samples"] = int(eq_results["uncorr_samples"])

        eq_results["equilibrated"] = eq_results["t0"] < 0.75 * len(
            self.uptake_list[production_start:]
        )

        eq_results["enthalpy_kJ_per_mol"] = float(enthalpy)
        eq_results["enthalpy_sd_kJ_per_mol"] = float(enthalpy_sd)

        self.equilibrated_results = eq_results

    def save_results(
        self,
        file_name: str | None = None,
        batch_size: int | bool = False,
        run_ADF: bool = False,
        uncertainty: str = "uSD",
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

        """

        if file_name is None:
            file_name = f"results_{self.T}_{self.P}.json"

        self.equilibrate(batch_size=batch_size, run_ADF=run_ADF, uncertainty=uncertainty)

        results = {
            "simulation": {
                "code_version": VERSION,
                "random_seed": self.random_seed,
                "temperature_K": self.T,
                "pressure_Pa": self.P,
                "fugacity_coefficient": self.fugacity_coeff,
                "fugacity_Pa": self.fugacity_coeff * self.P,
                "move_weights": self.move_weights,
                "n_steps": len(self.uptake_list),
                "enlapsed_time_hours": (datetime.datetime.now() - self.start_time).total_seconds()
                / 3600,
            },
            "equilibration": {
                "LLM": self.LLM,
                "t0": int(self.equilibrated_results.get("t0", 0)),
                "average": self.equilibrated_results.get("average", None),
                "uncertainty": self.equilibrated_results.get("uncertainty", None),
                "equilibrated": bool(self.equilibrated_results.get("equilibrated", False)),
                "ac_time": self.equilibrated_results.get("ac_time", None),
                "uncorr_samples": self.equilibrated_results.get("uncorr_samples", None),
            },
            "enthalpy": {
                "kJ_mol": {
                    "mean": self.equilibrated_results.get("enthalpy_kJ_per_mol", None),
                    "sd": self.equilibrated_results.get("enthalpy_sd_kJ_per_mol", None),
                }
            },
        }

        # --- Uptake data (computed from conversion factors) ---
        avrg = self.equilibrated_results.get("average", 0)
        stdv = self.equilibrated_results.get("uncertainty", 0)

        results["absolute_uptake"] = {
            unit: {
                "mean": avrg * factor,
                "sd": stdv * factor,
            }
            for unit, factor in self.conv_factors.items()
        }

        results["excess_uptake"] = {
            unit: {
                "mean": (avrg - self.excess_nmol) * factor,
                "sd": stdv * factor,
            }
            for unit, factor in self.conv_factors.items()
        }

        with open(os.path.join(self.out_folder, file_name), "w") as f:
            json.dump(results, f, indent=4)

    def _insertion_acceptance(self, deltaE) -> bool:
        """
        Calculate the acceptance probability for insertion of an adsorbate molecule as

        P_acc (N -> N + 1) = min(1, β * V * f * exp(-β ΔE) / (N + 1))

        Parameters
        ----------
        deltaE : float
            Energy difference between the new and old configuration in eV.
        """

        exp_value = np.exp(-self.beta * deltaE)

        pre_factor = self.V * self.beta * self.fugacity / (self.n_adsorbates + 1)

        acc = min(1, pre_factor * exp_value)

        rnd_number = self.rnd_generator.random()

        if self.debug:
            self.logger.print_debug_movement(
                movement="Insertion",
                deltaE=deltaE,
                prefactor=pre_factor,
                acc=acc,
                rnd_number=rnd_number,
            )

        # Apply Metropolis acceptance/rejection rule
        return rnd_number < acc

    def _deletion_acceptance(self, deltaE) -> bool:
        """
        Calculate the acceptance probability for deletion of an adsorbate molecule as

        P_del (N -> N - 1 ) = min(1, N / (β * V * f) * exp(-β ΔE) )

        Parameters
        ----------
        deltaE : float
            Energy difference between the new and old configuration in eV.
        """

        exp_value = np.exp(-self.beta * deltaE)

        pre_factor = self.n_adsorbates / (self.V * self.beta * self.fugacity)

        acc = min(1, pre_factor * exp_value)

        rnd_number = self.rnd_generator.random()

        if self.debug:
            self.logger.print_debug_movement(
                movement="Deletion",
                deltaE=deltaE,
                prefactor=pre_factor,
                acc=acc,
                rnd_number=rnd_number,
            )

        # Apply Metropolis acceptance/rejection rule
        return rnd_number < acc

    def _reinsertion_acceptance(self, deltaE) -> bool:
        """
        Calculate the acceptance probability for reinsertion of an adsorbate molecule as

        P_reins (N -> N ) = min(1, exp(-β ΔE) )

        Parameters
        ----------
        deltaE : float
            Energy difference between the new and old configuration in eV.
        """

        exp_value = np.exp(-self.beta * deltaE)
        acc = min(1, exp_value)

        rnd_number = self.rnd_generator.random()

        if self.debug:
            self.logger.print_debug_movement(
                movement="Reinsertion", deltaE=deltaE, prefactor=1, acc=acc, rnd_number=rnd_number
            )

        # Apply Metropolis acceptance/rejection rule
        return rnd_number < acc

    def _move_acceptance(self, deltaE, movement_name="Movement") -> bool:
        """
        Calculate the acceptance probability for translation or rotation of an adsorbate molecule as

        P_move = min(1, exp(-β ΔE))

        Parameters
        ----------
        deltaE : float
            Energy difference between the new and old configuration in eV.
        """

        exp_value = np.exp(-self.beta * deltaE)
        acc = min(1, exp_value)

        rnd_number = self.rnd_generator.random()

        if self.debug:
            self.logger.print_debug_movement(
                movement=movement_name, deltaE=deltaE, prefactor=1, acc=acc, rnd_number=rnd_number
            )

        # Apply Metropolis acceptance/rejection rule
        return rnd_number < acc

    def _save_state(self, actual_iteration: int) -> None:

        if actual_iteration % self.save_every == 0:

            self.trajectory.write(self.current_system)  # type: ignore

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

    def try_insertion(self) -> bool:
        """
        Try to insert a new adsorbate molecule into the framework.
        This method randomly places the adsorbate in the framework and checks for van der Waals overlap.
        If there is no overlap, it calculates the new potential energy and decides whether to accept the insertion
        based on the acceptance criteria.

        Returns
        -------
        bool
            True if the insertion was accepted, False otherwise.
        """

        atoms_trial = random_mol_insertion(self.current_system, self.adsorbate, self.rnd_generator)

        overlaped = check_overlap(
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

        deltaE = e_new - self.current_total_energy - self.adsorbate_energy

        if np.abs(deltaE) > np.abs(self.max_deltaE):
            self._save_rejected(atoms_trial)
            return False

        # Apply the acceptance criteria for insertion
        if self._insertion_acceptance(deltaE=deltaE):
            self.current_system = atoms_trial.copy()
            self.current_total_energy = e_new
            self.n_adsorbates += 1
            return True

        self._save_rejected(atoms_trial)
        return False

    def try_deletion(self) -> bool:
        """
        Try to delete an adsorbate molecule from the framework.
        This method randomly selects an adsorbate molecule and try to apply the deletion.

        If there are no adsorbates, it returns False.

        Returns
        -------
        bool
            True if the deletion was accepted, False otherwise.
        """
        if self.n_adsorbates == 0:
            return False

        # Randomly select an adsorbate molecule to delete
        i_ads = self.rnd_generator.integers(low=0, high=self.n_adsorbates, size=1)[0]

        # Get the indices of the adsorbate atoms to be deleted
        i_start = self.n_atoms_framework + self.n_adsorbate_atoms * i_ads
        i_end = self.n_atoms_framework + self.n_adsorbate_atoms * (i_ads + 1)

        # Create a trial system for the deletion
        atoms_trial = self.current_system.copy()
        atoms_trial.calc = self.model  # type: ignore

        # Delete the adsorbate atoms from the trial structure
        del atoms_trial[i_start:i_end]

        # Calculate the new potential energy of the trial structure
        e_new = atoms_trial.get_potential_energy()  # type: ignore

        deltaE = e_new + self.adsorbate_energy - self.current_total_energy

        if np.abs(deltaE) > np.abs(self.max_deltaE):
            return False

        # Apply the acceptance criteria for deletion
        if self._deletion_acceptance(deltaE=deltaE):

            self.current_system = atoms_trial.copy()
            self.current_total_energy = e_new
            self.n_adsorbates -= 1

            return True
        else:
            return False

    def try_reinsertion(self) -> bool:
        """
        Try to delete and reinsert an adsorbate molecule.
        This method randomly selects an adsorbate molecule, deletes it, and tries to reinsert it
        at a new random position within the framework.

        If there are no adsorbates, it returns False.

        Returns
        -------
        bool
            True if the reinsertion was accepted, False otherwise.
        """

        if self.n_adsorbates == 0:
            return False

        # Randomly select an adsorbate molecule to delete
        i_ads = self.rnd_generator.integers(low=0, high=self.n_adsorbates, size=1)[0]

        # Get the indices of the adsorbate atoms to be deleted
        i_start = self.n_atoms_framework + self.n_adsorbate_atoms * i_ads
        i_end = self.n_atoms_framework + self.n_adsorbate_atoms * (i_ads + 1)

        # Create a trial system for the deletion
        atoms_trial = self.current_system.copy()

        # Delete the adsorbate atoms from the trial structure
        del atoms_trial[i_start:i_end]

        inserted = False
        for _ in range(1000):
            # Try at least 1000 times to insert the molecule without overlap
            temp = random_mol_insertion(atoms_trial, self.adsorbate, self.rnd_generator)

            overlaped = check_overlap(
                atoms=temp,
                group1_indices=np.arange(len(atoms_trial)),
                group2_indices=np.arange(start=len(atoms_trial), stop=len(temp)),
                vdw_radii=self.vdw,
            )

            if not overlaped:
                inserted = True
                atoms_trial = temp
                break

        if not inserted:
            return False

        atoms_trial.calc = self.model  # type: ignore
        e_new = atoms_trial.get_potential_energy()

        deltaE = e_new - self.current_total_energy

        if np.abs(deltaE) > np.abs(self.max_deltaE):
            self._save_rejected(atoms_trial)
            return False

        # Apply the acceptance criteria for deletion
        if self._reinsertion_acceptance(deltaE=deltaE):

            self.current_system = atoms_trial.copy()
            self.current_total_energy = e_new

            return True
        else:
            return False

    def try_translation(self) -> bool:
        """
        Try to translate an adsorbate molecule within the framework.
        This method randomly selects an adsorbate molecule and applies a random translation.
        It checks for van der Waals overlap and calculates the new potential energy.

        Returns
        -------
        bool
            True if the translation was accepted, False otherwise.
        """

        if self.n_adsorbates == 0:
            return False

        i_ads = self.rnd_generator.integers(low=0, high=self.n_adsorbates, size=1)[0]
        atoms_trial = self.current_system.copy()

        pos = atoms_trial.get_positions()  # type: ignore

        i_start = self.n_atoms_framework + self.n_adsorbate_atoms * i_ads
        i_end = self.n_atoms_framework + self.n_adsorbate_atoms * (i_ads + 1)

        pos[i_start:i_end] = random_translation(
            original_position=pos[i_start:i_end],
            cell=self.current_system.cell.array,
            max_translation=self.max_translation,
            rnd_generator=self.rnd_generator,
        )

        atoms_trial.set_positions(pos)  # type: ignore

        overlaped = check_overlap(
            atoms=atoms_trial,
            group1_indices=np.concatenate(
                [np.arange(0, i_start), np.arange(i_end, len(atoms_trial))]
            ),
            group2_indices=np.arange(i_start, i_end),
            vdw_radii=self.vdw,
        )

        if overlaped:
            return False

        atoms_trial.calc = self.model  # type: ignore
        e_trial = atoms_trial.get_potential_energy()  # type: ignore

        deltaE = e_trial - self.current_total_energy

        if np.abs(deltaE) > np.abs(self.max_deltaE):
            self._save_rejected(atoms_trial)
            return False

        if self._move_acceptance(deltaE=deltaE, movement_name="Translation"):
            self.current_system = atoms_trial.copy()
            self.current_total_energy = e_trial
            return True
        else:
            self._save_rejected(atoms_trial)
            return False

    def try_rotation(self) -> bool:
        """
        Try to rotate an adsorbate molecule within the framework.
        This method randomly selects an adsorbate molecule and applies a random rotation.
        It checks for van der Waals overlap and calculates the new potential energy.

        Returns
        -------
        bool
            True if the rotation was accepted, False otherwise.
        """

        if self.n_adsorbates == 0:
            return False

        i_ads = self.rnd_generator.integers(low=0, high=self.n_adsorbates, size=1)[0]
        atoms_trial = self.current_system.copy()

        pos = atoms_trial.get_positions()  # type: ignore
        i_start = self.n_atoms_framework + self.n_adsorbate_atoms * i_ads
        i_end = self.n_atoms_framework + self.n_adsorbate_atoms * (i_ads + 1)

        pos[i_start:i_end] = random_rotation_limited(
            original_position=pos[i_start:i_end],
            cell=self.current_system.cell.array,
            rnd_generator=self.rnd_generator,
            theta_max=self.max_rotation,
        )
        atoms_trial.set_positions(pos)  # type: ignore

        overlaped = check_overlap(
            atoms=atoms_trial,
            group1_indices=np.concatenate(
                [np.arange(0, i_start), np.arange(i_end, len(atoms_trial))]
            ),
            group2_indices=np.arange(i_start, i_end),
            vdw_radii=self.vdw,
        )

        if overlaped:
            return False

        atoms_trial.calc = self.model  # type: ignore
        e_trial = atoms_trial.get_potential_energy()  # type: ignore

        deltaE = e_trial - self.current_total_energy

        if np.abs(deltaE) > np.abs(self.max_deltaE):
            self._save_rejected(atoms_trial)
            return False

        if self._move_acceptance(deltaE=deltaE, movement_name="Rotation"):
            self.current_system = atoms_trial.copy()
            self.current_total_energy = e_trial
            return True
        else:
            self._save_rejected(atoms_trial)
            return False

    def _pick_random_move(self) -> str:
        """
        Randomly select a move from the `move_weights` dict.
        If there is no molecule on the system, always return insertion.

        Returns
        -------
        move: str
        """

        if self.n_adsorbates == 0:
            move = "insertion"

        else:
            move = self.rnd_generator.choice(
                a=list(self.move_weights.keys()), p=list(self.move_weights.values())
            )

        return move

    def get_average_ads_energy(self) -> float:
        """
        Compute the average adsorption energy per adsorbed molecule.

        The adsorption energy is calculated as:
            E_ads_avg = [E_total - (N_ads * E_adsorbate) - E_framework] / N_ads

        where:
            - E_total: current total energy of the system (simulation)
            - N_ads: number of adsorbed molecules
            - E_adsorbate: energy of an isolated adsorbate molecule
            - E_framework: energy of the empty framework

        The result is converted from simulation units to kJ/mol per adsorbate.

        Returns
        -------
        float
            The average adsorption energy per molecule in kJ/mol.
            Returns 0.0 if no molecules are adsorbed (N_ads == 0).
        """

        if self.n_adsorbates == 0:
            return 0.0

        # Total adsorption energy (system - framework - isolated adsorbates * n adsorbates)
        adsorption_energy_total = (
            self.current_total_energy
            - self.framework_energy
            - (self.n_adsorbates * self.adsorbate_energy)
        )

        # Convert to kJ/mol and normalize per adsorbate
        E_ads_avg = adsorption_energy_total / (units.kJ / units.mol) / self.n_adsorbates

        return E_ads_avg

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
        move = self._pick_random_move()

        accepted = self.movements[move]()
        self.mov_dict[move].append(1 if accepted else 0)

        self.uptake_list.append(self.n_adsorbates)
        self.total_energy_list.append(self.current_total_energy)
        self.total_ads_list.append(
            self.current_total_energy
            - (self.n_adsorbates * self.adsorbate_energy)
            - self.framework_energy
        )

        average_ads_energy = self.get_average_ads_energy()

        self.logger.print_step_info(
            step=actual_iteration,
            average_ads_energy=average_ads_energy,
            step_time=(datetime.datetime.now() - step_time_start).total_seconds(),
        )

        self._save_state(actual_iteration)

    def run(self, N) -> None:
        """Run the Grand Canonical Monte Carlo simulation for N iterations."""

        self.logger.print_run_header()

        for iteration in tqdm(range(1, N + 1), disable=(self.out_file is None), desc="GCMC Step"):
            self.step(iteration)
