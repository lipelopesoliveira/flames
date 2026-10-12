from __future__ import annotations

import datetime
import json
import os

import ase
import numpy as np
from ase import units
from ase.calculators import calculator
from ase.io import Trajectory, read
from tqdm import tqdm

from flames import VERSION
from flames.adsorbate import Adsorbate
from flames.base_simulator import BaseSimulator
from flames.logger import TMMCLogger
from flames.operations import (
    check_overlap_vesin,
    random_mol_insertion,
    random_rotation_limited,
    random_translation,
)


class TMMC(BaseSimulator):
    """
    Base class for transition matrix Monte Carlo (TMMC) simulations using ASE.

    This class implements TMMC deletion/insertion moves, recording the
    deletion/insertion energies of the adsorbate.

    :param model:
        The calculator to use for energy calculations. Can be any ASE-compatible calculator.
        The output of the calculator should be in eV.
    :type model: ase.calculators.calculator.Calculator

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
            save_rejected=False,
            output_to_file=output_to_file,
            output_folder=output_folder,
            debug=debug,
            fugacity_coeff=0.0,
            random_seed=random_seed,
            cutoff_radius=cutoff_radius,
            automatic_supercell=automatic_supercell,
        )

        self.logger = TMMCLogger(simulation=self, output_file=self.out_file)

        self.start_time = datetime.datetime.now()

        # Tracking number of adsorbates per species
        self.n_adsorbates: dict[str, int] = {adsorbate.name: 0 for adsorbate in self.adsorbates}

        self.uptake_list: np.ndarray = np.zeros((1, len(self.adsorbates)), dtype=int)
        self.total_energy_list: list[float] = [0]
        self.total_ads_list: list[float] = [0]

        # Parameters for storing the main results during the simulation (partitioned by adsorbate)
        self.total_ins_energy_list: dict[str, list[float]] = {
            ads.name: [] for ads in self.adsorbates
        }
        self.total_del_energy_list: dict[str, list[float]] = {
            ads.name: [] for ads in self.adsorbates
        }
        self.volume_list: list[float] = []
        self._del_indices: dict = {}

        self.max_translation = max_translation
        self.max_rotation = max_rotation

        # Base iteration for restarting the simulation. This is for tracking the iteration count only
        self._base_iteration: int = 0

        self.movements: dict = {
            "insertion": self.try_insertion,
            "deletion": self.try_deletion,
            "rotation": self.try_rotation,
            "translation": self.try_translation,
            "reinsertion": self.try_reinsertion,
            "identity_swap": self.try_identity_swap,
        }

        self.n_movements = {move: [] for move in self.movements.keys()}

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

    def get_adsorbates_index(self, tag: int | None = None) -> list[list]:
        """
        Get a list of indices of the adsorbate molecules in the current system.
        """
        adsorbates_list = []

        for adsorbate in self.adsorbates:
            if tag is not None and adsorbate.tag != tag:
                continue

            indices = np.where(self.current_system.get_tags() == adsorbate.tag)[0]

            if len(indices) > 0:
                adsorbates_list.extend(indices.reshape(-1, len(adsorbate.structure)).tolist())

        return adsorbates_list

    def get_macrostate_str(self) -> str:
        """
        Return a string representation of the current multi-component macrostate.
        Example: For a binary mixture with 1 CO2 and 2 H2O, returns "0001_0002".
        """
        return "_".join([f"{self.n_adsorbates[ads.name]:04d}" for ads in self.adsorbates])

    def restart(self) -> None:
        """
        Restart the simulation from the last state.

        This method loads the last saved state from the trajectory file and restores the simulation to that state.
        It also loads the arrays handling the deletion/insertion energies.
        """

        print("Restarting simulation...")

        macrostate_str = self.get_macrostate_str()

        for ads in self.adsorbates:
            ins_file = os.path.join(self.out_folder, f"ins_energy_{ads.name}_{macrostate_str}.npy")
            del_file = os.path.join(self.out_folder, f"del_energy_{ads.name}_{macrostate_str}.npy")

            if os.path.exists(ins_file):
                self.total_ins_energy_list[ads.name] = np.load(ins_file).tolist()
            if os.path.exists(del_file):
                self.total_del_energy_list[ads.name] = np.load(del_file).tolist()

        vol_file = os.path.join(self.out_folder, f"volume_{macrostate_str}.npy")
        if os.path.exists(vol_file):
            self.volume_list = np.load(vol_file).tolist()

        # Set the base iteration to the length of the list (using the first adsorbate as reference)
        first_ads_name = self.adsorbates[0].name
        if len(self.total_ins_energy_list[first_ads_name]) > 0:
            self.base_iteration = len(self.total_ins_energy_list[first_ads_name])

        self.logger.print_restart_info()

        if os.path.exists(os.path.join(self.out_folder, "Movies", "Trajectory.traj")):
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

        # Workaround to load the labels from Trajectory.info since ASE's Trajectory does not support custom arrays
        if "labels" in state.info.keys():
            state.set_array("labels", state.info["labels"])

        # Trim off the molecule that was temporarily appended during the last saved insertion attempt
        if "inserted_length" in state.info:
            del state[-state.info["inserted_length"] :]

        self.set_state(state)
        self.n_adsorbates = self.get_number_of_adsorbates(state)

        self.logger.print_load_state_info(n_atoms=len(state))

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

    def _save_state(self, actual_iteration: int) -> None:
        """
        Save the simulation trajectory and generated energies.
        """
        if actual_iteration % self.save_every == 0:
            if hasattr(self, "_current_ins_atoms"):
                self.trajectory.write(self._current_ins_atoms)  # type: ignore
            else:
                self.trajectory.write(self.current_system)  # type: ignore

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
                os.path.join(self.out_folder, f"volume_{macrostate_str}.npy"),
                np.array(self.volume_list),
            )

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

        to_reinsert = atoms_trial[ads_indices[0] : ads_indices[-1] + 1].copy()

        # Delete the adsorbate atoms from the trial structure
        del atoms_trial[ads_indices[0] : ads_indices[-1] + 1]

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

        if len(self.adsorbates) < 2:
            return False
    
        adsorbate1_tag = adsorbate_tag
        adsorbate2_tag = self.rnd_generator.choice(
            [ads.tag for ads in self.adsorbates if ads.tag != adsorbate1_tag]
        )

        ads1_name = next((ads.name for ads in self.adsorbates if ads.tag == adsorbate1_tag), "None")
        ads2_name = next((ads.name for ads in self.adsorbates if ads.tag == adsorbate2_tag), "None")

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

        to_insert = [ads.structure.copy() for ads in self.adsorbates if ads.tag == adsorbate2_tag][
            0
        ]
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

        deltaE = (
            e_trial
            - self.current_total_energy
            - self.adsorbate_energy[ads2_name]
            + self.adsorbate_energy[ads1_name]
        )

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

    def try_virtual_insertion(self, adsorbate_tag: int) -> float:
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
        """
        adsorbate = next((ads for ads in self.adsorbates if ads.tag == adsorbate_tag), None)

        if adsorbate is None:
            raise ValueError(f"Adsorbate with tag {adsorbate_tag} not found.")


        atoms_trial = random_mol_insertion(
            self.current_system, adsorbate.structure, self.rnd_generator
        )

        overlaped = check_overlap_vesin(
            atoms=atoms_trial,
            group1_indices=np.arange(len(self.current_system)),
            group2_indices=np.arange(len(self.current_system), len(atoms_trial)),
            vdw_radii=self.vdw,
        )
        if overlaped:
            return self.max_deltaE * 100

        atoms_trial.calc = self.model
        e_new = atoms_trial.get_potential_energy()

        deltaE = e_new - self.current_total_energy - self.adsorbate_energy[adsorbate.name]

        if deltaE < -self.max_deltaE:
            self.logger._print_warning(
                f"WARNING: Energy difference {deltaE:.4f} eV exceeds the maximum allowed {self.max_deltaE:.4f} eV."
            )

        atoms_trial.info["ins_energy"] = deltaE
        atoms_trial.info["n_adsorbates"] = self.n_adsorbates
        atoms_trial.info["inserted_length"] = len(adsorbate.structure)
        self._current_ins_atoms = atoms_trial

        return deltaE

    def try_virtual_deletion(self, adsorbate_tag: int) -> float:
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
        """
        ads_name = next((ads.name for ads in self.adsorbates if ads.tag == adsorbate_tag), None)

        if ads_name is None:
            raise ValueError(f"Adsorbate with tag {adsorbate_tag} not found.")

        if self.n_adsorbates[ads_name] == 0:
            return 0.0

        # Randomly select an adsorbate molecule of this specific tag to delete
        ads_indices_list = self.get_adsorbates_index(tag=adsorbate_tag)
        ads_indices = self.rnd_generator.choice(ads_indices_list, axis=0)

        del_idx = tuple(ads_indices)

        if del_idx in self._del_indices:
            deltaE = self._del_indices[del_idx]
        else:
            atoms_trial = self.current_system.copy()
            atoms_trial.calc = self.model  # type: ignore

            # Delete the selected adsorbate atoms from the trial structure
            del atoms_trial[ads_indices[0] : ads_indices[-1] + 1]

            e_new = atoms_trial.get_potential_energy()  # type: ignore
            deltaE = e_new + self.adsorbate_energy[ads_name] - self.current_total_energy
            self._del_indices[del_idx] = deltaE

        if hasattr(self, "_current_ins_atoms"):
            self._current_ins_atoms.info["del_indices"] = del_idx
            self._current_ins_atoms.info["del_energy"] = deltaE

        return deltaE

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

    def gcmc_step(self, iteration: int) -> None:
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

        average_ads_energy = self.get_average_ads_energy()

        self.logger.print_gcmc_step_info(
            step=actual_iteration,
            average_ads_energy=average_ads_energy,
            step_time=(datetime.datetime.now() - step_time_start).total_seconds(),
            adsorbate_name=ads_name,
        )

        # Save the current state and trajectory
        self._save_trajectory(actual_iteration)
        self._save_state(actual_iteration)

    def run_gcmc(self, N) -> None:
        """Run the Grand Canonical Monte Carlo simulation for N iterations."""

        self.logger.print_gcmc_run_header()

        for iteration in tqdm(range(1, N + 1), disable=(self.out_file is None), desc="GCMC Step"):
            self.gcmc_step(iteration)

    def tmmc_step(self, iteration: int) -> None:
        """
        Perform a single Transition Matrix Monte Carlo step.
        It will attempt virtual insertions and deletions for all adsorbates and record the corresponding energies.

        Parameters
        ----------
        iteration : int
            The current iteration number.
        """

        actual_iteration = iteration + self.base_iteration

        step_time_start = datetime.datetime.now()

        ins_energies = {}
        del_energies = {}

        # TMMC probes insertion and deletion for all components at the fixed macrostate
        for ads in self.adsorbates:
            ins_energy = self.try_virtual_insertion(ads.tag)
            self.total_ins_energy_list[ads.name].append(ins_energy)
            ins_energies[ads.name] = ins_energy

            del_energy = self.try_virtual_deletion(ads.tag) if self.n_adsorbates[ads.name] > 0 else 0.0
            self.total_del_energy_list[ads.name].append(del_energy)
            del_energies[ads.name] = del_energy

        self.volume_list.append(self.current_system.get_volume())

        self.logger.print_tmmc_step_info(
            step=actual_iteration,
            del_energy=del_energies,
            ins_energy=ins_energies,
            step_time=(datetime.datetime.now() - step_time_start).total_seconds(),
        )

        # Save the current state and trajectory
        self._save_state(actual_iteration)

    def run_tmmc(self, N: int) -> None:
        """
        Run the transition matrix Monte Carlo simulation for N iterations.

        For each iteration, no real move are actually performed on the system,
        but virtual insertion and deletion are performed and the corresponding
        energies and the unit cell volume are recorded.
        """

        self.logger.print_tmmc_run_header()

        for iteration in tqdm(range(1, N + 1), disable=(self.out_file is None), desc="TMMC Step"):

            self.tmmc_step(iteration)
