import datetime
import json
import os

import ase
import numpy as np
from ase.calculators import calculator
from ase.io import Trajectory, read
from tqdm import tqdm

from flames import VERSION
from flames.base_simulator import BaseSimulator
from flames.logger import TMMCLogger
from flames.operations import (
    check_overlap,
    random_mol_insertion,
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

    :param vdw_factor:
        Factor to scale the Van der Waals radii. Default is ``0.6``.
    :type vdw_factor: float, optional

    :max_overlap_tries:
        Maximum tries for the insertion move. Default is ``100``.
    :type max_overlap_tries: int, optional

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
        adsorbate_atoms: ase.Atoms,
        temperature: float,
        pressure: float,
        device: str,
        vdw_radii: np.ndarray,
        vdw_factor: float = 0.6,
        max_overlap_tries: int = 100,
        max_deltaE: float = 25.0,
        save_frequency: int = 100,
        output_to_file: bool = True,
        output_folder: str | None = None,
        debug: bool = False,
        random_seed: int | None = None,
        cutoff_radius: float = 6.0,
    ) -> None:
        """
        Initialize the transition matrix Monte Carlo (TMMC) simulation.
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
            save_rejected=False,
            output_to_file=output_to_file,
            output_folder=output_folder,
            debug=debug,
            fugacity_coeff=0.0,
            random_seed=random_seed,
            cutoff_radius=cutoff_radius,
            automatic_supercell=False,
        )

        self.logger = TMMCLogger(simulation=self, output_file=self.out_file)

        self.start_time = datetime.datetime.now()

        # Parameters for storing the main results during the simulation
        self.total_ins_energy_list: list[float] = []
        self.total_del_energy_list: list[float] = []
        self.volume_list: list[float] = []
        self._del_indices: dict = {}

        # Maximum number of tries to insert a molecule without overlap
        self.max_overlap_tries = max_overlap_tries

        # Base iteration for restarting the simulation. This is for tracking the iteration count only
        self._base_iteration: int = 0

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

    def restart(self) -> None:
        """
        Restart the simulation from the last state.

        This method loads the last saved state from the trajectory file and restores the simulation to that state.
        It also loads the uptake, total energy, and total adsorbates lists from the saved files if they exist.
        """

        print("Restarting simulation...")

        ins_energy_restart, del_energy_restart = [], []

        if os.path.exists(
            os.path.join(self.out_folder, f"ins_ernergy_{self.n_adsorbates:04d}.npy")
        ):
            ins_energy_restart = np.load(
                os.path.join(self.out_folder, f"ins_ernergy_{self.n_adsorbates:04d}.npy")
            ).tolist()
        if os.path.exists(
            os.path.join(self.out_folder, f"del_ernergy_{self.n_adsorbates:04d}.npy")
        ):
            del_energy_restart = np.load(
                os.path.join(self.out_folder, f"del_ernergy_{self.n_adsorbates:04d}.npy")
            ).tolist()
        if os.path.exists(os.path.join(self.out_folder, f"volume_{self.n_adsorbates:04d}.npy")):
            volume_restart = np.load(
                os.path.join(self.out_folder, f"volume_{self.n_adsorbates:04d}.npy")
            ).tolist()

        # Check if the len of all restart elements are the same:
        if self.n_adsorbates == 0 and len(del_energy_restart) != 0:
            raise ValueError(
                """
            For 0 adsorbates the length of the deletion energy list should be zero.
            Please check the saved files."""
            )
        elif self.n_adsorbates > 0 and len(del_energy_restart) != len(ins_energy_restart):
            raise ValueError(
                f"""
            The lengths of insertion and deletion energy lists do not match.
            Please check the saved files.
            Found lengths: {len(ins_energy_restart)}, {len(del_energy_restart)}
            for insertion, and deletion energy respectively."""
            )

        self.total_ins_energy_list = ins_energy_restart
        self.total_del_energy_list = del_energy_restart
        self.volume_list = volume_restart

        # Set the base iteration to the length of the uptake list
        self.base_iteration = len(self.total_ins_energy_list)

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

        del state[-len(self.adsorbate) :]
        self.set_state(state)
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
            Name of the output file. Default is 'results_{T}_{n_adsorbates}.json'.
        """
        if file_name is None:
            file_name = f"results_{self.T}_{self.n_adsorbates:04d}.json"

        results = {
            "simulation": {
                "code_version": VERSION,
                "random_seed": self.random_seed,
                "temperature_K": self.T,
                "n_steps": len(self.total_ins_energy_list),
                "enlapsed_time_hours": (datetime.datetime.now() - self.start_time).total_seconds()
                / 3600,
            },
        }

        with open(os.path.join(self.out_folder, file_name), "w") as f:
            json.dump(results, f, indent=4)

    def _save_state(self, actual_iteration: int) -> None:
        if actual_iteration % self.save_every == 0:
            self.trajectory.write(self._current_ins_atoms)
            np.save(
                os.path.join(self.out_folder, f"ins_ernergy_{self.n_adsorbates:04d}.npy"),
                np.array(self.total_ins_energy_list),
            )
            np.save(
                os.path.join(self.out_folder, f"del_ernergy_{self.n_adsorbates:04d}.npy"),
                np.array(self.total_del_energy_list),
            )
            np.save(
                os.path.join(self.out_folder, f"volume_{self.n_adsorbates:04d}.npy"),
                np.array(self.volume_list),
            )

    def try_insertion(self):
        """
        Try to insert a new adsorbate molecule into the framework.
        This method randomly places the adsorbate in the framework and checks for van der Waals overlap.
        If there is no overlap, it calculates the new potential energy and decides whether to accept the insertion
        based on the acceptance criteria.
        If after a number of tries (self.max_overlap_tries) no valid position is found, the insertion is rejected.

        Returns
        -------
        deltaE
            Insertion energy.
        """
        for _ in range(self.max_overlap_tries):
            atoms_trial = random_mol_insertion(
                self.current_system, self.adsorbate, self.rnd_generator
            )

            overlaped = check_overlap(
                atoms=atoms_trial,
                group1_indices=np.arange(len(self.current_system)),
                group2_indices=np.arange(len(self.current_system), len(atoms_trial)),
                vdw_radii=self.vdw,
            )
            if overlaped:
                continue

            atoms_trial.calc = self.model
            e_new = atoms_trial.get_potential_energy()
            deltaE = e_new - self.current_total_energy - self.adsorbate_energy
            if np.abs(deltaE) > np.abs(self.max_deltaE):
                continue

            atoms_trial.info["ins_energy"] = deltaE
            atoms_trial.info["n_adsorbates"] = self.n_adsorbates
            self._current_ins_atoms = atoms_trial
            return deltaE
        raise ValueError("Could not insert molecule.")

    def try_deletion(self):
        """
        Try to delete an adsorbate molecule from the framework.
        This method randomly selects an adsorbate molecule and try to apply the deletion.

        Returns
        -------
        deltaE
            Deletion energy.
        """

        # Randomly select an adsorbate molecule to delete
        i_ads = self.rnd_generator.integers(low=0, high=self.n_adsorbates, size=1)[0]

        # Get the indices of the adsorbate atoms to be deleted
        i_start = self.n_atoms_framework + self.n_adsorbate_atoms * i_ads
        i_end = self.n_atoms_framework + self.n_adsorbate_atoms * (i_ads + 1)
        del_idx = tuple(list(range(i_start, i_end)))

        if del_idx in self._del_indices:
            deltaE = self._del_indices[del_idx]
        else:
            # Create a trial system for the deletion
            atoms_trial = self.current_system.copy()
            atoms_trial.calc = self.model  # type: ignore

            # Delete the adsorbate atoms from the trial structure
            del atoms_trial[i_start:i_end]

            # Calculate the new potential energy of the trial structure
            e_new = atoms_trial.get_potential_energy()  # type: ignore
            deltaE = e_new + self.adsorbate_energy - self.current_total_energy
            self._del_indices[del_idx] = deltaE

        self._current_ins_atoms.info["del_indices"] = del_idx
        self._current_ins_atoms.info["del_energy"] = deltaE
        return deltaE

    def run(self, N: int) -> None:
        """Run the transition matrix Monte Carlo simulation for N iterations."""

        self.logger.print_run_header()
        for iteration in tqdm(range(1, N + 1), disable=(self.out_file is None), desc="TMMC Step"):
            step_time_start = datetime.datetime.now()
            ins_energy = self.try_insertion()
            self.total_ins_energy_list.append(ins_energy)
            del_energy = self.try_deletion() if self.n_adsorbates > 0 else 0.0
            self.total_del_energy_list.append(del_energy)
            self.volume_list.append(self.framework.get_volume())
            self.logger.print_step_info(
                step=iteration + self.base_iteration,
                del_energy=del_energy,
                ins_energy=ins_energy,
                step_time=(datetime.datetime.now() - step_time_start).total_seconds(),
            )
            self._save_state(iteration + self.base_iteration)
