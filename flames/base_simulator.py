import os
from typing import TextIO

import ase
import numpy as np
from ase import units
from ase.build import make_supercell
from ase.calculators import calculator
from ase.io import Trajectory
from ase.io.trajectory import TrajectoryReader, TrajectoryWriter
from ase.optimize import LBFGS

from flames.adsorbate import Adsorbate
from flames.ase_utils import (
    crystalOptimization,
    nPT_Berendsen,
    nPT_MTKNPT,
    nPT_NoseHoover,
    nVT_Berendsen,
    nVT_Langevin,
    nVT_NoseHoover,
)
from flames.logger import BaseLogger
from flames.md import run_md_simulation
from flames.utilities import (
    calculate_unit_cells,
    get_density,
    get_perpendicular_lengths,
)


class BaseSimulator:
    """
    This class handles all base parameters for the simulations.
    Separates the basic logic used in the simulations from the specific implementation details.

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

    :param framework_energy:
        Total energy of the framework.
    :type framework_energy: float

    :param adsorbate_energy:
        Total energy of the adsorbate.
    :type adsorbate_energy: float

    :param vdw_factor:
        Factor to scale the Van der Waals radii. Default is ``0.6``.
    :type vdw_factor: float, optional

    :param max_deltaE:
        Maximum energy difference (in eV) to consider for acceptance criteria.
        This is used to avoid overflow due to problematic calculations. Default is ``1.555`` eV (approx. 150 kJ/mol).
    :type max_deltaE: float, optional

    :param save_frequency:
        Frequency at which to save the simulation state and results. Default is ``100``.
    :type save_frequency: int, optional

    :param save_rejected:
        If ``True``, saves the rejected moves in a trajectory file. Default is ``False``.
    :type save_rejected: bool, optional

    :param output_to_file:
        If ``True``, writes the output to a file named ``GCMC_Output.out`` in the results directory.
    :type output_to_file: bool, optional

    :param output_folder:
        Folder to save the output files. If ``None``, a folder named ``results_<T>_<P>`` will be created
        based on the temperature and pressure. Default is ``None``.
    :type output_folder: str or None, optional

    :param debug:
        If ``True``, prints detailed debug information during the simulation. Default is ``False``.
    :type debug: bool, optional

    :param fugacity_coeff:
        Fugacity coefficient to correct the pressure. Default is ``1.0``.
        Only used if ``criticalTemperature``, ``criticalPressure``, and ``acentricFactor`` are not provided.
    :type fugacity_coeff: float, optional

    :param random_seed:
        Random seed for reproducibility. Default is ``None``.
    :type random_seed: int or None, optional

    :param cutoff_radius:
        Interaction potential cut-off radius used to estimate the minimum unit cell. Default is ``6.0``.
    :type cutoff_radius: float, optional

    :param automatic_supercell:
        If ``True``, automatically creates a supercell based on the cutoff radius. Default is ``True``.
    :type automatic_supercell: bool, optional
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
        framework_energy: float | None = None,
        adsorbate_energy: float | None = None,
        vdw_factor: float = 0.6,
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
    ):

        self.random_seed = random_seed if random_seed is not None else np.random.randint(1, 1000000)
        self.rnd_generator = np.random.default_rng(self.random_seed)
        self.cutoff = cutoff_radius
        self.automatic_supercell = automatic_supercell

        # -- General definitions for output --
        if output_folder is not None:
            self.out_folder = output_folder
        else:
            self.out_folder = f"results_{temperature:.2f}_{pressure:.2f}"

        os.makedirs(self.out_folder, exist_ok=True)
        os.makedirs(os.path.join(self.out_folder, "Movies"), exist_ok=True)

        if output_to_file:
            self.out_file: TextIO | None = open(os.path.join(self.out_folder, "Output.out"), "a")
        else:
            self.out_file = None

        self.trajectory = Trajectory(
            os.path.join(self.out_folder, "Movies", "Trajectory.traj"),
            "a",
        )

        self.logger = BaseLogger(self, output_file=self.out_file)

        self.save_rejected = save_rejected

        if self.save_rejected:
            self.rejected_trajectory = Trajectory(
                os.path.join(self.out_folder, "Movies", "Trajectory_rejected.traj"),
                "a",
            )

        self.save_every = save_frequency
        self.debug = debug

        # -- General definitions for framework and adsorbate --

        self.model = model
        self.device = device

        self.set_framework(framework_atoms, framework_energy=framework_energy)

        self.current_system = self.framework.copy()
        self.current_system.calc = self.model

        if framework_energy:
            self.current_total_energy = framework_energy * np.prod(self.ideal_supercell)  # type: ignore
        else:
            self.current_total_energy = self.framework_energy  # type: ignore

        self.set_adsorbates(adsorbates, adsorbate_energy=adsorbate_energy)

        self.mol_fractions = {
            adsorbate.name: adsorbate.mol_fraction for adsorbate in self.adsorbates
        }

        # General definitions for simulation parameters
        self.T = temperature
        self.P = pressure

        self.fugacity_coeff = fugacity_coeff

        # Fugacity from Pa (J/m^3) to eV / m^3
        self.fugacity = (self.P * self.fugacity_coeff * units.J) / units.eV

        self.beta = 1 / (units.kB * self.T)  # Boltzmann weight, 1 / [eV/K * K]

        atm2pa = 101325
        mol2cm3 = units.kB / units.J * units.mol * 273.15 / atm2pa

        # Get the ideal supercell. This will be updated by the set_framework method
        self.ideal_supercell = self._get_ideal_supercell()

        self.conv_factors = {
            "nmol": {
                adsorbate.name: 1 / np.prod(self.ideal_supercell) for adsorbate in self.adsorbates
            },
            "mol/kg": {
                adsorbate.name: (1 / units.mol) / self.get_framework_mass()
                for adsorbate in self.adsorbates
            },
            "mg/g": {
                adsorbate.name: (self.adsorbate_mass[adsorbate.name] * 1e3)
                / self.get_framework_mass()
                for adsorbate in self.adsorbates
            },
            "cm^3 STP/gr": {
                adsorbate.name: mol2cm3 / units.mol / self.get_framework_mass() * 1e3
                for adsorbate in self.adsorbates
            },
            "cm^3 STP/cm^3": {
                adsorbate.name: 1e6
                * mol2cm3
                / units.mol
                / (self.framework.get_volume() * (1e-8**3))
                for adsorbate in self.adsorbates
            },
            "% wt": {
                adsorbate.name: (
                    self.adsorbate_mass[adsorbate.name] / self.get_framework_mass() * 100
                )
                for adsorbate in self.adsorbates
            },
        }

        self.vdw: np.ndarray = (
            np.array(vdw_radii) * vdw_factor
        )  # Adjust van der Waals radii to avoid overlap

        # Replace any NaN value by 1.5 on self.vdw to avoid potential problems
        self.vdw[np.isnan(self.vdw)] = 1.5

        # Use an max deltaE to avoid errors on the model
        self.max_deltaE = max_deltaE

    def _save_trajectory(self, actual_iteration: int) -> None:
        """
        Save the trajectory of the simulation to a trajectory file if
        the current iteration is a multiple of the save frequency.

        Parameters
                ----------
                actual_iteration : int
                    The current iteration number of the simulation.
        """

        if actual_iteration % self.save_every == 0:
            # Workaround to save the labels in Trajectory.info since ASE's Trajectory does not support custom arrays
            if "labels" in self.current_system.arrays.keys():
                self.current_system.info["labels"] = self.current_system.get_array("labels")

            self.trajectory.write(self.current_system)  # type: ignore

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

    def _get_ideal_supercell(self) -> np.ndarray:
        """
        Get the ideal supercell dimensions based on the cutoff radius.

        Returns
        -------
        np.ndarray
            An array of three integers representing the number of unit cells in each dimension.
        """
        return calculate_unit_cells(self.framework.get_cell(), cutoff=self.cutoff)

    def get_framework_mass(self) -> float:
        """
        Calculate the mass of the framework in kg.

        Returns
        -------
        float
            The mass of the framework in kg.
        """
        return float(np.sum(self.framework.get_masses()) / units.kg)

    def set_framework(
        self, framework_atoms: ase.Atoms, framework_energy: float | None = None
    ) -> None:
        """
        Set the framework structure for the simulation.

        Parameters
        ----------
        framework_atoms : ase.Atoms
            The new framework structure as an ASE Atoms object.
        framework_energy : float or None, optional
            The energy of the framework in eV. If None, the energy will be calculated using the provided model.
        """
        self.framework = framework_atoms
        self.framework.set_tags(np.zeros(len(self.framework), dtype=int))

        self.ideal_supercell = self._get_ideal_supercell()

        if (
            not np.array_equal(self.ideal_supercell, np.array([1, 1, 1]))
            and self.automatic_supercell
        ):
            self.framework = make_supercell(self.framework, np.eye(3) * self.ideal_supercell)

        self.cell = np.array(self.framework.get_cell())
        self.perpendicular_cell = get_perpendicular_lengths(self.framework.get_cell()) * np.eye(3)

        self.framework.calc = self.model
        if framework_energy:
            self.framework_energy = framework_energy * np.prod(self.ideal_supercell)
        else:
            self.framework_energy = self.framework.get_potential_energy()
        self.n_atoms_framework = len(self.framework)

        self.V = np.linalg.det(self.cell) / units.m**3
        self.framework_mass = self.get_framework_mass()

        # Get the framework density in g/cm^3
        self.framework_density = get_density(self.framework)

    def set_adsorbates(
        self,
        adsorbates: Adsorbate | list[Adsorbate],
        adsorbate_energy: float | list | dict[str, float] | None = None,
    ) -> None:
        """
        Set the adsorbate structure for the simulation.

        Parameters
        ----------
        adsorbates : Adsorbate
            The list of adsorbate structure(s) as an Adsorbate object.
        adsorbate_energy : float or None, optional
            The energy of the adsorbate in eV. If None, the energy will be calculated using the provided model.
        n_adsorbates : int
            Number of adsorbate molecules in the framework.
        """
        if not isinstance(adsorbates, (Adsorbate, list)):
            raise ValueError(
                "Adsorbates must be an Adsorbate object or a list of Adsorbate objects. Not ",
                type(adsorbates),
            )

        if isinstance(adsorbates, Adsorbate):
            adsorbates = [adsorbates]

        self.adsorbates = adsorbates

        # Check if mol fractions sum to 1
        self.adsorbate_mol_fractions = {
            adsorbate.name: adsorbate.mol_fraction for adsorbate in self.adsorbates
        }
        if not np.isclose(sum(self.adsorbate_mol_fractions.values()), 1.0):
            raise ValueError(
                f"Mole fractions must sum to 1. Current sum: {sum(self.adsorbate_mol_fractions.values())}"
            )

        # Set tags and properties for each adsorbate
        for i, adsorbate in enumerate(self.adsorbates):
            adsorbate.tag = i + 1
            adsorbate.structure.set_cell(self.framework.get_cell())  # type: ignore
            adsorbate.structure.set_pbc([True, True, True])  # type: ignore
            adsorbate.structure.calc = self.model  # type: ignore

        if isinstance(adsorbate_energy, float):
            if len(self.adsorbates) != 1:
                raise ValueError(
                    "If multiple adsorbates are provided, adsorbate_energy must be a list or dict."
                )

            self.adsorbate_energy = {self.adsorbates[0].name: adsorbate_energy}

        elif isinstance(adsorbate_energy, list):
            if len(adsorbate_energy) != len(self.adsorbates):
                raise ValueError(
                    "Length of adsorbate_energy list must match the number of adsorbates."
                )
            self.adsorbate_energy = {
                adsorbate.name: energy
                for adsorbate, energy in zip(self.adsorbates, adsorbate_energy)
            }

        elif isinstance(adsorbate_energy, dict):
            if set(adsorbate_energy.keys()) != set(adsorbate.name for adsorbate in self.adsorbates):
                raise ValueError(
                    "Keys of adsorbate_energy dict must match the names of the adsorbates."
                )
            self.adsorbate_energy = adsorbate_energy

        else:
            self.adsorbate_energy = {
                adsorbate.name: adsorbate.structure.get_potential_energy()  # type: ignore
                for adsorbate in self.adsorbates
            }

        self.n_adsorbate_atoms = {
            adsorbate.name: len(adsorbate.structure) for adsorbate in self.adsorbates
        }
        self.adsorbate_mass = {
            adsorbate.name: adsorbate.get_molar_mass() / units.kg for adsorbate in self.adsorbates
        }

        # self.n_atoms_framework -= self.n_adsorbate_atoms * n_adsorbates
        # self.n_adsorbates = n_adsorbates

    def set_state(self, state: ase.Atoms) -> None:
        """
        Set the current state of the simulation.

        Parameters
        ----------
        state : ase.Atoms
            The current state of the simulation as an ASE Atoms object.
        """
        self.current_system = state.copy()
        self.current_system.calc = self.model
        self.current_total_energy = self.current_system.get_potential_energy()

    def optimize_framework(
        self,
        max_steps: int = 1000,
        opt_cell: bool = True,
        fix_symmetry: bool = True,
        hydrostatic_strain: bool = True,
        symm_tol: float = 1e-3,
        max_force: float = 0.05,
    ) -> None:
        """
        Optimize the framework structure using the provided calculator.

        Parameters
        ----------
        max_steps : int, optional
            Maximum number of optimization steps (default is 1000).
        tol : float, optional
            Tolerance for convergence (default is 1e-5).

        Returns
        -------
        ase.Atoms
            The optimized framework structure.
        """

        print(
            """
=======================================================================================================
Start optimizing framework structure...
=======================================================================================================
              """,
            file=self.out_file,
            flush=True,
        )

        resultsDict, optFramework = crystalOptimization(
            atoms_in=self.framework,
            calculator=self.model,
            optimizer=LBFGS,  # type: ignore
            fmax=max_force,
            opt_cell=opt_cell,
            fix_symmetry=fix_symmetry,
            hydrostatic_strain=hydrostatic_strain,
            constant_volume=False,
            scalar_pressure=0,
            max_steps=max_steps,
            trajectory=self.trajectory,  # type: ignore
            verbose=True,
            symm_tol=symm_tol,
            out_file=self.out_file,  # type: ignore
        )

        # Remove any constraints from the optimized framework
        optFramework.set_constraint(None)

        self.set_framework(optFramework.copy())

    def optimize_adsorbate(self, max_steps: int = 1000, max_force: float = 0.05) -> None:
        """
        Optimize the adsorbate structure using the provided calculator.

        Parameters
        ----------
        max_steps : int, optional
            Maximum number of optimization steps (default is 1000).
        symm_tol : float, optional
            Tolerance for symmetry (default is 1e-3).
        max_force : float, optional
            Maximum force tolerance for convergence (default is 0.05 eV/Å).

        Returns
        -------
        ase.Atoms
            The optimized adsorbate structure.
        """

        print(
            """
=======================================================================================================
Start optimizing adsorbate structure...
=======================================================================================================
              """,
            file=self.out_file,
            flush=True,
        )

        for i in range(len(self.adsorbates)):

            resultsDict, optAdsorbate = crystalOptimization(
                atoms_in=self.adsorbates[i].structure,  # type: ignore
                calculator=self.model,
                optimizer=LBFGS,  # type: ignore
                fmax=max_force,
                opt_cell=False,
                fix_symmetry=False,
                hydrostatic_strain=True,
                constant_volume=True,
                scalar_pressure=self.P,
                max_steps=max_steps,
                trajectory=f"Adsorbate_{self.adsorbates[i].name}_Optimization.traj",
                verbose=True,
                symm_tol=1e3,
                out_file=self.out_file,  # type: ignore
            )

            self.adsorbates[i].structure = optAdsorbate.copy()
            self.adsorbates[i].structure.set_constraint(None)  # type: ignore
            self.adsorbates[i].structure.set_cell(self.framework.get_cell())  # type: ignore
            self.adsorbates[i].structure.calc = self.model  # type: ignore
            self.adsorbate_energy = {
                self.adsorbates[i].name: self.adsorbates[i].structure.get_potential_energy()  # type: ignore
            }

    def md(
        self,
        nsteps,
        atoms: ase.Atoms | None = None,
        time_step: float = 0.5,
        ensemble: str = "NVT",
        thermostat: str = "NoseHoover",
        output_interval: int = 100,
        movie_interval: int = 100,
        calculator: calculator.Calculator | None = None,
        update_state: bool = True,
        trajectory_file: TrajectoryReader | TrajectoryWriter | None = None,
        set_momenta: bool = True,
        **kwargs,
    ) -> None | ase.Atoms:
        """
        Run a molecular dynamics simulation using the specified ensemble and thermostat.

        Parameters
        ----------
        nsteps : int
            Number of steps to run the MD simulation.
        time_step : float, optional
            Time step for the MD simulation in femtoseconds (default is 0.5 fs).
        ensemble : str, optional
            The ensemble to use for the MD simulation (default is "NVT").
            Can be one of "NVT" or "NPT".
        thermostat : str, optional
            The thermostat to use for the MD simulation (default is "NoseHoover").
            For NVT, can be one of "Berendsen", "NoseHoover", or "Langevin".
            For NPT, can be one of "Berendsen", "NoseHoover", or "MTKNPT".
        output_interval : int, optional
            The interval for logging output (default is 100 steps).
        movie_interval : int, optional
            The interval for saving trajectory frames (default is 100 steps).
        calculator : ase.calculators.calculator.Calculator or None, optional
            The calculator to use for energy calculations. If None, the default model will be used.
        update_state : bool, optional
            If True, updates the current state of the simulation with the final state after MD.
            If False, returns the final state without updating. This difference is important for using
            the same method in the GCMC simulations when a MD step is used as a move. Default is True.
        trajectory_file : str or None, optional
            If provided, saves the trajectory to the specified file instead of the default trajectory.
        set_momenta : bool, optional
            Whether to set the atomic momenta to a Maxwell-Boltzmann distribution of the simulation temperature.
        **kwargs : optional
            Additional parameters passed directly to the specific MD thermostat (e.g., taut, tdamp, friction).

            NVT Berendsen:
                - taut : float, optional
                    Time constant for the Berendsen thermostat in fs (default is 1.0 fs).

            NVT Nose-Hoover:
                - tdamp : float, optional
                    Time constant for the Nose-Hoover thermostat in fs (default is 50.0 fs).
                - tchain : int, optional
                    Number of thermostats in the Nose-Hoover chain (default is 3).
                - tloop : int, optional
                    Number of loops for the Nose-Hoover chain (default is 1).

            NVT Langevin:
                - friction : float, optional
                    Friction coefficient for the Langevin dynamics (default is 0.01).

            NPT Berendsen:
                - isotropic : bool, optional
                    Whether to use isotropic pressure coupling (default is True).
                - compressibility : float, optional
                    Compressibility of the material in bar^-1 (default is 1e-4 bar^-1).
                - taut : float, optional
                    Time constant for the Berendsen thermostat in fs (default is 10.0 fs).
                - taup : float, optional
                    Time constant for the Berendsen barostat in fs (default is 500.0 fs).

            NPT Nose-Hoover:
                - ttime : float, optional
                    Time constant for the Nose-Hoover thermostat in fs (default is 25.0 fs).
                - ptime : float, optional
                    Time constant for the Parrinello-Rahman barostat in fs (default is 75.0 fs).
                - bulk_modulus : float, optional
                    Bulk modulus of the material in GPa (default is 30.0 GPa).

            NPT MTK:
                - tdamp : float, optional
                    Time constant for the Nose-Hoover thermostat in fs (default is 50.0 fs).
                - pdamp : float, optional
                    Time constant for the Nose-Hoover barostat in fs (default is 500.0 fs).
                - tchain : int, optional
                    Number of thermostats in the Nose-Hoover chain (default is 3).
                - pchain : int, optional
                    Number of barostats in the Nose-Hoover chain (default is 3).
                - tloop : int, optional
                    Number of loops for the Nose-Hoover thermostat chain (default is 1).
                - ploop : int, optional
                    Number of loops for the Nose-Hoover barostat chain (default is 1).
                - vol_constraint : bool, optional
                    If True, the (N, V, sigma_a = 0, T)-ensemble is sampled, which allows for full
                    cell fluctuations while keeping the cell volume fixed (default is False).
        """

        if atoms is not None:
            current_state = atoms.copy()
        else:
            current_state = self.current_system.copy()

        new_state = run_md_simulation(
            atoms=current_state,
            model=calculator if calculator else self.model,
            temperature=self.T,
            pressure=self.P * 1e-5,
            time_step=time_step,
            num_md_steps=nsteps,
            ensemble=ensemble,
            thermostat=thermostat,
            out_folder=self.out_folder,
            out_file=self.out_file,  # type: ignore
            output_interval=output_interval,
            movie_interval=movie_interval,
            trajectory_file=trajectory_file if trajectory_file is not None else self.trajectory,
            set_momenta=set_momenta,
            **kwargs,
        )

        if not update_state:
            return new_state

        self.set_state(new_state)
        self.set_framework(new_state[: self.n_atoms_framework].copy())  # type: ignore

    def npt(
        self,
        nsteps,
        time_step: float = 0.5,
        mode: str = "iso_shape",
        driver: str = "MTKNPT",
        set_momenta: bool = True,
        output_interval: int = 100,
        movie_interval: int = 100,
        calculator: calculator.Calculator | None = None,
        **kwargs,
    ) -> None | ase.Atoms:
        """
        Run a NPT simulation using the Berendsen thermostat and barostat.

        Parameters
        ----------
        nsteps : int
            Number of steps to run the NPT simulation.
        time_step : float, optional
            Time step for the NPT simulation (default is 0.5 fs).
        mode : str, optional
            The mode of the NPT simulation (default is "iso_shape").
            Can be one of "iso_shape", "aniso_shape", or "aniso_flex".
        driver : str, optional
            The driver to use for the NPT simulation.
            Can be Berendsen, NoseHoover or MTKNPT (default is "MTKNPT").
        set_momenta : bool, optional
            Whether to set the atomic momenta to a Maxwell-Boltzmann distribution of the simulation temperature.
        output_interval : int, optional
            The interval for logging output (default is 100 steps).
        movie_interval : int, optional
            The interval for saving trajectory frames (default is 100 step).
        calculator : ase.calculators.calculator.Calculator or None, optional
            The calculator to use for energy calculations. If None, the default model will be used.
        kwargs : optional
            Arguments passed to the ase molecular dynamics class.
        """

        allowed_modes = ["iso_shape", "aniso_shape", "aniso_flex"]
        assert mode in allowed_modes, f"Mode must be one of {allowed_modes}."

        if not calculator:
            calculator = self.model

        if (mode == "iso_shape" or mode == "aniso_shape") and driver == "Berendsen":

            self.logger._print_warning(
                "Berendsen barostat is not indicated to be used in GCMC simulations. "
                "It is implemented for testing purposes only. Use MTKNPT or NoseHoover instead."
            )

            new_state = nPT_Berendsen(
                atoms=self.current_system,
                model=calculator,
                temperature=self.T,
                pressure=self.P * 1e-5,
                time_step=time_step,
                num_md_steps=nsteps,
                isotropic=True if mode == "iso_shape" else False,
                out_folder=self.out_folder,
                out_file=self.out_file,  # type: ignore
                output_interval=output_interval,
                movie_interval=movie_interval,
                trajectory_file=self.trajectory,
                set_momenta=set_momenta,
                **kwargs,
            )
        elif driver == "NoseHoover":

            new_state = nPT_NoseHoover(
                atoms=self.current_system,
                model=calculator,
                temperature=self.T,
                pressure=self.P * 1e-5,
                time_step=time_step,
                num_md_steps=nsteps,
                out_folder=self.out_folder,
                out_file=self.out_file,  # type: ignore
                output_interval=output_interval,
                movie_interval=movie_interval,
                trajectory_file=self.trajectory,
                set_momenta=set_momenta,
                **kwargs,
            )

        elif driver == "MTKNPT":

            isotropic = True if (mode == "iso_flex" or mode == "iso_shape") else False

            new_state = nPT_MTKNPT(
                atoms=self.current_system,
                model=calculator,
                temperature=self.T,
                pressure=self.P * 1e-5,
                time_step=time_step,
                num_md_steps=nsteps,
                isotropic=isotropic,
                out_folder=self.out_folder,
                out_file=self.out_file,  # type: ignore
                output_interval=output_interval,
                movie_interval=movie_interval,
                trajectory_file=self.trajectory,
                set_momenta=set_momenta,
                **kwargs,
            )

        else:
            raise ValueError(
                f"Driver must be one of 'Berendsen', 'NoseHoover' or 'MTKNPT'. Not {driver}."
            )

        return new_state

    def nvt(
        self,
        nsteps,
        time_step: float = 0.5,
        set_momenta: bool = True,
        driver: str = "NoseHoover",
        output_interval: int = 100,
        movie_interval: int = 100,
        calculator: calculator.Calculator | None = None,
        **kwargs,
    ):
        """
        Run a NVT simulation using the Berendsen thermostat.

        Parameters
        ----------
        nsteps : int
            Number of steps to run the NVT simulation.
        time_step : float, optional
            Time step for the NVT simulation in femtoseconds (default is 0.5 fs).
        set_momenta : bool, optional
            Whether to set the atomic momenta to a Maxwell-Boltzmann distribution of the simulation temperature.
        driver : str, optional
            The driver to use for the NVT simulation. Can be "NoseHoover" (default), Langevin, or "Berendsen" (Not recommended, for testing purposes only).
        output_interval : int, optional
            The interval for logging output (default is 100 steps).
        movie_interval : int, optional
            The interval for saving trajectory frames (default is 100 step).
        calculator : ase.calculators.calculator.Calculator or None, optional
            The calculator to use for energy calculations. If None, the default model will be used.
        kwargs : optional
            Arguments passed to the ase molecular dynamics class.

        For NoseHoover:
        - tdamp : float, optional
            Time constant for the thermostat in fs (default is 50.0 fs).
        - tchain : int, optional
            Length of the Nose-Hoover chain (default is 3).
        - tloop : int, optional
            Number of loops for the Nose-Hoover chain (default is 1).

        For Langevin:
        - friction : float, optional
            Friction coefficient for the Langevin thermostat in fs^-1 (default is 0.1 fs^-1).

        For Berendsen:
        - tau : float, optional
            Time constant for the Berendsen thermostat in fs (default is 1.0 fs).
        """
        if not calculator:
            calculator = self.model

        assert driver in [
            "Berendsen",
            "Langevin",
            "NoseHoover",
        ], "Driver must be one of 'Berendsen', 'Langevin', or 'NoseHoover'."

        if driver == "NoseHoover":

            new_state = nVT_NoseHoover(
                atoms=self.current_system,
                model=calculator,
                temperature=self.T,
                time_step=time_step,
                num_md_steps=nsteps,
                out_folder=self.out_folder,
                out_file=self.out_file,  # type: ignore
                output_interval=output_interval,
                movie_interval=movie_interval,
                mc_trajectory=self.trajectory,
                set_momenta=set_momenta,
                **kwargs,
            )

            self.set_state(new_state)

        if driver == "Langevin":

            new_state = nVT_Langevin(
                atoms=self.current_system,
                model=calculator,
                temperature=self.T,
                time_step=time_step,
                num_md_steps=nsteps,
                out_folder=self.out_folder,
                out_file=self.out_file,  # type: ignore
                output_interval=output_interval,
                movie_interval=movie_interval,
                mc_trajectory=self.trajectory,
                set_momenta=set_momenta,
                **kwargs,
            )

            self.set_state(new_state)

        if driver == "Berendsen":
            self.logger._print_warning(
                "Berendsen barostat is not indicated to be used in GCMC simulations. "
                "It is implemented for testing purposes only. Use NoseHoover or Langevin instead."
            )

            new_state = nVT_Berendsen(
                atoms=self.current_system,
                model=calculator,
                temperature=self.T,
                time_step=time_step,
                num_md_steps=nsteps,
                out_folder=self.out_folder,
                out_file=self.out_file,  # type: ignore
                output_interval=output_interval,
                movie_interval=movie_interval,
                mc_trajectory=self.trajectory,
                set_momenta=set_momenta,
                **kwargs,
            )

            self.set_state(new_state)
