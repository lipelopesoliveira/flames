import datetime
import os
import sys
from typing import Any, Dict, Optional, TextIO, Type

import ase
import ase.units
import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator
from ase.io.trajectory import Trajectory, TrajectoryReader, TrajectoryWriter
from ase.md import MDLogger
from ase.md.langevin import Langevin
from ase.md.md import MolecularDynamics
from ase.md.melchionna import MelchionnaNPT
from ase.md.nose_hoover_chain import (
    IsotropicMTKNPT,
    MTKBarostat,
    NoseHooverChainNVT,
    NoseHooverChainThermostat,
)
from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen, NPTBerendsen
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from scipy.special import exprel


def run_md_simulation(
    atoms: Atoms,
    model: Calculator,
    ensemble: str,
    thermostat: str,
    temperature: float,
    pressure: float = 0.0,
    time_step: float = 0.5,
    num_md_steps: int = 1000000,
    output_interval: int = 100,
    movie_interval: int = 1,
    out_folder: str = ".",
    out_file: TextIO = sys.stdout,
    mc_trajectory: TrajectoryWriter | TrajectoryReader | None = None,
    set_momenta: bool = True,
    **kwargs,
) -> Atoms:
    """
    Wrapper for setting up and routing NVT and NPT molecular dynamics simulations.

    Selects the correct ensemble and driver, handles parameter defaults, applies
    unit conversions, and delegates execution to the core MD engine.

    Parameters
    ----------
    atoms : ase.Atoms
        The atomic structure to simulate.
    model : Calculator
        The ASE calculator to attach to the atoms.
    ensemble : str
        The ensemble to use: "NVT" or "NPT".
    thermostat : str
        The thermostat/barostat driver to use:
        For NVT:
          - "Berendsen", "NoseHoover", or "Langevin".
        For NPT:
          - "Berendsen", "NoseHoover", or "MTK".
    temperature : float
        Target temperature in Kelvin.
    pressure : float, optional
        Target pressure in bar (1 bar = 1e5 Pa). Default is 0.0.
    time_step : float, optional
        Simulation time step in femtoseconds. Default is 0.5 fs.
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
    ensemble = ensemble.upper()
    thermostat = thermostat.lower()

    assert ensemble in ["NVT", "NPT"], f"Unsupported ensemble: {ensemble}. Must be 'NVT' or 'NPT'."

    assert thermostat in [
        "berendsen",
        "nosehoover",
        "langevin",
        "mtk",
    ], f"Unsupported thermostat: {thermostat}."

    # Base dynamics parameters shared across all simulations
    dyn_params: Dict[str, Any] = {
        "atoms": atoms,
        "timestep": time_step * units.fs,
        "temperature_K": temperature,
        "loginterval": movie_interval,
        "append_trajectory": True,
    }

    # 1. Parameter routing and unit conversions based on Ensemble and Thermostat
    if ensemble == "NVT":
        if thermostat == "berendsen":
            dyn_class = NVTBerendsen
            dyn_params["taut"] = kwargs.pop("taut", 1.0) * units.fs
        elif thermostat == "nosehoover":
            dyn_class = NoseHooverChainNVT
            dyn_params["tdamp"] = kwargs.pop("tdamp", 50.0) * units.fs
            dyn_params["tchain"] = kwargs.pop("tchain", 3)
            dyn_params["tloop"] = kwargs.pop("tloop", 1)
        elif thermostat == "langevin":
            dyn_class = Langevin
            dyn_params["friction"] = kwargs.pop("friction", 0.01)
        else:
            raise ValueError(f"Unsupported NVT thermostat: {thermostat}")

    elif ensemble == "NPT":
        isotropic = kwargs.pop("isotropic", False)
        vol_constraint = kwargs.pop("vol_constraint", False)

        if thermostat == "berendsen":
            dyn_class = NPTBerendsen if isotropic else Inhomogeneous_NPTBerendsen
            dyn_params["pressure_au"] = pressure * units.bar
            dyn_params["compressibility_au"] = kwargs.pop("compressibility", 1e-4) / units.bar
            dyn_params["taut"] = kwargs.pop("taut", 10.0) * units.fs
            dyn_params["taup"] = kwargs.pop("taup", 500.0) * units.fs

        elif thermostat == "nosehoover":
            dyn_class = MelchionnaNPT
            dyn_params["externalstress"] = pressure * units.bar
            dyn_params["ttime"] = kwargs.pop("ttime", 25.0) * units.fs
            ptime = kwargs.pop("ptime", 75.0)
            bulk_modulus = kwargs.pop("bulk_modulus", 30.0)
            dyn_params["pfactor"] = (ptime * units.fs) ** 2 * bulk_modulus * units.GPa

        elif thermostat == "mtk":
            if isotropic and vol_constraint:
                raise ValueError(
                    "The combination of isotropic=True and vol_constraint=True is not supported for MTK."
                )
            if isotropic and not vol_constraint:
                dyn_class = IsotropicMTKNPT
            else:
                dyn_class = MTKNPT

            dyn_params["pressure_au"] = pressure * units.bar
            dyn_params["tdamp"] = kwargs.pop("tdamp", 50.0) * units.fs
            dyn_params["pdamp"] = kwargs.pop("pdamp", 500.0) * units.fs
            dyn_params["tchain"] = kwargs.pop("tchain", 3)
            dyn_params["pchain"] = kwargs.pop("pchain", 3)
            dyn_params["tloop"] = kwargs.pop("tloop", 1)
            dyn_params["ploop"] = kwargs.pop("ploop", 1)
            dyn_params["vol_constraint"] = vol_constraint
        else:
            raise ValueError(f"Unsupported NPT thermostat: {thermostat}")
    else:
        raise ValueError(f"Unsupported ensemble: {ensemble}")

    # Inject any remaining unpopped kwargs (e.g. trajectory paths)
    dyn_params.update(kwargs)

    # 2. Formulate proper labeling for outputs
    method_label = f"{ensemble}-{thermostat.capitalize()}"

    # 3. Delegate to the core execution engine
    return _md_core(
        atoms=atoms,
        model=model,
        dyn_class=dyn_class,
        dyn_params=dyn_params,
        ensemble=ensemble,
        method_label=method_label,
        temperature=temperature,
        num_md_steps=num_md_steps,
        output_interval=output_interval,
        out_folder=out_folder,
        out_file=out_file,
        set_momenta=set_momenta,
        mc_trajectory=mc_trajectory,
    )


def _md_core(
    atoms: Atoms,
    model: Calculator,
    dyn_class: Type[MolecularDynamics],
    dyn_params: Dict[str, Any],
    ensemble: str,
    method_label: str,
    temperature: float,
    num_md_steps: int,
    output_interval: int,
    out_folder: str,
    out_file: TextIO,
    set_momenta: bool,
    mc_trajectory: Optional[Trajectory],  # type: ignore
) -> Atoms:
    """
    Internal engine that runs the ASE molecular dynamics process,
    managing file paths, loggers, momenta initialization, and terminal outputs.
    """
    atoms.calc = model

    # Determine next available trajectory index
    existing_md_traj = [
        f for f in os.listdir(out_folder) if f.startswith(method_label) and f.endswith(".traj")
    ]
    run_idx = len(existing_md_traj)

    traj_filename = os.path.join(out_folder, f"{method_label}_{temperature:.2f}K_{run_idx}.traj")
    log_filename = os.path.join(out_folder, f"{method_label}_{temperature:.2f}K_{run_idx}.log")

    # Hook up trajectories
    if "trajectory" not in dyn_params:
        traj_file = Trajectory(filename=traj_filename, mode="a", atoms=atoms)
        dyn_params["trajectory"] = mc_trajectory if mc_trajectory else traj_file

    # Set initial momenta
    if set_momenta:
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, force_temp=True)
        Stationary(atoms)

    # Initialize dynamics driver
    dyn = dyn_class(**dyn_params)
    start_time = datetime.datetime.now()

    # Generic Headers
    header = f"""
======================================================================================
    Starting {ensemble} MD Simulation using {method_label}

    General Parameters:
        Temperature: {temperature:.2f} K
        Time Step: {dyn_params['timestep'] / units.fs:.2f} fs
        Number of MD Steps: {num_md_steps}
        Output Interval: {output_interval} steps
        Movie Interval: {dyn_params['loginterval']} steps

    Method-Specific Parameters:
        Driver: {dyn_class.__name__}
"""
    for key, value in dyn_params.items():
        if key not in [
            "atoms",
            "trajectory",
            "timestep",
            "loginterval",
            "temperature_K",
            "pressure_au",
        ]:
            header += f"        {key}: {value}\n"

    print(header, file=out_file, flush=True)

    print(
        "    Step   |  Pot. Energy   |  Total Energy  |  Temperature  |  Stress  |   Volume    | Elapsed Time ",
        file=out_file,
        flush=True,
    )
    print(
        "    [-]    |      [eV]      |      [eV]      |      [K]      |   [GPa]  |    [A^3]    |      [s]      ",
        file=out_file,
        flush=True,
    )
    print(
        " --------- | -------------- | -------------- | ------------- | -------- | ----------- | -------------",
        file=out_file,
        flush=True,
    )

    # Custom Terminal Logger
    def print_md_log():
        step = dyn.get_number_of_steps()
        epot = atoms.get_potential_energy()
        etot = atoms.get_total_energy()
        temp_K = atoms.get_temperature()
        stress = atoms.get_stress(include_ideal_gas=True) / units.GPa
        stress_ave = sum(stress[:3]) / 3.0
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        vol = atoms.get_volume()

        print(
            f"  {step:>7}  | {epot:13.6f}  | {etot:13.6f}  |  {temp_K:11.3f}  |  {stress_ave:7.2f} | {vol:11.2f} | {elapsed_time:9.1f}",
            file=out_file,
            flush=True,
        )

    # Attach Loggers & Run
    dyn.attach(print_md_log, interval=output_interval)
    dyn.attach(
        MDLogger(dyn, atoms, log_filename, header=True, stress=True, peratom=False, mode="a"),
        interval=output_interval,
    )

    dyn.run(num_md_steps)

    # Footer
    footer = f"""
======================================================================================
    {ensemble} MD simulation completed at {datetime.datetime.now()}
    Log file saved to: {log_filename}
    Total simulation time: {(datetime.datetime.now() - start_time).total_seconds():.2f} seconds
======================================================================================
    """
    print(footer, file=out_file, flush=True)

    return atoms


class MTKNPT(MolecularDynamics):
    """Isothermal-isobaric molecular dynamics with volume-and-cell fluctuations
    by Martyna-Tobias-Klein (MTK) method [1].

    See also :class:`NoseHooverChainNVT` for the references.
    The factorization of the Liouville operator is the same as Reference [1].

    - [1] G. J. Martyna, D. J. Tobias, and M. L. Klein, J. Chem. Phys. 101,
          4177-4189 (1994). https://doi.org/10.1063/1.467468
    """

    def __init__(
        self,
        atoms: Atoms,
        timestep: float,
        temperature_K: float,
        pressure_au: float,
        tdamp: float,
        pdamp: float,
        tchain: int = 3,
        pchain: int = 3,
        tloop: int = 1,
        ploop: int = 1,
        vol_constraint: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        atoms: ase.Atoms
            The atoms object.
        timestep: float
            The time step in ASE time units.
        temperature_K: float
            The target temperature in K.
        pressure_au: float
            The external pressure in eV/Ang^3.
        tdamp: float
            The characteristic time scale for the thermostat in ASE time units.
            Typically, it is set to 100 times of `timestep`.
        pdamp: float
            The characteristic time scale for the barostat in ASE time units.
            Typically, it is set to 1000 times of `timestep`.
        tchain: int
            The number of thermostat variables in the Nose-Hoover thermostat.
        pchain: int
            The number of barostat variables in the MTK barostat.
        tloop: int
            The number of sub-steps in thermostat integration.
        ploop: int
            The number of sub-steps in barostat integration.
        vol_constraint: bool
            If True, the (N, V, sigma_a = 0, T)-ensemble is sampled, which allows for full cell fluctuations
            while keeping the cell volume fixed.
            This ensemble was introduced in [doi:10.1021/acs.jctc.5b00748].
        **kwargs : dict, optional
            Additional arguments passed to :class:~ase.md.md.MolecularDynamics
            base class.
        """
        super().__init__(
            atoms=atoms,
            timestep=timestep,
            **kwargs,
        )
        assert self.masses.shape == (len(self.atoms), 1)

        self.vol_constraint = vol_constraint

        if len(atoms.constraints) > 0:
            raise NotImplementedError("Current implementation does not support constraints")

        self._num_atoms_global = self.atoms.get_global_number_of_atoms()
        self._thermostat = NoseHooverChainThermostat(
            num_atoms_global=self._num_atoms_global,
            masses=self.masses,
            temperature_K=temperature_K,
            tdamp=tdamp,
            tchain=tchain,
            tloop=tloop,
        )

        self._barostat = MTKBarostat(
            num_atoms_global=self._num_atoms_global,
            temperature_K=temperature_K,
            pdamp=pdamp,
            pchain=pchain,
            ploop=ploop,
        )

        self._temperature_K = temperature_K
        self._pressure_au = pressure_au

        self._kT = ase.units.kB * self._temperature_K

        # The following variables are updated during self.step()
        self._q = self.atoms.get_positions()  # positions
        self._p = self.atoms.get_momenta()  # momenta
        self._h = np.array(self.atoms.get_cell())  # cell

        self._init_cell_momenta()

    @property
    def mask(self) -> tuple[bool, bool, bool] | None:
        return None

    @mask.setter
    def mask(self, mask: tuple[bool, bool, bool]) -> None:
        raise AttributeError()

    def _init_cell_momenta(self) -> None:
        self._p_g = np.zeros((3, 3))  # cell momenta

    def step(self) -> None:
        dt2 = self.dt / 2

        self._integrate_p_cell_by_barostat(dt2)
        self._p = self._thermostat.integrate_nhc(self._p, dt2)
        self._integrate_p_cell(dt2)
        self._integrate_p(dt2)
        self._integrate_q(self.dt)
        self._integrate_q_cell(self.dt)
        self._integrate_p(dt2)
        self._integrate_p_cell(dt2)
        self._p = self._thermostat.integrate_nhc(self._p, dt2)
        self._integrate_p_cell_by_barostat(dt2)

        self._update_atoms()

    def get_conserved_energy(self) -> float:
        conserved_energy = (
            self.atoms.get_total_energy()
            + self._thermostat.get_thermostat_energy()
            + self._barostat.get_barostat_energy()
            + self._get_cell_kinetic_energy()
            + self._pressure_au * self._get_volume()
        )
        return float(conserved_energy)

    def _update_atoms(self) -> None:
        self.atoms.set_positions(self._q)
        self.atoms.set_momenta(self._p)
        self.atoms.set_cell(self._h, scale_atoms=False)

    def _get_volume(self) -> float:
        return np.abs(np.linalg.det(self._h))

    def _get_forces(self) -> np.ndarray:
        self._update_atoms()
        return self.atoms.get_forces(md=True)

    def _get_stress(self) -> np.ndarray:
        self._update_atoms()
        stress = self.atoms.get_stress(voigt=False, include_ideal_gas=True)
        return -stress

    def _get_cell_kinetic_energy(self) -> float:
        return float(np.sum(self._p_g**2) / (2 * self._barostat.W))

    def _integrate_q(self, delta: float) -> None:
        """Integrate exp(i * L_1 * delta)"""
        # eigvals: (3-eigvec), U: (3-xyz, 3-eigvec)
        eigvals, U = np.linalg.eigh(self._p_g)
        x = self._q @ U  # (num_atoms, 3-eigvec)
        y = self._p @ U  # (num_atoms, 3-eigvec)
        sol = (
            x * np.exp(eigvals * delta / self._barostat.W)[None, :]
            + delta * y / self.masses * exprel(eigvals * delta / self._barostat.W)[None, :]
        )  # (num_atoms, 3-eigvec)
        self._q = sol @ U.T

    def _integrate_p(self, delta: float) -> None:
        """Integrate exp(i * L_2 * delta)"""
        forces = self._get_forces()  # (num_atoms, 3-xyz)

        # eigvals: (3-eigvec), U: (3-xyz, 3-eigvec)
        eigvals, U = np.linalg.eigh(self._p_g)
        kappas = eigvals + np.trace(self._p_g) / (3 * self._num_atoms_global)  # (3-eigvec)
        y = self._p @ U  # (num_atoms, 3-eigvec)
        sol = (
            y * np.exp(-kappas * delta / self._barostat.W)[None, :]
            + delta * (forces @ U) * exprel(-kappas * delta / self._barostat.W)[None, :]
        )  # (num_atoms, 3-eigvec)
        self._p = sol @ U.T

    def _integrate_q_cell(self, delta: float) -> None:
        """Integrate exp(i * L_(g, 1) * delta)"""
        # U @ np.diag(eigvals) @ U.T = self._p_g
        # eigvals: (3-eigvec), U: (3-xyz, 3-eigvec)
        eigvals, U = np.linalg.eigh(self._p_g)
        n = self._h @ U  # (3-axis, 3-eigvec)
        sol = n * np.exp(eigvals * delta / self._barostat.W)[None, :]  # (3-axis, 3-eigvec)
        self._h = sol @ U.T

    def _integrate_p_cell(self, delta: float) -> None:
        """Integrate exp(i * L_(g, 2) * delta)"""
        stress = self._get_stress()
        volume = self._get_volume()
        particle_dof = 3 * self._num_atoms_global
        kinetic_term = np.sum(self._p**2 / self.masses) / particle_dof
        pv_tensor = volume * (stress - self._pressure_au * np.eye(3))
        G = pv_tensor + kinetic_term * np.eye(3)

        # A traceless constraint is applied to the cell momenta to ensure that the cell volume remains constant.
        # Rogge, S. M. J. et al. Theory Comput. 11, 5583–5597 (2015) DOI: 10.1021/acs.jctc.5b00748
        if self.vol_constraint:
            G -= np.trace(G) / 3.0 * np.eye(3)

        self._p_g += delta * G

    def _integrate_p_cell_by_barostat(self, delta: float) -> None:
        self._p_g = self._barostat.integrate_nhc_baro(self._p_g, delta)
