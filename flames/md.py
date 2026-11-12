from __future__ import annotations

import datetime
import os
import sys
from typing import Any, TextIO

from ase import Atoms, units
from ase.calculators.calculator import Calculator
from ase.io.trajectory import Trajectory, TrajectoryReader, TrajectoryWriter
from ase.md import MDLogger
from ase.md.langevin import Langevin
from ase.md.md import MolecularDynamics
from ase.md.melchionna import MelchionnaNPT
from ase.md.nose_hoover_chain import (
    IsotropicMTKNPT,
    NoseHooverChainNVT,
)
from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen, NPTBerendsen
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet

from flames.mtknpt import MTKNPT


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
    trajectory_file: TrajectoryWriter | TrajectoryReader | None = None,
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

        NVE Velocity Verlet:
            - No additional parameters.

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

    assert ensemble in [
        "NVE",
        "NVT",
        "NPT",
    ], f"Unsupported ensemble: {ensemble}. Must be 'NVE', 'NVT' or 'NPT'."

    assert thermostat in [
        "velocityverlet",
        "berendsen",
        "nosehoover",
        "langevin",
        "mtk",
    ], f"Unsupported thermostat: {thermostat}."

    # Base dynamics parameters shared across all simulations
    dyn_params: dict[str, Any] = {
        "atoms": atoms,
        "timestep": time_step * units.fs,
        "loginterval": movie_interval,
        "append_trajectory": True,
    }

    # 1. Parameter routing and unit conversions based on Ensemble and Thermostat
    if ensemble == "NVE":
        if thermostat != "velocityverlet":
            raise ValueError("For NVE ensemble, only 'velocityverlet' thermostat is supported.")
        dyn_class = VelocityVerlet

    elif ensemble == "NVT":
        dyn_params["temperature_K"] = temperature
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
        dyn_params["temperature_K"] = temperature
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
        trajectory_file=trajectory_file,
    )


def _md_core(
    atoms: Atoms,
    model: Calculator,
    dyn_class: type[MolecularDynamics],
    dyn_params: dict[str, Any],
    ensemble: str,
    method_label: str,
    temperature: float,
    num_md_steps: int,
    output_interval: int,
    out_folder: str,
    out_file: TextIO,
    set_momenta: bool,
    trajectory_file: Trajectory | None = None,  # type: ignore
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
        dyn_params["trajectory"] = (
            trajectory_file
            if trajectory_file is not None
            else Trajectory(filename=traj_filename, mode="a")
        )

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
    header += f"        Trajectory File: {dyn_params['trajectory'].filename}\n"

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
        "    Step   |  Pot. Energy   |  Kin. Energy   |  Total Energy  |  Temperature  |  Stress  |   Volume    | Elapsed Time ",
        file=out_file,
        flush=True,
    )
    print(
        "    [-]    |      [eV]      |      [eV]      |      [eV]      |      [K]      |   [GPa]  |    [A^3]    |      [s]      ",
        file=out_file,
        flush=True,
    )
    print(
        " --------- | -------------- | -------------- | -------------- | ------------- | -------- | ----------- | -------------",
        file=out_file,
        flush=True,
    )

    # Custom Terminal Logger
    def print_md_log():
        step = dyn.get_number_of_steps()
        epot = atoms.get_potential_energy()
        ekin = atoms.get_kinetic_energy()
        etot = atoms.get_total_energy()
        temp_K = atoms.get_temperature()
        stress = atoms.get_stress(include_ideal_gas=True) / units.GPa
        stress_ave = sum(stress[:3]) / 3.0
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        vol = atoms.get_volume()

        print(
            f"  {step:>7}  | {epot:13.6f}  | {ekin:13.6f}  | {etot:13.6f}  |  {temp_K:11.3f}  |  {stress_ave:7.2f} | {vol:11.2f} | {elapsed_time:9.1f}",
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

    # Finish the trajectory file
    # dyn_params["trajectory"].close()

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
