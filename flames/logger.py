import datetime
import itertools
import os
import platform
import sys
from typing import Optional, TextIO

import ase
import numpy as np
from ase import units

from flames import VERSION


class BaseLogger:
    """
    Handles all logging and printing for the simulators.
    Separates the presentation logic from the simulation logic.
    """

    def __init__(self, simulation, output_file: Optional[TextIO] = None):
        """
        Initializes the logger.

        Parameters
        ----------
        simulation : Simulator
            The simulator instance to log.
        output_file : TextIO | None, optional
            A file path or stream to write the output to. If None, prints to stdout.
        """
        self.sim = simulation
        self.out_file = output_file
        self.warnings: list[str] = []

    def _print(self, *args, **kwargs) -> None:
        """Internal print function to direct output to file or console."""
        print(*args, **kwargs, file=self.out_file, flush=True)

    def _print_warning(self, message: str) -> None:
        """Internal warning function to direct warnings to file or console."""

        self.warnings.append(message)
        print(f"WARNING: {message}", file=self.out_file, flush=True)

    def print_header(self) -> None:
        """Prints the header for the simulation output."""
        atomic_numbers = set(
            set(list(self.sim.framework.get_atomic_numbers()))
            | set().union(
                *[
                    set(adsorbate.structure.get_atomic_numbers())
                    for adsorbate in self.sim.adsorbates
                ]
            )
        )

        header = rf"""
===========================================================================
         _______  __          ___      .___  ___.  _______     _______.
        |   ____||  |        /   \     |   \/   | |   ____|   /       |
        |  |__   |  |       /  ^  \    |  \  /  | |  |__     |   (----`
        |   __|  |  |      /  /_\  \   |  |\/|  | |   __|     \   \    
        |  |     |  `----./  _____  \  |  |  |  | |  |____.----)   |   
        |__|     |_______/__/     \__\ |__|  |__| |_______|_______/    
                                                               
        Flexible Lattice Adsorption by Monte Carlo Engine Simulation
                      powered by Python + ASE
                  Author: Felipe Lopes de Oliveira
===========================================================================

Code version: {VERSION}
Simulation started at {self.sim.start_time.strftime('%Y-%m-%d %H:%M:%S')}
Hostname: {platform.node()}
OS type: {platform.system()}
OS release: {platform.release()}
OS version: {platform.version()}

Python version: {sys.version.split()[0]}
Numpy version: {np.__version__}
ASE version: {ase.__version__}

Current directory: {os.getcwd()}
Random Seed: {self.sim.random_seed}

Model: {self.sim.model.name}
Running on device: {self.sim.device}

===========================================================================

Constants used:
Boltzmann constant:     {units.kB} eV/K
Beta (1/kT):            {self.sim.beta:.3f} eV^-1

===========================================================================

Simulation Parameters:
Temperature: {self.sim.T} K
Pressure: {self.sim.P / 1e5:.5f} bar
Fugacity: {self.sim.fugacity / units.J:.3f} Pa
Fugacity: {self.sim.fugacity:.5e} eV/m^3
(1/kB.T) * V * f = {self.sim.V * self.sim.beta * self.sim.fugacity} [-]

===========================================================================

System Information:
Framework: {self.sim.framework.get_chemical_formula()}
Framework: {self.sim.n_atoms_framework} atoms,
Framework mass: {np.sum(self.sim.framework.get_masses())} g/mol, {self.sim.framework_mass} kg
Framework energy: {self.sim.framework_energy} eV
Framework volume: {self.sim.V} m^3
Framework density: {self.sim.framework_density * 1e3} kg/m^3, {self.sim.framework_density} g/cm^3
Framework cell:
    {self.sim.cell[0, 0]:12.7f} {self.sim.cell[0, 1]:12.7f} {self.sim.cell[0, 2]:12.7f}
    {self.sim.cell[1, 0]:12.7f} {self.sim.cell[1, 1]:12.7f} {self.sim.cell[1, 2]:12.7f}
    {self.sim.cell[2, 0]:12.7f} {self.sim.cell[2, 1]:12.7f} {self.sim.cell[2, 2]:12.7f}

Perpendicular cell:
    {self.sim.perpendicular_cell[0, 0]:12.7f} {self.sim.perpendicular_cell[0, 1]:12.7f} {self.sim.perpendicular_cell[0, 2]:12.7f}
    {self.sim.perpendicular_cell[1, 0]:12.7f} {self.sim.perpendicular_cell[1, 1]:12.7f} {self.sim.perpendicular_cell[1, 2]:12.7f}
    {self.sim.perpendicular_cell[2, 0]:12.7f} {self.sim.perpendicular_cell[2, 1]:12.7f} {self.sim.perpendicular_cell[2, 2]:12.7f}

"""
        if not np.array_equal(self.sim._get_ideal_supercell(), np.array([1, 1, 1])):

            warning_msg = f"""\n
WARNING: Ideal supercell size is {self.sim._get_ideal_supercell()} (x, y, z).
Consider using automatic_supercell=True to create a supercell that
fits the cutoff radius of {self.sim.cutoff} Å or manually create a supercell.\n
"""
            self.warnings.append(warning_msg)
            header += warning_msg

        header += "Atomic positions:\n"

        for atom in self.sim.framework:
            header += "  {:2} {:12.7f} {:12.7f} {:12.7f}\n".format(atom.symbol, *atom.position)

        for adsorbate in self.sim.adsorbates:
            header += f"""
===========================================================================
Adsorbate: {adsorbate.structure.get_chemical_formula()}
Adsorbate: {self.sim.n_adsorbate_atoms[adsorbate.name]} atoms, {self.sim.adsorbate_mass[adsorbate.name]} kg
Adsorbate energy: {self.sim.adsorbate_energy[adsorbate.name]} eV

Atomic positions:
"""
            for atom in adsorbate.structure:
                header += "  {:2} {:12.7f} {:12.7f} {:12.7f}\n".format(atom.symbol, *atom.position)

            # Only prints if EOS parameters are set in the simulator
            if adsorbate.eos:
                header += f"""
===========================================================================
Equation of State Parameters: {type(adsorbate.eos).__name__}

    Critical temparure [K]: {adsorbate.eos.Tc:.6f}
    Critical pressure [Pa]: {adsorbate.eos.Pc:.6f}
    Acentric factor [-]:    {adsorbate.eos.omega:.6f}

    {adsorbate.eos.get_stable_phase_properties(self.sim.T, self.sim.P)[2]}

    MolFraction:           {adsorbate.mol_fraction:.8f} [-]
    Compressibility:       {adsorbate.eos.get_compressibility(self.sim.T, self.sim.P):.6f} [-]
    Fugacity coeff.:       {adsorbate.eos.get_fugacity_coefficient(self.sim.T, self.sim.P):.10f} [-]
    Bulk phase pressure:   {self.sim.P * adsorbate.eos.get_fugacity_coefficient(self.sim.T, self.sim.P):.6f} [Pa]

    Density of the bulk fluid phase:      {adsorbate.eos.get_bulk_phase_density(self.sim.T, self.sim.P):.6f} [kg/m^3]

    Amount of excess molecules:        {adsorbate.eos.get_bulk_phase_molar_density(self.sim.T, self.sim.P) * self.sim.V * self.sim.void_fraction:.10f} [-]

"""         
            partial_pressure = self.sim.P * adsorbate.eos.get_fugacity_coefficient(self.sim.T, self.sim.P) * adsorbate.mol_fraction
            header += f"""
===========================================================================
Conversion factors:
    Conversion factor molecules/unit cell -> mol/kg:         {self.sim.conv_factors['mol/kg'][adsorbate.name]:.9f}
    Conversion factor molecules/unit cell -> mg/g:           {self.sim.conv_factors['mg/g'][adsorbate.name]:.9f}
    Conversion factor molecules/unit cell -> cm^3 STP/gr:    {self.sim.conv_factors['cm^3 STP/gr'][adsorbate.name]:.9f}
    Conversion factor molecules/unit cell -> cm^3 STP/cm^3:  {self.sim.conv_factors['cm^3 STP/cm^3'][adsorbate.name]:.9f}
    Conversion factor molecules/unit cell -> %wt:            {self.sim.conv_factors['mg/g'][adsorbate.name] * 1e-1:.9f}

Partial pressure:
        {partial_pressure:>25.15f} Pascal
        {partial_pressure / 1e5:>25.15f} bar
        {partial_pressure / 101325:>25.15f} atm
        {partial_pressure / (101325 * 760):>25.15f} Torr
===========================================================================
"""

        header += """
===========================================================================
Shortest distances:
"""

        for i, j in list(itertools.combinations(atomic_numbers, 2)):
            header += f"  {ase.Atom(i).symbol:2} - {ase.Atom(j).symbol:2}: {self.sim.vdw[i] + self.sim.vdw[j]:.3f} Å\n"

        self._print(header)

    def print_restart_info(self) -> None:
        """Prints information when a simulation is restarted."""
        state = self.sim.current_system
        avg_binding_energy = (
            (
                self.sim.current_total_energy
                - self.sim.framework_energy
                - self.sim.n_adsorbates * self.sim.adsorbate_energy
            )
            / (units.kJ / units.mol)
            / self.sim.n_adsorbates
            if self.sim.n_adsorbates > 0
            else 0
        )
        self._print(f"Restarting simulation from step {self.sim.base_iteration}...")
        self._print(f"""
===========================================================================
Restart file requested.
Loaded state with {len(state)} total atoms.
Current total energy: {self.sim.current_total_energy:.3f} eV
Current number of adsorbates: {self.sim.n_adsorbates}
Current average binding energy: {avg_binding_energy:.3f} kJ/mol
===========================================================================
""")

    def print_debug_movement(
        self, movement, deltaE, prefactor, acc, rnd_number, adsorbate_name
    ) -> None:
        """
        Print debug information about the current state of the simulation.
        This method is called to provide detailed information about the current state of the system.
        """
        self._print(f"""
=======================================================================================================
Movement type: {movement}
Adsorbate: {adsorbate_name}
Interaction energy: {deltaE} eV, {(deltaE / (units.kJ / units.mol))} kJ/mol
Exponential factor:     {-self.sim.beta * deltaE:.3E}
Exponential:            {np.exp(-self.sim.beta * deltaE):.3E}
Prefactor:              {prefactor:.3E}
Acceptance probability: {acc:.3f}
Random number:          {rnd_number:.3f}
Accepted: {rnd_number < acc}
=======================================================================================================
""")


class GCMCLogger(BaseLogger):
    """
    Handles all logging and printing for a GCMC simulation.
    Separates the presentation logic from the simulation logic.
    """

    def _get_move_pct(self, move_name: str) -> float:
        moves = self.sim.n_movements[move_name]
        return np.average(moves) * 100 if len(moves) > 0 else 0.0

    def print_run_header(self) -> None:
        """Prints the header for the main GCMC loop."""

        header = "Movement statistics:\n"

        for adsorbate in self.sim.adsorbates:
            header += f"\nAdsorbate: {adsorbate.name}\n"
            for key, value in adsorbate.weights.__dict__.items():
                header += f" {key.capitalize():11}: {value:.3f}\n"

        header += """
===========================================================================

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Starting GCMC simulation
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

 Iteration |  Number of  |  Uptake  |    Tot En.   |Av. Ads. En.|  Pacc  |  Pdel  |  Ptra  |  Prot  |  Prin  |  Pswap  | Time
     -     |  Molecules  | [mmol/g] |     [eV]     |  [kJ/mol]  |    %   |    %   |   %    |   %    |   %    |    %    |  [s]
---------- | ----------- | -------- | ------------ | ---------- | ------ | ------ | ------ | ------ | ------ | ------- | -----"""
        self._print(header)

    def print_step_info(self, step, average_ads_energy, step_time, adsorbate_name) -> None:

        line_str = "{:^11}|{:^13}|{:>9.2f} |{:>13.4f} |{:>11.4f} |{:7.2f} |{:7.2f} |{:7.2f} |{:7.2f} |{:7.2f} |{:7.2f} |{:9.2f}"

        self._print(
            line_str.format(
                step,
                sum(self.sim.uptake_list[-1]),
                sum(self.sim.uptake_list[-1]) * self.sim.conv_factors["mol/kg"][adsorbate_name],
                self.sim.current_total_energy,
                average_ads_energy,
                self._get_move_pct("insertion"),
                self._get_move_pct("deletion"),
                self._get_move_pct("translation"),
                self._get_move_pct("rotation"),
                self._get_move_pct("reinsertion"),
                self._get_move_pct("particle_swap"),
                step_time,
            )
        )

    def print_optimization_start(self, target: str) -> None:
        """Prints a header for framework or adsorbate optimization."""
        self._print(f"""
===========================================================================
Start optimizing {target} structure...
===========================================================================
""")

    def print_load_state_info(self, n_atoms, average_ads_energy) -> None:
        """Prints information about the loading state."""
        self._print(f"""
===========================================================================

Restarting GCMC simulation from previous configuration...

Loaded state with {n_atoms} total atoms.

Current total energy: {self.sim.current_total_energy:.3f} eV
Current number of adsorbates: {self.sim.n_adsorbates}
Current average binding energy: {average_ads_energy:.3f} kJ/mol

Current steps are: {self.sim.base_iteration}

===========================================================================
""")

    def print_iteration_info(self, iteration_data: dict) -> None:
        """Prints a single log line for a GCMC iteration."""
        line_str = "{:^11}|{:^13}|{:>9.2f} |{:>13.4f} |{:>11.4f} |{:7.2f} |{:7.2f} |{:7.2f} |{:7.2f} |{:9.2f}"
        self._print(line_str.format(*iteration_data.values()))

    def print_debug_movement(
        self, movement, deltaE, prefactor, acc, rnd_number, adsorbate_name
    ) -> None:
        """Prints detailed debug information for a single MC move."""
        self._print(f"""
=======================================================================================================
Movement type: {movement}
Current number of adsorbates: {self.sim.n_adsorbates}
Adsorbate: {adsorbate_name}
Interaction energy: {deltaE} eV, {(deltaE / (units.kJ / units.mol))} kJ/mol
Exponential factor:     {-self.sim.beta * deltaE:.3E}
Exponential:            {np.exp(-self.sim.beta * deltaE):.3E}
Prefactor:              {prefactor:.3E}
Acceptance probability: {acc:.3f}
Random number:          {rnd_number:.3f}
Accepted: {rnd_number < acc}
=======================================================================================================
""")

    def print_summary(self) -> None:
        """Prints the final summary of the simulation results."""

        self.sim.equilibrate()
        eq_results = self.sim.equilibrated_results

        self._print("\nMovement statistics:\n")
        for move, stats in self.sim.n_movements.items():
            total_attempts = len(stats)
            acceptance_rate = np.mean(stats) * 100 if total_attempts > 0 else 0.0
            self._print(
                f"Move: {move.capitalize():14} | Total attempts: {total_attempts:6} | Acceptance rate: {acceptance_rate:6.2f}%"
            )

        self._print(f"""
===========================================================================

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Finishing GCMC simulation
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    pyMSER Equilibration Results:
    ------------------------------------------------------------------------------
    Start of equilibrated data:          {eq_results['t0']} of {len(self.sim.uptake_list)}
    Total equilibrated steps:            {len(self.sim.uptake_list) - eq_results['t0']}  ({(len(self.sim.uptake_list) - eq_results['t0']) / len(self.sim.uptake_list) * 100:.2f}%)
    Equilibrated:                        {eq_results['t0'] < 0.75 * len(self.sim.uptake_list)}
    Average over equilibrated data:      {eq_results['average']:.4f} ± {eq_results['uncertainty']:.4f} molecules/unit cell
    Number of uncorrelated samples:      {eq_results['uncorr_samples']:.1f}
    Autocorrelation time:                {eq_results['ac_time']:.1f}
    ------------------------------------------------------------------------------
    """)

        for ads in self.sim.adsorbates:

            avg_uptake = eq_results[f"average_{ads.name}"]
            std_uptake = eq_results[f"uncertainty_{ads.name}"]

            enthalpy = eq_results[f"enthalpy_{ads.name}_kJ_per_mol"]
            enthalpy_sd = eq_results[f"enthalpy_{ads.name}_sd_kJ_per_mol"]

            avg_uptake_excess = avg_uptake - self.sim.excess_nmol[ads.name]

            cf = self.sim.conv_factors

            self._print(f"""
    Average properties of the system: {ads.name}
    ------------------------------------------------------------------------------
    Average loading absolute [molecules/unit cell]       {avg_uptake:12.5f} +/- {std_uptake:12.5f} [-]
    Average loading absolute [mol/kg framework]          {avg_uptake * cf["mol/kg"][ads.name]:12.5f} +/- {std_uptake * cf["mol/kg"][ads.name]:12.5f} [-]
    Average loading absolute [mg/g framework]            {avg_uptake * cf["mg/g"][ads.name]:12.5f} +/- {std_uptake * cf["mg/g"][ads.name]:12.5f} [-]
    Average loading absolute [cm^3 (STP)/gr framework]   {avg_uptake * cf["cm^3 STP/gr"][ads.name]:12.5f} +/- {std_uptake * cf["cm^3 STP/gr"][ads.name]:12.5f} [-]
    Average loading absolute [cm^3 (STP)/cm^3 framework] {avg_uptake * cf["cm^3 STP/cm^3"][ads.name]:12.5f} +/- {std_uptake * cf["cm^3 STP/cm^3"][ads.name]:12.5f} [-]
    Average loading absolute [%wt framework]             {avg_uptake * cf["mg/g"][ads.name] * 1e-1:12.5f} +/- {std_uptake * cf["mg/g"][ads.name] * 1e-1:12.5f} [-]

    Average excess absolute [molecules/unit cell]        {avg_uptake_excess:12.5f} +/- {std_uptake:12.5f} [-]
    Average loading absolute [mol/kg framework]          {avg_uptake_excess * cf["mol/kg"][ads.name]:12.5f} +/- {std_uptake * cf["mol/kg"][ads.name]:12.5f} [-]
    Average loading absolute [mg/g framework]            {avg_uptake_excess * cf["mg/g"][ads.name]:12.5f} +/- {std_uptake * cf["mg/g"][ads.name]:12.5f} [-]
    Average loading absolute [cm^3 (STP)/gr framework]   {avg_uptake_excess * cf["cm^3 STP/gr"][ads.name]:12.5f} +/- {std_uptake * cf["cm^3 STP/gr"][ads.name]:12.5f} [-]
    Average loading absolute [cm^3 (STP)/cm^3 framework] {avg_uptake_excess * cf["cm^3 STP/cm^3"][ads.name]:12.5f} +/- {std_uptake * cf["cm^3 STP/cm^3"][ads.name]:12.5f} [-]
    Average loading absolute [%wt framework]             {avg_uptake_excess * cf["mg/g"][ads.name] * 1e-1:12.5f} +/- {std_uptake * cf["mg/g"][ads.name] * 1e-1:12.5f} [-]


    Enthalpy of adsorption: [kJ/mol]                     {enthalpy:12.5f} +/- {enthalpy_sd:12.5f} [kJ/mol]
""")
        warning_text = "\n".join([f"WARNING: {warning}" for warning in self.warnings])

        self._print(f"""
===========================================================================
GCMC simulation finished successfully!

{len(self.warnings)} Warnings during the simulation.
{warning_text}
===========================================================================

Simulation finished at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Simulation duration: {datetime.datetime.now() - self.sim.start_time}
===========================================================================
""")


class TMMCLogger(BaseLogger):
    """
    Handles all logging and printing for a TMMC simulation.
    Separates the presentation logic from the simulation logic.
    """

    def print_run_header(self) -> None:
        """Prints the header for the main TMMC loop."""
        header = """
===========================================================================

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Starting TMMC simulation
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Iteration  |  Number of  |    Tot En.   | Del. Energy  | Ins. Energy  |  Time
     -     |  Molecules  |     [eV]     |     [eV]     |     [eV]     |   [s]
---------- | ----------- | ------------ | ------------ | ------------ | -------"""
        self._print(header)

    def print_step_info(self, step, del_energy, ins_energy, step_time) -> None:
        """Prints info on one TMMC step."""
        line_str = "{:^11}|{:^13}|{:>13.4f} |{:>13.4f} |{:>13.4f} |{:7.2f}"
        self._print(
            line_str.format(
                step,
                self.sim.n_adsorbates,
                self.sim.current_total_energy,
                del_energy,
                ins_energy,
                step_time,
            )
        )

    def print_load_state_info(self, n_atoms):
        """Prints information about the loading state."""
        self._print(f"""
===========================================================================

Restarting TMMC simulation from previous configuration...

Loaded state with {n_atoms} total atoms.

Current total energy: {self.sim.current_total_energy:.3f} eV
Current number of adsorbates: {self.sim.n_adsorbates}

Current steps are: {self.sim.base_iteration}

===========================================================================
""")

    def print_restart_info(self) -> None:
        """Prints information when a simulation is restarted."""
        state = self.sim.current_system
        self._print(f"Restarting simulation from step {self.sim.base_iteration}...")
        self._print(f"""
===========================================================================
Restart file requested.
Loaded state with {len(state)} total atoms.
Current total energy: {self.sim.current_total_energy:.3f} eV
Current number of adsorbates: {self.sim.n_adsorbates}
===========================================================================""")


class WidomLogger(BaseLogger):
    """
    Handles all logging and printing for a Widom insertion simulation.
    Separates the presentation logic from the simulation logic.
    """

    def _print(self, *args, **kwargs) -> None:
        """Internal print function to direct output to file or console."""
        print(*args, **kwargs, file=self.out_file, flush=True)

    def print_run_header(self) -> None:
        """Prints the header for the main Widom loop."""
        header = """
===========================================================================

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Starting Widom simulation
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Iteration  |     dE (eV)    |  dE (kJ/mol)  | kH [mol kg-1 Pa-1]  |  dH (kJ/mol) | Time (s)
-------------------------------------------------------------------------------------------"""
        self._print(header)

    def print_iteration_info(self, iteration_data: list) -> None:
        """Prints a single log line for a Widom iteration."""
        line_str = "{:^10} | {:>14.6e} | {:>13.2f} | {:>19.3e} | {:12.2f} | {:8.2f}"
        self._print(line_str.format(*iteration_data))

    def print_summary(self) -> None:
        """
        Print the footer for the simulation output.
        This method is called at the end of the simulation to display the final results and elapsed time.
        """

        warning_text = "\n".join([f"WARNING: {warning}" for warning in self.warnings])

        self._print(f"""
===========================================================================

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Finishing Widom simulation
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    Average properties of the system:
    ------------------------------------------------------------------------------
    Henry coefficient: [mol/kg/Pa]      {self.sim.kH:12.5e} +/- {self.sim.kH_std_dv:12.5e} [-]
    Enthalpy of adsorption: [kJ/mol]    {self.sim.dH:12.5f} +/- {self.sim.dH_std_dv:12.5f} [-]

===========================================================================
Simulation finished successfully!

{len(self.warnings)} Warnings during the simulation.
{warning_text}
===========================================================================

Simulation finished at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Simulation duration: {datetime.datetime.now() - self.sim.start_time}
===========================================================================
""")

        if len(self.warnings) > 0:
            print("\n".join(["=" * 75] * 3))
            print(f"{len(self.warnings)} Warnings during the simulation:")
            for warning in self.warnings:
                self._print(f"WARNING: {warning}")
