"""
This script runs a hybrid TMMC-MD simulation using the FLAMES package. It performs a series of GCMC simulations followed by MD simulations for a specific numnber of molecules, and then runs TMMC on the resulting trajectories.
The script takes command-line arguments for external pressure, external temperature, and the maximum number of molecules to simulate. It uses a custom MACE model for energy calculations and saves the results in specified output folders
for each molecule count.
"""

import argparse
import os

# Hide UserWarning and RuntimeWarning messages
import warnings

import ase
import torch
from ase.data import vdw_radii
from ase.io import Trajectory, read
from mace.calculators import MACECalculator

from flames.adsorbate import Adsorbate
from flames.gcmc import GCMC
from flames.tmmc import TMMC

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


parser = argparse.ArgumentParser(description="Run TMMC with FLAMES.")
parser.add_argument(
    "--ExternalPressure",
    type=float,
    default=100000.0,
    required=False,
    action="store",
    metavar="EXTERNAL_PRESSURE",
    help="External pressure for the MD simulation in Pascals. Default: 100000.0 Pa = 1 bar.",
)
parser.add_argument(
    "--ExternalTemperature",
    type=float,
    default=298.0,
    required=False,
    action="store",
    metavar="EXTERNAL_TEMPERATURE",
    help="External temperature for the simulation in Kelvin. Default: 298.0K. = 25°C.",
)
parser.add_argument(
    "--MDSteps",
    type=int,
    default=1000,
    required=False,
    action="store",
    metavar="MD_STEPS",
    help="Number of MD steps for each simulation. Default: 1000.",
)
parser.add_argument(
    "--MCSteps",
    type=int,
    default=2000,
    required=False,
    action="store",
    metavar="MC_STEPS",
    help="Number of MC steps for each simulation. Default: 2000.",
)
parser.add_argument(
    "--MDProdSteps",
    type=int,
    default=300000,
    required=False,
    action="store",
    metavar="MD_PROD_STEPS",
    help="Number of MD production steps for each simulation. Default: 300000.",
)
parser.add_argument(
    "--TMMCSteps",
    type=int,
    default=1000,
    required=False,
    action="store",
    metavar="TMMC_STEPS",
    help="Number of TMMC steps for each simulation. Default: 1000.",
)
parser.add_argument(
    "--EquilibrationCycles",
    type=int,
    default=10,
    required=False,
    action="store",
    metavar="EQUILIBRATION_CYCLES",
    help="Number of equilibration cycles for each simulation. Default: 10.",
)
parser.add_argument(
    "--NMolecules",
    type=int,
    default=1,
    required=False,
    action="store",
    metavar="NUM_MOLECULES",
    help="Number of molecules to simulate. Default: 1.",
)

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

model = MACECalculator(model_paths="MOF-303_mace.model", device=device, default_dtype="float64")

FrameworkPath = "MOF-303_symm.cif"
AdsorbatePath = "H2O.xyz"

# Load the framework structure
framework: ase.Atoms = read(FrameworkPath)  # type: ignore

# Load the adsorbate structure
adsorbate = Adsorbate(
    name="H2O",
    structure=read(AdsorbatePath),  # type: ignore
    move_weights={
        "insertion": 0.0,
        "deletion": 0.0,
        "translation": 0.5,
        "rotation": 0.5,
        "reinsertion": 0.0,
    },
    eos={"criticalTemperature": 647.096, "criticalPressure": 22064000.0, "acentricFactor": 0.3449},
)


print(f"\n\nRunning simulation for {args.NMolecules} molecules...\n")

outputFolder = (
    f"results_{args.ExternalTemperature:.2f}_{args.ExternalPressure:.2f}_{args.NMolecules:03d}"
)


gcmc = GCMC(
    model=model,
    framework_atoms=framework,
    adsorbates=adsorbate,
    temperature=args.ExternalTemperature,
    pressure=args.ExternalPressure,
    device=device,
    vdw_radii=vdw_radii,
    vdw_factor=0.6,
    adsorbate_energy=-467.837350,
    max_deltaE=1.555,
    save_frequency=100,
    save_rejected=False,
    output_to_file=True,
    fugacity_coeff=1.0,
    random_seed=None,
    cutoff_radius=5.0,
    automatic_supercell=False,
    output_folder=outputFolder,
)


gcmc.logger.print_header()


gcmc.insert_adsorbates(ads_name=adsorbate.name, n_adsorbates=args.NMolecules, max_attempts=10000)

for i in range(args.EquilibrationCycles):
    gcmc.run(args.MCSteps)

    gcmc.md(
        nsteps=args.MDSteps, time_step=0.5, ensemble="NPT", thermostat="MTK", movie_interval=100
    )

md_traj = Trajectory(f"Trajectory_{args.NMolecules:04d}.traj", "w")

gcmc.md(
    nsteps=args.MDProdSteps,
    time_step=0.5,
    ensemble="NPT",
    thermostat="MTK",
    movie_interval=100,
    set_momenta=False,
    trajectory_file=md_traj,
)


tmmc = TMMC(
    model,
    framework,
    adsorbate,
    temperature=args.ExternalTemperature,
    pressure=args.ExternalPressure,
    device=device,
    vdw_radii=vdw_radii,
    adsorbate_energy=-467.837350,
    vdw_factor=0.5,
    save_frequency=1,
    automatic_supercell=False,
    output_folder=outputFolder,
)

tmmc.logger.print_header()

tmmc.run_tmmc(args.TMMCSteps, f"Trajectory_{args.NMolecules:04d}.traj")

tmmc.logger.print_summary()
tmmc.save_results()
