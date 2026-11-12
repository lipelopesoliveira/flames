import os

# Hide UserWarning and RuntimeWarning messages
import warnings
import argparse

import ase
import torch
from ase.data import vdw_radii
from ase.io import read, Trajectory
from mace.calculators import MACECalculator

from flames.tmmc import TMMC
from flames.gcmc import GCMC
from flames.adsorbate import Adsorbate

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


parser = argparse.ArgumentParser(description='Run TMMC with FLAMES.')
parser.add_argument('--ExternalPressure',
                    type=float,
                    default=100000.0,
                    required=False,
                    action='store',
                    metavar='EXTERNAL_PRESSURE',
                    help='External pressure for the MD simulation in Pascals. Default: 100000.0 Pa = 1 bar.')
parser.add_argument('--ExternalTemperature',
                    type=float,
                    default=298.0,
                    required=False,
                    action='store',
                    metavar='EXTERNAL_TEMPERATURE',
                    help='External temperature for the simulation in Kelvin. Default: 298.0K. = 25°C.')
parser.add_argument('--MDSteps',
                    type=int,
                    default=1000,
                    required=False,
                    action='store',
                    metavar='MD_STEPS',
                    help='Number of MD steps for each simulation. Default: 1000.')
parser.add_argument('--MCSteps',
                    type=int,
                    default=2000,
                    required=False,
                    action='store',
                    metavar='MC_STEPS',
                    help='Number of MC steps for each simulation. Default: 2000.')
parser.add_argument('--MDProdSteps',
                    type=int,
                    default=300000,
                    required=False,
                    action='store',
                    metavar='MD_PROD_STEPS',
                    help='Number of MD production steps for each simulation. Default: 300000.')
parser.add_argument('--TMMCSteps',
                    type=int,
                    default=1000,
                    required=False,
                    action='store',
                    metavar='TMMC_STEPS',
                    help='Number of TMMC steps for each simulation. Default: 1000.')
parser.add_argument('--EquilibrationCycles',
                    type=int,
                    default=10,
                    required=False,
                    action='store',
                    metavar='EQUILIBRATION_CYCLES',
                    help='Number of equilibration cycles for each simulation. Default: 10.')
parser.add_argument('--MaxMolecules',
                    type=int,
                    default=45,
                    required=False,
                    action='store',
                    metavar='MAX_MOLECULES',
                    help='Maximum number of molecules to simulate. Default: 45.')

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

model = MACECalculator(
    model_paths="MOF-303_mace.model",
    device=device,
    default_dtype="float64"
    )

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
        "reinsertion": 0.0
        },
    eos={
        "criticalTemperature": 647.096,
        "criticalPressure": 22064000.0,
        "acentricFactor": 0.3449
        }
    )

for NMolecules in range(0, args.MaxMolecules + 1):
    if os.path.exists(f"Trajectory_{NMolecules:04d}.traj"):
        print(f"Trajectory for {NMolecules} molecules already exists. Skipping simulation.")
        continue

    print(f"\n\nRunning simulation for {NMolecules} molecules...\n")


    outputFolder = f'results_{args.ExternalTemperature:.2f}_{args.ExternalPressure:.2f}_{NMolecules:03d}'

    previous_state = os.path.exists(f"Trajectory_{NMolecules - 1:04d}.traj")

    gcmc = GCMC(
        model=model,
        framework_atoms=framework,
        adsorbates=adsorbate,
        temperature=args.ExternalTemperature,
        pressure=args.ExternalPressure,
        device=device,
        vdw_radii=vdw_radii,
        vdw_factor = 0.6,
        adsorbate_energy=-467.837350,
        max_deltaE = 1.555,
        save_frequency = 100,
        save_rejected = False,
        output_to_file = True,
        fugacity_coeff = 1.0,
        random_seed = None,
        cutoff_radius = 5.0,
        automatic_supercell = False,
        output_folder=outputFolder
        )


    gcmc.logger.print_header()

    if previous_state:
        gcmc.load_state(f"Trajectory_{NMolecules - 1:04d}.traj")
        gcmc.insert_adsorbates(ads_name=adsorbate.name, n_adsorbates=1, max_attempts=10000)
    else:    
        gcmc.insert_adsorbates(ads_name=adsorbate.name, n_adsorbates=NMolecules, max_attempts=10000)
        
    for i in range(args.EquilibrationCycles):
        gcmc.run(args.MCSteps)

        gcmc.md(
            nsteps=args.MDSteps,
            time_step=0.5,
            ensemble="NPT",
            thermostat="MTK",
            movie_interval=100
        )

    md_traj = Trajectory(f"Trajectory_{NMolecules:04d}.traj", "w")

    gcmc.md(
        nsteps=args.MDProdSteps,
        time_step=0.5,
        ensemble="NPT",
        thermostat="MTK",
        movie_interval=100,
        set_momenta=False,
        trajectory_file=md_traj
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
        output_folder=outputFolder
    )

    tmmc.logger.print_header()

    tmmc.run_tmmc(args.TMMCSteps, f"Trajectory_{NMolecules:04d}.traj")

    tmmc.logger.print_summary()
    tmmc.save_results()
