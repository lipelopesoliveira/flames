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
                    default=100.0,
                    required=True,
                    action='store',
                    metavar='EXTERNAL_PRESSURE',
                    help='External pressure for the MD simulation in Pascals. Default: 100.')
parser.add_argument('--ExternalTemperature',
                    type=float,
                    default=298.0,
                    required=True,
                    action='store',
                    metavar='EXTERNAL_TEMPERATURE',
                    help='External temperature for the simulation in Kelvin. Default: 298.0K.')
parser.add_argument('--NMolecules',
                    type=int,
                    default=1,
                    required=True,
                    action='store',
                    metavar='N_MOLECULES',
                    help='Number of molecules in the system to build the macrostate')
 
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
    structure=read(AdsorbatePath),
    move_weights={
        "insertion": 0.0,
        "deletion": 0.0,
        "translation": 0.5,
        "rotation": 0.5,
        "reinsertion": 0.1
        },
    eos={
        "criticalTemperature": 647.096,
        "criticalPressure": 22064000.0,
        "acentricFactor": 0.3449
        }
    )

Temperature = 298.0  # in Kelvin
pressure = 100_000  # in Pa = 1 bar only used for the MD steps
MDSteps = 1000

MCSteps = max(100 * (args.NMolecules), 2000)

outputFolder = f'results_{args.ExternalTemperature:.2f}_{args.ExternalPressure:.2f}_{args.NMolecules:03d}'

gcmc = GCMC(
    model=model,
    framework_atoms=framework,
    adsorbates=adsorbate,
    temperature=args.ExternalTemperature,
    pressure=args.ExternalPressure,
    device=device,
    vdw_radii=vdw_radii,
    vdw_factor = 1,
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

gcmc.insert_adsorbates(ads_name=adsorbate.name, n_adsorbates=args.NMolecules, max_attempts=10000)
    
for i in range(10):
    gcmc.run(MCSteps)

    gcmc.md(
        nsteps=MDSteps,
        time_step=0.5,
        ensemble="NPT",
        thermostat="MTK",
        movie_interval=100
    )

md_traj = Trajectory(f"Trajectory_{args.NMolecules:04d}.traj", "w")

gcmc.md(
    nsteps=300000,
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

tmmc.run_tmmc(3000, f"Trajectory_{args.NMolecules:04d}.traj")

tmmc.logger.print_summary()
tmmc.save_results()
