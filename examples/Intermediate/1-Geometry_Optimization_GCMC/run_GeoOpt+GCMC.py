import os
import sys

# Hide UserWarning and RuntimeWarning messages
import warnings

import ase
import torch
from ase.data import vdw_radii
from ase.io import read
from ase.optimize import LBFGS
from mace.calculators import mace_mp

from flames.ase_utils import crystalOptimization
from flames.gcmc import GCMC

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = "cuda" if torch.cuda.is_available() else "cpu"

FrameworkPath = "mg-mof-74.cif"
AdsorbatePath = "co2.xyz"

model = mace_mp(
    model="medium-0b2",
    dispersion=False,
    damping="zero",  # choices: ["zero", "bj", "zerom", "bjm"]
    dispersion_xc="pbe",
    default_dtype="float32",
    device=device,
)


# Load the framework structure
framework: ase.Atoms = read(FrameworkPath)  # type: ignore

resultsDict, frameworkOpt = crystalOptimization(
    atoms_in=framework,
    calculator=model,
    optimizer=LBFGS,  # type: ignore
    out_file=sys.stdout,
    fmax=0.001,
    opt_cell=True,
    fix_symmetry=True,
    hydrostatic_strain=True,
    constant_volume=False,
    scalar_pressure=0.0,
    max_steps=1000,
    trajectory="FrameworkOpt.traj",
    verbose=False,
)

# Remove constrains from frameworkOpt
frameworkOpt.set_constraint(None)

# Load the adsorbate structure
adsorbate: ase.Atoms = read(AdsorbatePath)  # type: ignore

resultsDict, adsorbateOpt = crystalOptimization(
    atoms_in=adsorbate,
    calculator=model,
    optimizer=LBFGS,  # type: ignore
    out_file=sys.stdout,
    fmax=0.001,
    opt_cell=False,
    fix_symmetry=False,
    hydrostatic_strain=False,
    constant_volume=False,
    scalar_pressure=0.0,
    max_steps=1000,
    trajectory="AdsorbateOpt.traj",
    verbose=False,
)

# Remove constrains from adsorbateOpt
adsorbateOpt.set_constraint(None)


Temperature = 298.0  # in Kelvin
pressure = 100_000  # in Pa = 1 bar
MCSteps = 30_000


print(
    f"Running GCMC simulation for pressure: {pressure:.2f} Pa at temperature: {Temperature:.2f} K"
)

gcmc = GCMC(
    model=model,
    framework_atoms=frameworkOpt,
    adsorbate_atoms=adsorbateOpt,
    temperature=Temperature,
    pressure=pressure,
    device=device,
    vdw_radii=vdw_radii,
    vdw_factor=0.6,
    save_frequency=1,
    debug=False,
    output_to_file=True,
    criticalTemperature=304.1282,
    criticalPressure=7377300.0,
    acentricFactor=0.22394,
    cutoff_radius=6.0,
    automatic_supercell=True,
)

gcmc.logger.print_header()

gcmc.run(MCSteps)

gcmc.logger.print_summary()

gcmc.save_results()
