import os
import sys

sys.path.append("C:\\Users\\flopes\\Documents\\PRs\\flames")

# Hide UserWarning and RuntimeWarning messages
import warnings

import ase
import torch
from ase.data import vdw_radii
from ase.io import read
from mace.calculators import mace_mp

from flames.adsorbate import Adsorbate
from flames.gcmc import GCMC

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the framework structure
framework: ase.Atoms = read("mg-mof-74.cif")  # type: ignore

adsorbate_1 = Adsorbate(
    name="CO2",
    structure="co2.xyz",
    eos={"criticalTemperature": 304.1282, "criticalPressure": 7377300.0, "acentricFactor": 0.22394},
    mol_fraction=0.7,
)

adsorbate_2 = Adsorbate(
    name="H2O",
    structure="h2o.xyz",
    eos={"criticalTemperature": 628.0, "criticalPressure": 14100000.0, "acentricFactor": 0.5293},
    mol_fraction=0.3,
)

model = mace_mp(
    model="medium-0b2",
    dispersion=False,
    damping="zero",  # choices: ["zero", "bj", "zerom", "bjm"]
    dispersion_xc="pbe",
    default_dtype="float32",
    device=device,
)

Temperature = 298.0  # in Kelvin
pressure = 1_000_000  # in Pa = 1 bar
MCSteps = 100


print(
    f"Running GCMC simulation for pressure: {pressure:.2f} Pa at temperature: {Temperature:.2f} K"
)

gcmc = GCMC(
    model=model,
    framework_atoms=framework,
    adsorbates=[adsorbate_1, adsorbate_2],
    temperature=Temperature,
    pressure=pressure,
    device=device,
    vdw_radii=vdw_radii,
    vdw_factor=0.6,
    save_frequency=1,
    debug=True,
    output_to_file=True,
    random_seed=42,
    cutoff_radius=6.0,
    automatic_supercell=True,
)


gcmc.logger.print_header()

gcmc.run(MCSteps)

gcmc.equilibrate()
#gcmc.logger.print_summary()

gcmc.save_results()
