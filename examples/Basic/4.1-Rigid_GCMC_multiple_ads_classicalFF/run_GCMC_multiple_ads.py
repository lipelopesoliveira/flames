#!/usr/bin/env python

import os

# Hide UserWarning and RuntimeWarning messages
import warnings

import ase
import json
import numpy as np
from ase.data import vdw_radii
from ase.io import read
from ase.calculators import mixing

from flames.adsorbate import Adsorbate
from flames.gcmc import GCMC
from flames.calculators.ewald_new import CustomEwald
from flames.calculators.lennard_jones_new import CustomLennardJones

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load the framework structure with partial charges
framework: ase.Atoms = read("MgMOF-74_DDEC.cif", store_tags=True)  # type: ignore
framework.set_initial_charges(framework.info["_atom_site_charge"])
framework.arrays["labels"] = np.array(framework.get_chemical_symbols(), dtype=object)
framework.info = {}

adsorbate_1 = Adsorbate(
    name="CO2",
    structure="co2_labels.xyz",
    move_weights={'insertion': 1.0, 'deletion': 1.0, 'translation': 1.0, 'rotation': 1.0, 'identity_swap': 1.0,},
    eos={"criticalTemperature": 304.1282, "criticalPressure": 7377300.0, "acentricFactor": 0.22394},
    mol_fraction=0.7,
)

adsorbate_2 = Adsorbate(
    name="H2O",
    structure="H2O_labels.xyz",
    move_weights={'insertion': 1.0, 'deletion': 1.0, 'translation': 1.0, 'rotation': 1.0, 'identity_swap': 1.0,},
    eos={"criticalTemperature": 628.0, "criticalPressure": 14100000.0, "acentricFactor": 0.5293},
    mol_fraction=0.3,
)

with open("UFF.json", "r") as f:
    uff_lj_params = json.loads(f.read())

with open("TraPPE.json", "r") as f:
    trappe_lj_params = json.loads(f.read())

with open("TIP4Ew.json", "r") as f:
    tip4ew_params = json.loads(f.read())

ewald = CustomEwald(cutoff=12.0, precision=1e-6)
lj = CustomLennardJones({**uff_lj_params, **trappe_lj_params, **tip4ew_params}, vdw_cutoff=12.0)

calc = mixing.SumCalculator([lj, ewald])

Temperature = 298.0  # in Kelvin
pressure = 1_000_000  # in Pa = 1 bar
MCSteps = 50000


print(
    f"Running GCMC simulation for pressure: {pressure:.2f} Pa at temperature: {Temperature:.2f} K"
)

gcmc = GCMC(
    model=calc,
    framework_atoms=framework,
    adsorbates=[adsorbate_1, adsorbate_2],
    temperature=Temperature,
    pressure=pressure,
    device='cpu',
    vdw_radii=vdw_radii,
    vdw_factor=0.6,
    save_frequency=1,
    debug=False,
    output_to_file=True,
    random_seed=42,
    cutoff_radius=12.0,
    automatic_supercell=True,
)


gcmc.logger.print_header()

gcmc.run(MCSteps)

gcmc.logger.print_summary()

gcmc.save_results()
