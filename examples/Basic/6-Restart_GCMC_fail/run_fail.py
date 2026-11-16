import json

import ase
from ase.data import vdw_radii

from flames.adsorbate import Adsorbate
from flames.calculators.lennard_jones import CustomLennardJones
from flames.gcmc import GCMC
from flames.utilities import read_cif

with open("TraPPE_zeo.json", "r") as f:
    lj_params = json.loads(f.read())

calc = CustomLennardJones(lj_params, vdw_cutoff=12.8, shifted=True)

# Load the framework structure
framework: ase.Atoms = read_cif("MFI.cif")

adsorbate = Adsorbate(
    name="CH4",
    structure="ch4.xyz",
    move_weights={
        "insertion": 0.333,
        "deletion": 0.333,
        "translation": 0,
        "rotation": 0,
        "reinsertion": 0.3333,
    },
)

gcmc = GCMC(
    model=calc,
    framework_atoms=framework,
    adsorbates=adsorbate,
    temperature=298.15,
    pressure=1e5,
    device="cpu",
    vdw_radii=vdw_radii,
    debug=False,
    output_to_file=True,
    cutoff_radius=12.0,
    automatic_supercell=True,
)

gcmc.logger.print_header()

gcmc.restart()

gcmc.run(10000 - gcmc.base_iteration)
gcmc.logger.print_summary()

gcmc.save_results()
