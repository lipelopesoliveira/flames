import json

import ase
from ase.constraints import FixBondLengths
from ase.data import vdw_radii
from mace.calculators import mace_mp
from numba import get_num_threads, set_num_threads

from flames.adsorbate import Adsorbate
from flames.calculators.lennard_jones import CustomLennardJones
from flames.gcmc import GCMC
from flames.utilities import read_cif

NUM_THREADS_TO_USE = 1
set_num_threads(NUM_THREADS_TO_USE)

print(get_num_threads())

with open("DREIDING.json", "r") as f:
    lj_params = json.loads(f.read())

calc1 = CustomLennardJones(lj_params, vdw_cutoff=12.8, shifted=True)

calc2 = mace_mp(
    model="medium-0b2",
    dispersion=False,
    damping="zero",  # choices: ["zero", "bj", "zerom", "bjm"]
    dispersion_xc="pbe",
    default_dtype="float32",
    device="cuda",
)

# Load the framework structure
framework: ase.Atoms = read_cif("GZU-1.cif")  # type: ignore

adsorbate = Adsorbate(
    name="CH4",
    structure="ch4.xyz",
    weights={
        "insertion": 0.333,
        "deletion": 0.333,
        "translation": 0,
        "rotation": 0,
        "reinsertion": 0.3333,
    },
)

c = FixBondLengths([[0, 1], [0, 2], [0, 3], [0, 4]])
adsorbate.structure.set_constraint(c)  # type: ignore

Temperature = 198.0  # in Kelvin
pressure = 1_000_000  # in Pa = 1 bar
MCSteps = 3_00
MDSteps = 3_000

print(
    f"Running GCMC simulation for pressure: {pressure:.2f} Pa at temperature: {Temperature:.2f} K"
)

gcmc = GCMC(
    model=calc1,
    framework_atoms=framework,
    adsorbates=adsorbate,
    temperature=Temperature,
    pressure=pressure,
    device="cpu",
    vdw_radii=vdw_radii,
    vdw_factor=0.6,
    save_frequency=1,
    debug=False,
    output_to_file=True,
    cutoff_radius=8.0,
    automatic_supercell=True,
)

gcmc.logger.print_header()

for j in range(5):
    gcmc.run(MCSteps)
    gcmc.npt(
        nsteps=MDSteps,
        time_step=0.5,
        mode="aniso_flex",
        calculator=calc2,
        movie_interval=1,
        output_interval=1000,
    )

gcmc.run(MCSteps)
gcmc.logger.print_summary()
gcmc.save_results()
