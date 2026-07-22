=========================================
Mixed calculators for GCMC/MD simulations
=========================================

This example explores the use of different calculators for the GCMC and MD simulations.
The GCMC simulation will use a classical force field (DREIDING) using the custom `LennarJones` calculator,
while the MD simulation will use the `mace` calculator. 
This is useful when you want to introduce some dynamics in the system using a MLP calculator for the MD simulation,
but you still want to use a classical force field calculator for the GCMC simulation.

.. warning:: This model is used here just as an example, you should look carefully into the mace documentation to undestand how the potential work and which one is the best for your specific case.

The script below will run the simulation at 298 K (25°C). 

.. code:: python

    import json

    import ase
    from ase.constraints import FixBondLengths
    from ase.data import vdw_radii
    from ase.io import read
    from mace.calculators import mace_mp
    from numba import get_num_threads, set_num_threads

    from flames.gcmc import GCMC
    from flames.lennard_jones import CustomLennardJones
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

    # Load the adsorbate structure
    adsorbate: ase.Atoms = read("ch4.xyz")  # type: ignore

    c = FixBondLengths([[0, 1], [0, 2], [0, 3], [0, 4]])
    adsorbate.set_constraint(c)

    Temperature = 298.0  # in Kelvin
    pressure = 1_000_000  # in Pa = 1 bar
    MCSteps = 10_000
    MDSteps = 10_000

    print(
        f"Running GCMC simulation for pressure: {pressure:.2f} Pa at temperature: {Temperature:.2f} K"
    )

    gcmc = GCMC(
        model=calc1,
        framework_atoms=framework,
        adsorbate_atoms=adsorbate,
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
        LLM=False,
        move_weights={
            "insertion": 0.25,
            "deletion": 0.25,
            "translation": 0.25,
            "rotation": 0,
            "reinsertion": 0.25,
        },
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


----

Breaking down the input script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The script starts by importing the necessary libraries and modules, including ASE, MACE, and FLAMES. It then sets the number of threads to use for parallel processing.

The `calc1` variable is initialized with the `CustomLennardJones` calculator using the DREIDING force field parameters,
while `calc2` is initialized with the `mace_mp` calculator for the MD simulation.

The `for` loop runs the GCMC simulation for a specified number of steps, followed by an NPT MD simulation using the MACE calculator.
Note that for the `GCMC` simulation, the `model` parameter is set to `calc1`, while for the `npt` method, the `calculator` parameter is set to `calc2`.