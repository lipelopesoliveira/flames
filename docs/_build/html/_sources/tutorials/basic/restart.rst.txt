============================
Restarting Simulations
============================


Restarting a Widom Simulation
=============================

It is possible to restart a Widom simulation in two scenarions. First, if the previous simulation fail and you want to continue it from the last saved point.
Second, if you believe that the number of steps that you have run is not enough to reach the convergence and you want to continue the simulation from the last saved point.

Since all moves in the Widom method are virtual moves, the only information that is loaded is the interaction energy for each step. The new simulation is completly independent of the previous one,
and the previous data will only be used to calculate the average enthaly of adsorption and the Henry coefficient.

On the ``flames/examples/Basic/1-Restart_Widom`` folder you will find the ``cif`` file for the Mg-MOF-74. The simulation will be executed with the machine learning potential MACE, using the ``medium-0b2`` pre-trained foundation model with ``D3(0)`` dispersion correction.

The script below will run the Widom simulation at 298 K (25°C).

.. code:: python

    import os

    # Hide UserWarning and RuntimeWarning messages
    import warnings

    import ase
    import torch
    from ase.data import vdw_radii
    from ase.io import read
    from mace.calculators import mace_mp

    from flames.widom import Widom

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

    # Load the adsorbate structure
    adsorbate: ase.Atoms = read(AdsorbatePath)  # type: ignore

    Temperature = 298.0

    NSteps = 3000

    widom = Widom(
        model=model,
        framework_atoms=framework,
        adsorbate_atoms=adsorbate,
        temperature=Temperature,
        device=device,
        vdw_radii=vdw_radii,
        debug=False,
        output_to_file=True,
        random_seed=42,
        cutoff_radius=6.0,
        automatic_supercell=True,
    )

    widom.logger.print_header()

    widom.restart()

    widom.run(NSteps)
    widom.logger.print_summary()
    widom.save_results()


The `widom.restart()` method will try to read the `int_energy_0.00000.npy` file, which is the last saved file from the previous simulation.

On the `Output.out` file will be printed the message:

.. code:: none

    Restarting simulation from step 171...

    ===========================================================================
    Restart file requested.
    Loaded state with 324 total atoms.
    Current total energy: -2351.243 eV
    Current number of adsorbates: 0
    Current average binding energy: 0.000 kJ/mol
    ===========================================================================


----

Restarting a GCMC Simulation
============================

It is possible to restart a existing GCMC simulation in two ways.

First, you can restart a failed simulation from the last saved point, continuing it from where it left off.
This is useful if the simulation was interrupted or if you want to extend the simulation time.

Second, you start a new simulation using the specific configuration of a previous simulation as the initial configuration for the new one.

Restarting a failed simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this example we will evaluate the MFI zeolite with the `TraPPE-Zeo`, a transferable potential for phase equilibria force field for all-silica zeolites [1]_.

On the ``flames/examples/Basic/4-Restart_fail`` folder you will find the ``cif`` file for the MFI [2]_.

.. note:: Since this model is a united-atom model, the hydrogen is not explicitly included in the structure, and are save on the `Trajectory.traj` file only for visualization purposes. Because of that, there is no point in using rotation moves in this simulation, so we will set the rotation move weight to zero.

The script below will run the simulation at 298 K (25°C) and 1 bar. 

.. code:: python

    import json

    import ase
    from ase.data import vdw_radii
    from ase.io import read

    from flames.gcmc import GCMC
    from flames.lennard_jones import CustomLennardJones
    from flames.utilities import read_cif

    with open("TraPPE_zeo.json", "r") as f:
        lj_params = json.loads(f.read())

    calc = CustomLennardJones(lj_params, vdw_cutoff=12.8, shifted=True)

    # Load the framework structure
    framework: ase.Atoms = read_cif("MFI.cif")

    # Load the adsorbate structure
    adsorbate: ase.Atoms = read("ch4.xyz")

    gcmc = GCMC(
        model=calc,
        framework_atoms=framework,
        adsorbate_atoms=adsorbate,
        temperature=298.15,
        pressure=1e5,
        device="cpu",
        vdw_radii=vdw_radii,
        debug=False,
        output_to_file=True,
        cutoff_radius=12.0,
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

    gcmc.restart()

    gcmc.run(10000 - gcmc.base_iteration)
    gcmc.logger.print_summary()

    gcmc.save_results()

----

Breaking down the input script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `gcmc.restart()` method will try to read the `total_ads_100000.00000.npy`, `total_energy_100000.00000.npy`, `uptake_100000.00000.npy`, and `Trajectory_100000.00000.traj` files, which are the last saved files from the previous simulation.
The `gcmc.base_iteration` variable will be set to the last iteration of the previous simulation, and the new simulation will continue from that point, ensuring that you obtain the desired number of steps. 

In the `Output.out` file will be printed the message:

.. code:: none


    Restarting simulation from step 2100...

    ===========================================================================
    Restart file requested.
    Loaded state with 2304 total atoms.
    Current total energy: 33166.818 eV
    Current number of adsorbates: 0
    Current average binding energy: 0.000 kJ/mol
    ===========================================================================


    ===========================================================================

    Restarting GCMC simulation from previous configuration...

    Loaded state with 2414 total atoms.

    Current total energy: 33163.019 eV
    Current number of adsorbates: 22
    Current average binding energy: -16.661 kJ/mol

    Current steps are: 2100

    ===========================================================================


If these files are not found, the code will print a warning message and will start a new simulation from scratch.

.. code:: none

    Restarting simulation from step 0...

    ===========================================================================
    Restart file requested.
    Loaded state with 2304 total atoms.
    Current total energy: 33166.818 eV
    Current number of adsorbates: 0
    Current average binding energy: 0.000 kJ/mol
    ===========================================================================

    ============================================================================
    WARNING: Error occurred while loading trajectory file:
    This is not an ULM formatted file.
    Cannot load the last state of the simulation.
    This is likely due to empty or corrupted trajectory file.
    Simulation will start from scratch.
    ============================================================================



Restarting from a previous configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On the ``flames/examples/Basic/5-Restart_state`` folder you will find the ``Trajectory.traj`` file from a previous simulation.
You can use this file to restart a new simulation from the last configuration of the previous one.


The script below will read the last configuration from the `Trajectory.traj` file and will use it as the initial configuration for a new simulation.

.. code:: python

    import json

    import ase
    from ase.data import vdw_radii
    from ase.io import read

    from flames.gcmc import GCMC
    from flames.lennard_jones import CustomLennardJones
    from flames.utilities import read_cif

    with open("TraPPE_zeo.json", "r") as f:
        lj_params = json.loads(f.read())

    calc = CustomLennardJones(lj_params, vdw_cutoff=12.8, shifted=True)

    # Load the framework structure
    framework: ase.Atoms = read_cif("MFI.cif")

    # Load the adsorbate structure
    adsorbate: ase.Atoms = read("ch4.xyz")

    gcmc = GCMC(
        model=calc,
        framework_atoms=framework,
        adsorbate_atoms=adsorbate,
        temperature=298.15,
        pressure=1e5,
        device="cpu",
        vdw_radii=vdw_radii,
        debug=False,
        output_to_file=True,
        cutoff_radius=12.0,
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

    gcmc.load_state('Trajectory.traj')

    gcmc.run(10000)
    gcmc.logger.print_summary()

    gcmc.save_results()


Note that the `gcmc.load_state('Trajectory.traj')` method will read the last configuration from the `Trajectory.traj` file and will use it as the initial configuration for the new simulation.
You are starting a new simulation, so the `gcmc.base_iteration` variable will be set to zero, and the new simulation will run for the specified number of steps.
The state that you are loading does not need to be generated in the same conditions of the new simulation that you are running, but it is recommended that the conditions are similar to ensure a smooth transition between the two simulations.
You can use this to start the simulation from a configuration that is closer to the equilibrium state, which can help to reduce the equilibration time of the new simulation.
Or to start the simulation from the last state of the previous point in a isotherm simulation, which can help to reduce the equilibration time of the new simulation and ensure a smooth transition between the two points.

On the `Output.out` file will be printed the message:

.. code:: none

    ===========================================================================

    Restarting GCMC simulation from previous configuration...

    Loaded state with 2394 total atoms.

    Current total energy: 33163.932 eV
    Current number of adsorbates: 18
    Current average binding energy: -15.467 kJ/mol

    Current steps are: 0

    ===========================================================================

Always check if that is what you want.

.. note:: The state file don't necessarily need to be a `Trajectory.traj` file, it can be any ASE readable file format, such as `xyz`, `cif`, `pdb`, etc.


References
~~~~~~~~~~

.. [1] Bai, P., Tsapatsis, M. and Siepmann, J.I., 2013. TraPPE-zeo: Transferable potentials for phase equilibria force field for all-silica zeolites. The Journal of Physical Chemistry C, 117(46), pp.24375-24387. https://pubs.acs.org/doi/10.1021/jp4074224
.. [2] Van Koningsveld, H., Van Bekkum, H. and Jansen, J.C., 1987. On the location and disorder of the tetrapropylammonium (TPA) ion in zeolite ZSM-5 with improved framework accuracy. Structural Science, 43(2), pp.127-132. https://journals.iucr.org/paper?S0108768187098173