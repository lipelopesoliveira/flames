=========================
Hybrid TMMC/MD Simulation
=========================


Brief introduction about TMMC simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The transition matrix Monte Carlo (TMMC) scheme is based on the determination of the system's transition probabilities going from one macrostate (referring to a fixed number of adsorbates inside the porous material) to another.
The change of the total energy for a virtual insertion and virtual deletion moves is recorded and used to construct the transition matrix (more details are given in the theoretical background) from which all ensemble properties can be derived.
The TMMC simulation needs to be run in combination with a Grand Canonical Monte Carlo (GCMC) or Molecular Dynamics (MD) simulation to propagate the system. The simulation follows five steps:

1. Initiate the system with a fixed number of adsorbates in the pore;
2. Equilibrate the system using MC moves, MD moves, or a combination of both;
3. Generate a set of equilibrated configurations;
4. Perform virtual insertions and deletions on the equilibrated configurations to construct the transition matrix.
5. Analyze the results to obtain the macrostate probability distribution (MPD) and all ensemble properties (adsorption isotherm).

----

Step 1. Initializing the simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this example, we will investigate water adsorption in MOF-303, a promising candidate material for water harvesting due to its large water uptake at low relative pressures.
Specifically, the seeding molecules are stabilized by a slight reorientation of the ligands, allowing the water molecule to form strong hydrogen-bonds with the framework.

The MOF and adsorbate geometry can be found in ``flames/examples/Advanced/02-TMMC/MOF-303_symm.cif`` and ``flames/examples/Advanced/02-TMMC/H2O.xyz``, respectively.
For this example, we will use a basepoke MACE [1]_ machine learning potential is located at ``flames/examples/Advanced/02-TMMC/MOF-303_mace.model``.

.. warning:: In this example, we will only run a TMMC simulation for the macrostate of 5 molecules, to obtain the full picture every possible macrostates needs to be sufficiently covered.

.. code:: python

    import os
    import warnings

    import ase
    import torch
    from ase.data import vdw_radii
    from ase.io import read
    from mace.calculators import mace_mp

    from flames.tmmc import TMMC
    from flames.gcmc import GCMC
    from flames.adsorbate import Adsorbate

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # Only required for windows users

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MACECalculator(
        model_path="MOF-303_mace.model",
        device=device,
        default_dtype="float64"
        )

    FrameworkPath = "MOF-303_symm.cif"
    AdsorbatePath = "H2O.xyz"

    # Load the framework structure
    framework = read(FrameworkPath) 

    # Load the adsorbate structure
    adsorbate = Adsorbate(
        name="H2O",
        structure=read(AdsorbatePath),
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

    NMolecules = 5
    ExternalTemperature = 298.0  # in Kelvin
    # The pressure below is used only for the MD steps, the pressures for generating the isotherm will be used later
    ExternalPressure = 100_000  # in Pa = 1 bar 

    outputFolder = f'results_{ExternalTemperature:.2f}_{ExternalPressure:.2f}_{NMolecules:03d}'

    gcmc = GCMC(
        model=model,
        framework_atoms=framework,
        adsorbates=adsorbate,
        temperature=ExternalTemperature,
        pressure=ExternalPressure,
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

    gcmc.insert_adsorbates(ads_name=adsorbate.name, n_adsorbates=NMolecules, max_attempts=10000)


----

Breaking down the input script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The first block of the code is similar to the one used in the Widom insertion tutorial, where we import the necessary libraries, set up the device, load the model and read the framework and adsorbate structures.

Initializing the simulator
--------------------------

.. code:: python

    NMolecules = 5
    ExternalTemperature = 298.0  # in Kelvin
    # The pressure below is used only for the MD steps, the pressures for generating the isotherm will be used later
    ExternalPressure = 100_000  # in Pa = 1 bar 

    outputFolder = f'results_{ExternalTemperature:.2f}_{ExternalPressure:.2f}_{NMolecules:03d}'

    gcmc = GCMC(
        model=model,
        framework_atoms=framework,
        adsorbates=adsorbate,
        temperature=ExternalTemperature,
        pressure=ExternalPressure,
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

    gcmc.insert_adsorbates(ads_name=adsorbate.name, n_adsorbates=NMolecules, max_attempts=10000)

Here, we are starting the system with 5 water molecules ramdonly inserted in the MOF-303 framework. At this point, we also add the DFT energy of a single water molecule (``-467.837350``) to the input since the used model was not trained on single molecules or the bulk water phase.


2. Equilibrating the system
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    MDSteps = 5000

    MCSteps = 5000

    for i in range(10):
    gcmc.run(MCSteps)

    gcmc.md(
        nsteps=MDSteps,
        time_step=0.5,
        ensemble="NPT",
        thermostat="MTK",
        movie_interval=100
    )

Here we will run cycles of MC steps followed by MD steps to equilibrate the system. The number of MC and MD steps can be adjusted based on the system and the desired level of equilibration.
We will run 10 cycles of 5000 MC steps followed by 5000 MD steps, which should be sufficient to equilibrate the system. The MD simulation is run in the NPT ensemble with a time step of 0.5 fs and a MTK thermostat.


3. Generating equilibrated configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    md_traj = Trajectory(f"Trajectory_{NMolecules:04d}.traj", "w")

    gcmc.md(
        nsteps=300000,
        time_step=0.5,
        ensemble="NPT",
        thermostat="MTK",
        movie_interval=100,
        set_momenta=False,
        trajectory_file=md_traj
        )

Here we will generate the equilibrated configurations by running a longer MD simulation in the NPT ensemble. The trajectory will be saved to a file for later analysis.
We are saving the trajectory to a file named ``Trajectory_0005.traj`` for later analysis. The MD simulation is run for 300,000 steps with a time step of 0.5 fs and a MTK thermostat, saving 
the trajectory every 100 steps, generating a total of 3000 frames to run the TMMC simulation on.
The momenta are not reset at the beginning of the simulation to maintain the system's dynamics from the previous equilibration step.

.. warning:: For production calculations, the system needs to be well equilibrated for the simulations to be accurate. The number of MC and MD steps should be adjusted based on the system and the desired level of equilibration.

4. Running the TMMC simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    tmmc = TMMC(
        model,
        framework,
        adsorbate,
        temperature=ExternalTemperature,
        pressure=ExternalPressure,
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

Here 3000 virtual insertions and deletions will be performed on each of the 3000 frames generated in the previous step, generating in total 9.000.000 evaluations per macrostate, and 405.000.000 in total.
This will allow the construction of the transition matrix and allowing to calculate the macrostate probability distribution (MPD) and all ensemble properties.

After this step, three new files will be created in the output folder: ``del_ernergy_0005.npy``, ``ins_ernergy_0005.npy``, and ``volume_0005.npy``, containing the deletion energies, insertion energies and volumes, respectively.

.. warning:: For the results analysis it is necessary to have data for all macrostates, which is not the case in this example. The following code snippet is only for demonstration purposes. For the MOF-303, it is necessary to run from N=0 to N=44 to obtain the full adsorption isotherm.

Each macrostate can be executed in parallel, allowing to speed up the calculations.

In case you plan to run them serially (if you only have access to one GPU), you can start the each new step from the results of the previous one to speed up the equilibration.
For example, you can start the N=6 macrostate from the final configuration of the N=5 macrostate, and so on:

.. code:: python

    PreviousN = 5
    gcmc.load_state(os.path.join(f'results_{ExternalTemperature:.2f}_{ExternalPressure:.2f}_{PreviousN:03d}', 'Movies', 'Trajectory.traj'))


In the example folder, there are two python scripts, ``run_tmmc_paralell.py`` and ``run_tmmc_serial.py``. In the second one, each point is started from the final configuration of the previous macrostate, while in the first one, each macrostate runs independently.

5. Analyzing the output
~~~~~~~~~~~~~~~~~~~~~~~

The stored values now allow to calculate the macrostate probability distribution (MPD), given access to all ensemble properties. Specifically, we can calculate the adsorption isotherm as follows.


References
~~~~~~~~~~

.. [1] Batatia, I., Kovacs, D. P., Simm, G. N. C., Ortner, C., and Csanyi, G., 2022. MACE: Higher order equivariant message passing neural networks for fast and accurate force fields. Advances in Neural Information Processing Systems, edited by A. H. Oh, A. Agarwal, D. Belgrave, and K. Cho (2022). https://openreview.net/forum?id=YPpSngE-ZU