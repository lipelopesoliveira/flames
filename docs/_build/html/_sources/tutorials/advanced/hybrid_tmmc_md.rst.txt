=========================
Hybrid TMMC/MD Simulation
=========================


Brief introduction about TMMC simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The transition matrix Monte Carlo (TMMC) scheme is based on the determination of the system's transition probabilities going from one macrostate (referring to the number of adsorbates inside the porous material) to another.
Thus, the change of the total energy for insertion and deletion moves is recorded and used to construct the transition matrix (more details are given in the theoretical background) from which all ensemble properties can be derived.
The TMMC simulation needs to be run in combination with a Grand Canonical Monte Carlo (GCMC) or Molecular Dynamics (MD) simulation to propagate the system.

----

Setting the stage
~~~~~~~~~~~~~~~~~

In this example, we will investigate water adsorption in MOF-303, a promising candidate material for water harvesting due to its large water uptake at low relative pressures.
Specifically, the seeding molecules are stabilized by a slight reorientation of the ligands, allowing the water molecule to form hydrogen-bonds with the framework.

The input system and the adsorbate geometry can be found in ``flames/tests/mofs/MOF-303_5xH2O.xsf`` and ``flames/tests/adsorbates/H2O.xyz``, respectively.
The used MACE [1]_ machine learning potential is located at ``flames/tests/models/MOF-303_mace.model``.

.. warning:: In this example, we will only run a TMMC simulation for the macrostate of 5 molecules, to obtain the full picture every possible macrostates needs to be sufficiently covered.

The script below will execute 5 MD runs at 298 K (25°C) and 1 bar followed by one virtual insertion and deletion step after each run.

.. code:: python

    import os

    # Hide UserWarning and RuntimeWarning messages
    import warnings

    import ase
    import torch
    from aim2dat.elements import get_atomic_radius
    from ase.io import read
    from mace.calculators import mace_mp

    from flames.tmmc import TMMC

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MACECalculator(model_path="tests/models/MOF-303_mace.model", device=device, default_dtype="float64")

    FrameworkPath = "tests/mofs/MOF-303_5xH2O.xsf"
    AdsorbatePath = "tests/adsorbates/H2O.xyz"

    # Load the framework structure
    framework: ase.Atoms = read(FrameworkPath)  # type: ignore

    # Load the adsorbate structure
    adsorbate: ase.Atoms = read(AdsorbatePath)  # type: ignore

    TTemperature = 298.0  # in Kelvin
    pressure = 100_000  # in Pa = 1 bar
    MDSteps = 100
    MCSteps = 1

    vdw_radii = [0.0]
    for i in range(1, 97):
        rad = get_atomic_radius(i, radius_type="chen_manz")
        if rad is None:
            vdw_radii.append(2.5)
        else:
            vdw_radii.append(rad)

    tmmc = TMMC(
        model,
        framework,
        adsorbate,
        temperature=TTemperature,
        pressure=pressure,
        device=device,
        vdw_radii=vdw_radii,
        vdw_factor=1.15,
        save_frequency=1,
        max_overlap_tries=10000,
    )
    tmmc.set_adsorbate(adsorbate, -467.837350, n_adsorbates=5)
    tmmc.logger.print_header()

    for i in range(5):
        tmmc.npt(
            MDSteps,
            set_momenta=i==0,
            mode="aniso_flex",
            driver="NoseHoover",
            output_interval=10,
            ttime=50.0 * units.fs,
            pfactor=20.0
        )
        tmmc.run(MCSteps)

    tmmc.logger.print_summary()
    tmmc.save_results()

----

Breaking down the input script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The first block of the code is similar to the one used in the Widom insertion tutorial, where we import the necessary libraries, set up the device, load the model and read the framework and adsorbate structures.

Initializing the simulation
---------------------------

.. code:: python

    TTemperature = 298.0  # in Kelvin
    pressure = 100_000  # in Pa = 1 bar
    MDSteps = 100
    MCSteps = 1

    vdw_radii = [0.0]
    for i in range(1, 97):
        rad = get_atomic_radius(i, radius_type="chen_manz")
        if rad is None:
            vdw_radii.append(2.5)
        else:
            vdw_radii.append(rad)

    tmmc = TMMC(
        model,
        framework,
        adsorbate,
        temperature=TTemperature,
        pressure=pressure,
        device=device,
        vdw_radii=vdw_radii,
        vdw_factor=1.15,
        save_frequency=1,
        max_overlap_tries=10000,
    )
    tmmc.set_adsorbate(adsorbate, -467.837350, n_adsorbates=5)
    tmmc.logger.print_header()

Herein, we use the ``aim2dat`` Python package, [2]_ obtaining the atomic radii defined by Chen and Manz, [3]_ to define the overlap criteria.

Since the starting configuration of the MOF already contains 5 water molecules, we use the ``set_adsorbate_function`` to overwrite the internal adsorbate counter.
At this point, we also add the DFT energy of a single water molecule (``-467.837350``) to the input since the used model was not trained on single molecules or the bulk water phase.




Running the simulation
----------------------

.. code:: python

    for i in range(5):
        tmmc.npt(
            MDSteps,
            set_momenta=i==0,
            mode="aniso_flex",
            driver="NoseHoover",
            output_interval=10,
            ttime=50.0 * units.fs,
            pfactor=20.0
        )
        tmmc.run(MCSteps)

    tmmc.logger.print_summary()
    tmmc.save_results()

The latter part of the script runs the MD and TMMC simulations within in a ``for`` loop and stores the results at the very end.

.. warning:: For production calculations, the system needs to be well equilibrated and the number of recorded insertion and deletion steps should be much larger.



Analyzing the output
~~~~~~~~~~~~~~~~~~~~

The main output obtained from the simulation is contained in the ``del_ernergy_0005.npy``, ``ins_ernergy_0005.npy``, and ``volume_0005.npy``, containing the deletion energies, insertion energies and volumes, respectively.
The stored values allow to calculate the macrostate probability distribution (MPD), given access to all ensemble properties (more details are given in the theoretical background).

References
~~~~~~~~~~

.. [1] Batatia, I., Kovacs, D. P., Simm, G. N. C., Ortner, C., and Csanyi, G., 2022. MACE: Higher order equivariant message passing neural networks for fast and accurate force fields. Advances in Neural Information Processing Systems, edited by A. H. Oh, A. Agarwal, D. Belgrave, and K. Cho (2022). https://openreview.net/forum?id=YPpSngE-ZU
.. [2] Saßnick, H.-D., Edzards J., Reents T., and Cocchi C., 2026. AIM2DAT: A Python-Based Automated Ab Initio Material Modeling and Data Analysis Toolkit. Electronic Structure, 8(3), 037001. https://doi.org/10.1088/2516-1075/ae8964
.. [3] Chen, T., and Manz T. A, 2019. A Collection of Forcefield Precursors for Metal–Organic Frameworks. RSC Advances, 9(63), 36492–36507. https://doi.org/10.1039/C9RA07327B
