========================
GCMC Adsorption Isotherm
========================

In this example we will run a adsorption isotherm of CO :sub:`2` on the Mg-MOF-74.

On the ``flames/examples/Basic/4-Rigid_GCMC_Isotherm`` folder you will find the ``cif`` file for the Mg-MOF-74 [1]_. The simulation will be executed with the machine learning potential MACE [2]_, using the ``medium-0b2`` pre-trained foundation model with ``D3(0)`` dispersion correction. 

.. warning:: This model is used here just as an example, you should look carefully into the mace documentation to undestand how the potential work and which one is the best for your specific case.


A adsorption isotherm is just a series of GCMC simulations at different pressures, and the results are plotted as a function of the pressure.
These simulations can be run independently, with each point starting from zero loading. This has the advantage of being able to run the simulations in parallel,
but it has the disadvantage of not being able to use the previous simulation as a starting point for the next one, which can be useful to speed up the convergence of the simulation.

One aditional way is to run the simulations sequentially, where each simulation starts from the last configuration of the previous simulation.
This has the advantage of being able to use the previous simulation as a starting point for the next one, which can be useful to speed up the convergence of the simulation,
but it has the disadvantage of not being able to run the simulations in parallel.

The script below will run the simulation sequentially at 298 K (25°C) with pressures ranging from 0.001 to 1 bar. 

.. code:: python

    import os

    # Hide UserWarning and RuntimeWarning messages
    import warnings

    import ase
    import torch
    from ase.data import vdw_radii
    from ase.io import read
    from mace.calculators import mace_mp

    from flames.gcmc import GCMC

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    FrameworkPath = "mg-mof-74.cif"
    AdsorbatePath = "co2.xyz"

    model = mace_mp(
        model="medium-0b2",
        dispersion=True,
        damping="zero",  # choices: ["zero", "bj", "zerom", "bjm"]
        dispersion_xc="pbe",
        default_dtype="float32",
        device=device,
    )

    # Load the framework structure
    framework: ase.Atoms = read(FrameworkPath)  # type: ignore

    # Load the adsorbate structure
    adsorbate: ase.Atoms = read(AdsorbatePath)  # type: ignore

    Temperature = 298.0  # in Kelvin
    pressure_list = [
        100,
        200,
        500,
        1000,
        2000,
        5000,
        10000,
        20000,
        50000,
        100000,
    ]  # Example pressure list in Pa

    MCSteps = 1000

    for i, pressure in enumerate(pressure_list):

        print(
            f"Running GCMC simulation for pressure: {pressure:.2f} Pa at temperature: {Temperature:.2f} K"
        )

        gcmc = GCMC(
            model=model,
            framework_atoms=framework,
            adsorbate_atoms=adsorbate,
            temperature=Temperature,
            pressure=pressure,
            device=device,
            vdw_radii=vdw_radii,
            debug=False,
            output_to_file=True,
            criticalTemperature=304.1282,
            criticalPressure=7377300.0,
            acentricFactor=0.22394,
            cutoff_radius=6.0,
            automatic_supercell=True,
        )

        gcmc.logger.print_header()

        if pressure > pressure_list[0]:
            print("Loading previous state for continuation...")
            output_dir = f"results_{Temperature:.2f}_{pressure_list[i-1]:.2f}"
            gcmc.load_state(os.path.join(output_dir, "Movies", "Trajectory.traj"))

        gcmc.run(MCSteps)
        gcmc.logger.print_summary()

        gcmc.save_results()

----

Breaking down the input script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The block below will automatically read the last configuration of the previous point and use it as a starting state for the current point, which can speedup the convergence considerably.

.. code:: python

    if pressure > pressure_list[0]:
            print("Loading previous state for continuation...")
            output_dir = f"results_{Temperature:.2f}_{pressure_list[i-1]:.2f}"
            gcmc.load_state(os.path.join(output_dir, "Movies", "Trajectory.traj"))


.. warning:: Here we are using only 1.000 steps as an example, but it is recommended that evaluate the best value for the simulations you are interested in, and check if in fact your results are equilibrated. Usually 100.000 is a good starting point, but it strongly depends on the system.

After finishing all the points you can plot the results using your preffered software.

References
~~~~~~~~~~

.. [1] Mason, J.A., Sumida, K., Herm, Z.R., Krishna, R. and Long, J.R., 2011. Evaluating metal-organic frameworks for post-combustion carbon dioxide capture via temperature swing adsorption. Energy & Environmental Science, 4(8), pp.3030-3040. https://pubs.rsc.org/en/content/articlelanding/2011/ee/c1ee01720a
.. [2] Batatia, I., Benner, P., Chiang, Y., Elena, A.M., Kovács, D.P., Riebesell, J., Advincula, X.R., Asta, M., Avaylon, M., Baldwin, W.J. and Berger, F., 2025. A foundation model for atomistic materials chemistry. The Journal of Chemical Physics, 163(18). https://pubs.aip.org/aip/jcp/article/163/18/184110/3372267/A-foundation-model-for-atomistic-materials