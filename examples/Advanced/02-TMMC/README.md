# Hybrid TMMC-MD Simulation Example

This folder contains an example of a hybrid TMMC-MD simulation using the FLAMES package.

There are two scripts provided: `run_tmmc_serial.py` and `run_tmmc_independent.py`. The first script runs a series of GCMC simulations followed by MD simulations for a range of molecule counts in sequence, while the second script runs the same simulations for a specific number of molecules.

To run the simulation of each molecule sequentially, you can use the following commands:

```bash
python run_tmmc_serial.py --ExternalPressure 100000 --ExternalTemperature 298.0 --MDSteps 50000 --MCSteps 50000 --MDProdSteps 300000 --TMMCSteps 1000 --EquilibrationCycles 10 --MaxMolecules 45
```

This script will run the simulations for molecule counts from 0 to 45, saving the results in separate output folders for each molecule count. Each new state will continue from the previous state, allowing for a more efficient simulation.

To run each molecule independently, you can use the following command, replacing `<NMolecules>` with the desired number of molecules:

```bash

python run_tmmc_independent.py --ExternalPressure 100000 --ExternalTemperature 298.0 --MDSteps 100000 --MCSteps 100000 --MDProdSteps 300000 --TMMCSteps 1000 --EquilibrationCycles 10 --NMolecules <NMolecules>
```

This script will run the simulation for the specified number of molecules, starting it from scratch. In this case, the simulation of each N molecules can be run in parallel, allowing for a more efficient use of computational resources. However, it will be necessary more equilibration steps for each molecule count, since the simulation will not continue from the previous state.

## About the parameters

The ``ExternalPressure`` and ``ExternalTemperature`` parameters only control the molecular dynamics part of the simulation.
The simulation will perform 10 cycles (``EquilibrationCycles``) of 50000 Monte Carlo steps (``MCSteps``) followed by 50000 molecular dynamics steps (``MDSteps``) for each molecule count. After the equilibration cycles, the simulation will run 300000 production molecular dynamics steps (``MDProdSteps``), saving a frame of the system at regular intervals (each 100 steps, generating 3000 frames) and them 1000 TMMC steps (``TMMCSteps``), that consists of virtual insertion and deletion steps for each frame saved on the production trajectory.

THESE ARE JUST EXAMPLE PARAMETER VALUES! 

These parameters should be adjusted according to the system being simulated and the desired accuracy of the results. Please check the relevant literature for guidance on appropriate parameter values for your specific system or perform convergence tests to determine the optimal values for your simulation.