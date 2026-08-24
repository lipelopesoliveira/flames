![GitHub-Mark-Light](docs/source/_static/img/logo_horizontal_light.png#gh-light-mode-only)
![GitHub-Mark-Dark ](docs/source/_static/img/logo_horizontal_dark.png#gh-dark-mode-only)

# FLAMES - Flexible Lattice Adsorption by Monte Carlo Engine Simulation

[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/license/mit)
[![Paper](https://img.shields.io/badge/ChemArxiv-10.26434/15004623/v1-blue)](https://chemrxiv.org/doi/full/10.26434/chemrxiv.15004623/v1)
[![This project supports Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)](https://python.org/downloads)
[![Main](https://github.com/lipelopesoliveira/flames/actions/workflows/main.yml/badge.svg)](https://github.com/lipelopesoliveira/flames/actions/workflows/main.yml)

The FLAMES is a general purpose adsorption simulation toolbox built around the Atomic Simulation Environment (ASE), which provides tools for molecular simulations and adsorption studies using machine learning potentials, classical force fields, and other advanced techniques.

## Requirements

0. Python >= 3.10
1. pymatgen
2. numpy
3. scipy
4. simplejson
5. ase
6. gemmi

The Python dependencies are most easily satisfied using a conda
([anaconda](https://www.anaconda.com/distribution)/[miniconda](https://docs.conda.io/en/latest/miniconda.html))
installation by running

```Shell
conda env create --file environment.yml
```

## Installation

Soon you will be able to install the FLAMES package using pip:

> [!WARNING]
>
> The code is not yet published on PyPI, so you need to import it manually from the GitHub repository.

You can use FLAMES by manually importing it using the `sys` module, as exemplified below:

```python
# importing module
import sys
 
# appending a path
sys.path.append('{PATH_TO_FLAMES}/flames')

from flames.gcmc import GCMC
```

Just remember to change the `{PATH_TO_FLAMES}` to the directory where you downloaded the FLAMES package.

## Feature Overview

The FLAMES package includes modules for performing Grand Canonical Monte Carlo (GCMC) simulations and Widom insertion tests. It is designed to work with the ASE (Atomic Simulation Environment) framework and supports various machine learning potentials.

## Usage

To use the FLAMES package, you can import the necessary classes from the `flames` module. For example, to perform GCMC simulations, you can use the `GCMC` class:

```python
import os

import ase
import torch
from ase.data import vdw_radii
from ase.io import read
from mace.calculators import mace_mp

from flames.gcmc import GCMC

device = "cuda" if torch.cuda.is_available() else "cpu"

FrameworkPath = "mg-mof-74.cif"
AdsorbatePath = "co2.xyz"

model = mace_mp(
    model="medium-0b2",
    dispersion=True,
    damping="zero",   # choices: ["zero", "bj", "zerom", "bjm"]
    dispersion_xc="pbe",
    default_dtype="float32",
    device=device,
)

# Load the framework structure
framework: ase.Atoms = read(FrameworkPath)  # type: ignore

# Load the adsorbate structure
adsorbate: ase.Atoms = read(AdsorbatePath)  # type: ignore

Temperature = 298.0  # in Kelvin
pressure = 100_000  # in Pa = 1 bar
MCSteps = 30_000


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
    vdw_factor=0.6,
    save_frequency=1,
    debug=True,
    output_to_file=True,
    criticalTemperature=304.1282,
    criticalPressure=7377300.0,
    acentricFactor=0.22394,
    move_weights={
            "insertion": 0.25,
            "deletion": 0.25,
            "translation": 0.25,
            "rotation": 0.25,
        },
    random_seed=42,
    cutoff_radius=6.0,
    automatic_supercell=True,
)


gcmc.logger.print_header()

gcmc.run(MCSteps)

gcmc.logger.print_summary()

gcmc.save_results()
```

## Examples

You can find example scripts in the [examples](https://github.com/lipelopesoliveira/flames/tree/main/examples) directory. These scripts demonstrate how to use the FLAMES package for various tasks, such as running GCMC simulations and performing Widom insertion tests, etc.

On the [online documentation](https://lipelopesoliveira.github.io/flames/docs/_build/html/index.html) you can find more information about the examples.

## Citation

If you find **flames** useful in your research, please consider citing the following paper:

> F. L. Oliveira, H-D Saßnick, and G. Maurin,
> _FLAMES – A flexible and extensible code for Monte Carlo simulations of nanoporous materials_
> ChemRxiv 2026
> _10.26434/chemrxiv.15004623/v1_ [DOI](https://chemrxiv.org/doi/full/10.26434/chemrxiv.15004623/v1)
