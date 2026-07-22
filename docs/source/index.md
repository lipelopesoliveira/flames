```{image} _static/img/logo_horizontal_light.png
:align: center
:class: only-light
```

```{image} _static/img/logo_horizontal_dark.png
:align: center
:class: only-dark
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Getting started

installation
overview
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Tutorials: Basic

tutorials/basic_widom
tutorials/basic_gcmc
tutorials/basic_restart
tutorials/basic_isotherm
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Tutorials: Intermediate

```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Tutorials: Advanced

tutorials/advanced_hybrid_gcmc_md_mixed_calc
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Theoretical Background

theoretical_background/widom
theoretical_background/gcmc
theoretical_background/tmmc
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Simulator Description

input_description/base_simulator
input_description/gcmc
input_description/widom
input_description/tmmc
```

```{toctree}
:hidden:
:caption: Development

CHANGELOG
CONTRIBUTING
License <https://raw.githubusercontent.com/lipelopesoliveira/flames/main/LICENSE>
GitHub Repository <https://github.com/lipelopesoliveira/flames>
```

# Welcome to FLAMES documentation

FLAMES is a general purpose adsorption simulation toolbox built around the Atomic Simulation Environment (ASE), which provides tools for molecular simulations and adsorption studies using machine learning potentials, classical force fields, and other advanced techniques.

Here you will find tutorials and examples on how to install and use FLAMES to run different adsorption simulations on porous structures. These are aimed for new users and people with more experience on molecular modeling.

## First steps

If you are not familiar with Python or molecular simulations, maybe it is best to start here, otherwise you can just skip and go to the next sections.

- [Installation](installation): Instructions on how to install FLAMES and its dependencies.
- [Overview](overview): A brief overview of FLAMES features, code organization and capabilities.

## Tutorials

The tutorials section contains step-by-step guides on how to use FLAMES for different types of simulations. We recommend starting with the basic tutorials and then moving on to more advanced topics as you become more comfortable with the code.

### Basic

- [Widom Insertion Method](tutorials/basic_widom): A tutorial on how to perform Widom insertion simulations using FLAMES.