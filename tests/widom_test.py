import json
import os

import numpy as np
from ase.io import read
from mace.calculators import MACECalculator

from flames import VERSION
from flames.widom import Widom

MOFS_PATH = os.path.dirname(__file__) + "/mofs/"
ADSORBATES_PATH = os.path.dirname(__file__) + "/adsorbates/"
MODELS_PATH = os.path.dirname(__file__) + "/models/"


# -----------------------------
# SIMPLE WIDOM RUN AND RESTART
# -----------------------------
def test_widom_run(tmpdir):
    vdw_radii = [0.0, 0.38, 2.5, 0.86, 0.53, 1.01, 0.88, 0.86, 0.89, 0.82, 2.5, 1.15, 1.28, 1.53]
    int_energy_list = [0.0, -0.201905, -0.001457, -0.016145, -0.165637, -0.150568]
    ref_results = {
        "code_version": VERSION,
        "enthalpy_of_adsorption_std_kJ_mol-1": 0.0,
        "henry_coefficient_std_mol_kg-1_Pa-1": 0.0,
        "random_seed": 10,
        "temperature_K": 298.15,
        "total_insertions": 6,
    }
    enthalpy_ads_ref = -20.835212757381015
    henry_coeff_ref = 0.00023053295198257289
    framework = read(MOFS_PATH + "MOF-303_5xH2O.xsf")[:-15]
    adsorbate = read(ADSORBATES_PATH + "H2O.xyz")
    model = MACECalculator(
        model_paths=MODELS_PATH + "MOF-303_mace.model", device="cpu", default_dtype="float64"
    )
    widom = Widom(
        model=model,
        framework_atoms=framework,
        adsorbate_atoms=adsorbate,
        temperature=298.15,
        device="cpu",
        vdw_radii=vdw_radii,
        vdw_factor=1.15,
        output_folder=tmpdir,
        random_seed=10,
        automatic_supercell=False,
    )
    widom.set_adsorbate(adsorbate, n_adsorbates=0, adsorbate_energy=adsorbate.info["total_energy"])
    assert abs(widom.adsorbate_energy - adsorbate.info["total_energy"]) < 1e-12
    assert widom.n_adsorbate_atoms == 3
    assert widom.n_adsorbates == 0

    widom.run(5)
    np.testing.assert_allclose(widom.int_energy_list, int_energy_list, rtol=1e-2)
    np.testing.assert_allclose(
        np.load(str(tmpdir) + "/int_energy_0.00000.npy"), int_energy_list, rtol=1e-2
    )

    widom.save_results()
    results = json.load(open(str(tmpdir) + "/Widom_Results.json"))
    results.pop("enlapsed_time_hours")
    enthalpy_ads = results.pop("enthalpy_of_adsorption_kJ_mol-1")
    henry_coeff = results.pop("henry_coefficient_mol_kg-1_Pa-1")
    assert abs(enthalpy_ads - enthalpy_ads_ref) < 1.0e-3
    assert abs(henry_coeff - henry_coeff_ref) < 1.0e-3
    assert results == ref_results

    widom = Widom(
        model=model,
        framework_atoms=framework,
        adsorbate_atoms=adsorbate,
        temperature=298.15,
        device="cpu",
        vdw_radii=vdw_radii,
        vdw_factor=1.15,
        output_folder=tmpdir,
        random_seed=10,
        automatic_supercell=False,
    )
    widom.set_adsorbate(adsorbate, n_adsorbates=0, adsorbate_energy=adsorbate.info["total_energy"])
    widom.restart()
    assert widom.base_iteration == 6
    assert widom.n_adsorbates == 0
    np.testing.assert_allclose(widom.int_energy_list, int_energy_list, rtol=1e-2)
