import json
import os

import numpy as np
from ase.io import read
from mace.calculators import MACECalculator

from flames import VERSION
from flames.gcmc import GCMC

MOFS_PATH = os.path.dirname(__file__) + "/mofs/"
ADSORBATES_PATH = os.path.dirname(__file__) + "/adsorbates/"
MODELS_PATH = os.path.dirname(__file__) + "/models/"


# -----------------------------
# SIMPLE GCMC RUN AND RESTART
# -----------------------------
def test_gcmc_run(tmpdir):
    vdw_radii = [0.0, 0.38, 2.5, 0.86, 0.53, 1.01, 0.88, 0.86, 0.89, 0.82, 2.5, 1.15, 1.28, 1.53]
    uptake_list = [10, 10, 10, 10, 10, 10, 10, 10, 10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]
    total_energy_list = [
        -48151.039763837936,
        -48151.039763837936,
        -48151.039763837936,
        -48151.17240166944,
        -48151.17240166944,
        -48151.164280432786,
        -48151.164280432786,
        -48151.164280432786,
        -48151.164280432786,
        -47684.786049783674,
        -47684.786049783674,
        -47684.786049783674,
        -47684.786049783674,
        -47684.786049783674,
        -47684.786049783674,
        -47684.786049783674,
        -47684.786049783674,
        -47684.786049783674,
        -47685.2232901233,
        -47685.733143471596,
    ]
    total_ads_list = [
        2343.1960250360717,
        2343.1960250360717,
        2343.1960250360717,
        2343.0633872045655,
        2343.0633872045655,
        2343.0715084412222,
        2343.0715084412222,
        2343.0715084412222,
        2343.0715084412222,
        2341.6123890903327,
        2341.6123890903327,
        2341.6123890903327,
        2341.6123890903327,
        2341.6123890903327,
        2341.6123890903327,
        2341.6123890903327,
        2341.6123890903327,
        2341.6123890903327,
        2341.1751487507063,
        2340.6652954024103,
    ]
    ref_results = {
        "absolute_uptake": {
            "% wt": {
                "mean": 10.165924235784082,
                "sd": 0.5511625465173858,
            },
            "cm^3 STP/cm^3": {
                "mean": 139.29836609823388,
                "sd": 7.552293367892886,
            },
            "cm^3 STP/gr": {
                "mean": 126.482731461905,
                "sd": 6.85747235038698,
            },
            "mg/g": {
                "mean": 101.65924235784082,
                "sd": 5.511625465173858,
            },
            "mol/kg": {
                "mean": 5.643033160365922,
                "sd": 0.3059464594277931,
            },
            "nmol": {
                "mean": 9.45,
                "sd": 0.5123475193977356,
            },
        },
        "enthalpy": {
            "kJ_mol": {
                "mean": 138.3047059340562,
                "sd": 0.0,
            },
        },
        "equilibration": {
            "LLM": False,
            "ac_time": 4,
            "average": 9.45,
            "equilibrated": True,
            "t0": 0,
            "uncertainty": 0.5123475193977356,
            "uncorr_samples": 5,
        },
        "excess_uptake": {
            "% wt": {
                "mean": 10.165924235784082,
                "sd": 0.5511625465173858,
            },
            "cm^3 STP/cm^3": {
                "mean": 139.29836609823388,
                "sd": 7.552293367892886,
            },
            "cm^3 STP/gr": {
                "mean": 126.482731461905,
                "sd": 6.85747235038698,
            },
            "mg/g": {
                "mean": 101.65924235784082,
                "sd": 5.511625465173858,
            },
            "mol/kg": {
                "mean": 5.643033160365922,
                "sd": 0.3059464594277931,
            },
            "nmol": {
                "mean": 9.45,
                "sd": 0.5123475193977356,
            },
        },
        "simulation": {
            "code_version": VERSION,
            "fugacity_Pa": 3098.534471109497,
            "fugacity_coefficient": 0.9995272487449991,
            "move_weights": {
                "deletion": 0.2,
                "insertion": 0.2,
                "reinsertion": 0.2,
                "rotation": 0.2,
                "translation": 0.2,
            },
            "n_steps": 20,
            "pressure_Pa": 3100.0,
            "random_seed": 10,
            "temperature_K": 298.15,
        },
    }
    framework = read(MOFS_PATH + "MOF-303_5xH2O.xsf")
    n_framework = len(framework)
    adsorbate = read(ADSORBATES_PATH + "H2O.xyz")
    model = MACECalculator(
        model_paths=MODELS_PATH + "MOF-303_mace.model", device="cpu", default_dtype="float64"
    )
    gcmc = GCMC(
        model=model,
        framework_atoms=framework,
        adsorbate_atoms=adsorbate,
        temperature=298.15,
        pressure=3100.0,
        device="cpu",
        vdw_radii=vdw_radii,
        vdw_factor=1.15,
        save_frequency=1,
        output_folder=tmpdir,
        random_seed=10,
        criticalTemperature=647.096,
        criticalPressure=22.064e6,
        acentricFactor=0.3449,
        automatic_supercell=False,
    )
    gcmc.set_adsorbate(adsorbate, n_adsorbates=5, adsorbate_energy=adsorbate.info["total_energy"])
    gcmc.insert_adsorbates(5)
    assert abs(gcmc.adsorbate_energy - adsorbate.info["total_energy"]) < 1e-12
    assert gcmc.n_adsorbate_atoms == 3
    assert gcmc.n_adsorbates == 10

    gcmc.run(20)
    assert gcmc.n_adsorbates == 9
    assert len(gcmc.current_system) == n_framework + 3 * 4
    assert gcmc.uptake_list == uptake_list
    np.testing.assert_allclose(gcmc.total_energy_list, total_energy_list, rtol=1e-2)
    np.testing.assert_allclose(gcmc.total_ads_list, total_ads_list, rtol=1e-2)
    np.testing.assert_array_equal(np.load(str(tmpdir) + "/uptake_3100.00000.npy"), uptake_list)
    np.testing.assert_allclose(
        np.load(str(tmpdir) + "/total_energy_3100.00000.npy"), total_energy_list, rtol=1e-2
    )
    np.testing.assert_allclose(
        np.load(str(tmpdir) + "/total_ads_3100.00000.npy"), total_ads_list, rtol=1e-2
    )

    gcmc.save_results()
    results = json.load(open(str(tmpdir) + "/results_298.15_3100.0.json"))
    results["simulation"].pop("enlapsed_time_hours")
    assert results == ref_results

    gcmc = GCMC(
        model=model,
        framework_atoms=framework,
        adsorbate_atoms=adsorbate,
        temperature=298.15,
        pressure=3100.0,
        device="cpu",
        vdw_radii=vdw_radii,
        vdw_factor=1.15,
        save_frequency=1,
        output_folder=tmpdir,
        random_seed=10,
        criticalTemperature=647.096,
        criticalPressure=22.064e6,
        acentricFactor=0.3449,
        automatic_supercell=False,
    )
    gcmc.set_adsorbate(adsorbate, n_adsorbates=5, adsorbate_energy=adsorbate.info["total_energy"])
    gcmc.restart()
    assert gcmc.base_iteration == 20
    assert gcmc.n_adsorbates == 9
    assert gcmc.uptake_list == uptake_list
    np.testing.assert_allclose(gcmc.total_energy_list, total_energy_list, rtol=1e-2)
    np.testing.assert_allclose(gcmc.total_ads_list, total_ads_list, rtol=1e-2)
