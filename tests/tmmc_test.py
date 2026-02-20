import json
import os

import numpy as np
from ase.io import read
from mace.calculators import MACECalculator

from flames import VERSION
from flames.tmmc import TMMC

MOFS_PATH = os.path.dirname(__file__) + "/mofs/"
ADSORBATES_PATH = os.path.dirname(__file__) + "/adsorbates/"
MODELS_PATH = os.path.dirname(__file__) + "/models/"


# -----------------------------
# SIMPLE TMMC RUN AND RESTART
# -----------------------------
def test_tmmc_run(tmpdir):
    vdw_radii = [0.0, 0.38, 2.5, 0.86, 0.53, 1.01, 0.88, 0.86, 0.89, 0.82, 2.5, 1.15, 1.28, 1.53]
    ins_energy_list = [0.062041, 0.98445, -0.210626, 0.322975, 1.504674]
    del_energy_list = [0.569542, 0.718969, 0.950904, 0.569542, 0.718969]
    ref_results = {
        "simulation": {
            "code_version": VERSION,
            "random_seed": 10,
            "temperature_K": 298.15,
            "n_steps": 5,
        }
    }
    framework = read(MOFS_PATH + "MOF-303_5xH2O.xsf")
    adsorbate = read(ADSORBATES_PATH + "H2O.xyz")
    model = MACECalculator(
        model_paths=MODELS_PATH + "MOF-303_mace.model", device="cpu", default_dtype="float64"
    )
    tmmc = TMMC(
        model=model,
        framework_atoms=framework,
        adsorbate_atoms=adsorbate,
        temperature=298.15,
        pressure=1e5,
        device="cpu",
        vdw_radii=vdw_radii,
        vdw_factor=1.15,
        save_frequency=1,
        output_folder=tmpdir,
        random_seed=10,
    )
    tmmc.set_adsorbate(adsorbate, n_adsorbates=5, adsorbate_energy=adsorbate.info["total_energy"])

    assert abs(tmmc.adsorbate_energy - adsorbate.info["total_energy"]) < 1e-12
    assert tmmc.n_adsorbate_atoms == 3
    assert tmmc.n_adsorbates == 5

    tmmc.run(5)
    np.testing.assert_allclose(tmmc.total_ins_energy_list, ins_energy_list, rtol=1e-2)
    np.testing.assert_allclose(tmmc.total_del_energy_list, del_energy_list, rtol=1e-2)
    np.testing.assert_allclose(tmmc.volume_list, [framework.get_volume()] * 5)
    np.testing.assert_allclose(
        np.load(str(tmpdir) + "/ins_ernergy_0005.npy"), ins_energy_list, rtol=1e-2
    )
    np.testing.assert_allclose(
        np.load(str(tmpdir) + "/del_ernergy_0005.npy"), del_energy_list, rtol=1e-2
    )
    np.testing.assert_allclose(
        np.load(str(tmpdir) + "/volume_0005.npy"), [framework.get_volume()] * 5
    )

    tmmc.save_results()
    results = json.load(open(str(tmpdir) + "/results_298.15_0005.json"))
    results["simulation"].pop("enlapsed_time_hours")
    assert results == ref_results

    tmmc = TMMC(
        model=model,
        framework_atoms=framework,
        adsorbate_atoms=adsorbate,
        temperature=298.15,
        pressure=1e5,
        device="cpu",
        vdw_radii=vdw_radii,
        vdw_factor=1.15,
        save_frequency=1,
        output_folder=tmpdir,
        random_seed=10,
    )
    tmmc.set_adsorbate(adsorbate, n_adsorbates=5, adsorbate_energy=adsorbate.info["total_energy"])
    tmmc.restart()
    assert tmmc.base_iteration == 5
    np.testing.assert_allclose(tmmc.total_ins_energy_list, ins_energy_list, rtol=1e-2)
    np.testing.assert_allclose(tmmc.total_del_energy_list, del_energy_list, rtol=1e-2)
    np.testing.assert_allclose(tmmc.volume_list, [framework.get_volume()] * 5)
