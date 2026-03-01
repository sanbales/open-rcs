from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
PLATE_STL = REPO_ROOT / "stl_models" / "plate.stl"
MATERIAL_PATH = REPO_ROOT / "matrl.txt"


def _single_point_sweep():
    from open_rcs import AngleSweep

    return AngleSweep(
        phi_start_deg=0.0,
        phi_stop_deg=0.0,
        phi_step_deg=1.0,
        theta_start_deg=0.0,
        theta_stop_deg=0.0,
        theta_step_deg=1.0,
    )


def _grid_sweep():
    from open_rcs import AngleSweep

    return AngleSweep(
        phi_start_deg=0.0,
        phi_stop_deg=90.0,
        phi_step_deg=45.0,
        theta_start_deg=0.0,
        theta_stop_deg=90.0,
        theta_step_deg=45.0,
    )


def _write_pec_material_file(path: Path, n_triangles: int) -> None:
    rows = [f"PEC,triangle_{index}" for index in range(n_triangles)]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_run_monostatic_and_bistatic_generate_artifacts(tmp_path: Path, monkeypatch) -> None:
    from open_rcs import (
        BistaticSimulationConfig,
        MaterialConfig,
        MonostaticSimulationConfig,
        build_geometry_from_stl,
        rcs_functions as rf,
        run_bistatic,
        run_monostatic,
    )

    monkeypatch.setattr(rf, "RESULTS_DIR", tmp_path / "results")
    geometry_data = build_geometry_from_stl(PLATE_STL, rs_value=0.2)

    monostatic_config = MonostaticSimulationConfig(
        input_model="plate.stl",
        frequency_hz=10.0e9,
        correlation_distance_m=0.0,
        standard_deviation_m=0.0,
        incident_polarization=0,
        angle_sweep=_grid_sweep(),
        material=MaterialConfig(resistivity_mode=0.2, material_path=str(MATERIAL_PATH)),
        use_numba=False,
    )
    bistatic_config = BistaticSimulationConfig(
        input_model="plate.stl",
        frequency_hz=10.0e9,
        correlation_distance_m=0.0,
        standard_deviation_m=0.0,
        incident_polarization=0,
        angle_sweep=_grid_sweep(),
        incident_theta_deg=35.0,
        incident_phi_deg=40.0,
        material=MaterialConfig(resistivity_mode=0.2, material_path=str(MATERIAL_PATH)),
        use_numba=False,
    )

    monostatic_result = run_monostatic(monostatic_config, geometry_data)
    bistatic_result = run_bistatic(bistatic_config, geometry_data)

    assert Path(monostatic_result.figure_path).exists()
    assert Path(monostatic_result.plot_path).exists()
    assert Path(monostatic_result.data_path).exists()
    assert Path(bistatic_result.figure_path).exists()
    assert Path(bistatic_result.plot_path).exists()
    assert Path(bistatic_result.data_path).exists()


def test_simulate_progress_callback_and_material_lookup_paths(tmp_path: Path) -> None:
    from open_rcs import (
        AngleSweep,
        BistaticSimulationConfig,
        MaterialConfig,
        MonostaticSimulationConfig,
        build_geometry_from_stl,
        rcs_functions as rf,
        simulate_bistatic,
        simulate_monostatic,
    )

    geometry_data = build_geometry_from_stl(PLATE_STL, rs_value=rf.MATERIAL_SPECIFIC)
    material_file = tmp_path / "materials.txt"
    _write_pec_material_file(material_file, int(geometry_data.n_triangles))

    sweep = AngleSweep(
        phi_start_deg=0.0,
        phi_stop_deg=90.0,
        phi_step_deg=45.0,
        theta_start_deg=0.0,
        theta_stop_deg=90.0,
        theta_step_deg=45.0,
    )
    material = MaterialConfig(
        resistivity_mode=float(rf.MATERIAL_SPECIFIC),
        material_path=str(material_file),
    )
    monostatic_config = MonostaticSimulationConfig(
        input_model="plate.stl",
        frequency_hz=10.0e9,
        correlation_distance_m=0.0,
        standard_deviation_m=0.0,
        incident_polarization=0,
        angle_sweep=sweep,
        material=material,
        use_numba=True,
    )
    bistatic_config = BistaticSimulationConfig(
        input_model="plate.stl",
        frequency_hz=10.0e9,
        correlation_distance_m=0.0,
        standard_deviation_m=0.0,
        incident_polarization=0,
        angle_sweep=sweep,
        incident_theta_deg=35.0,
        incident_phi_deg=40.0,
        material=material,
        use_numba=True,
    )

    monostatic_callbacks: list[tuple[int, int]] = []
    bistatic_callbacks: list[tuple[int, int]] = []
    monostatic_result = simulate_monostatic(
        monostatic_config,
        geometry_data,
        progress_callback=lambda done, total: monostatic_callbacks.append((done, total)),
        progress_update_stride=2,
    )
    bistatic_result = simulate_bistatic(
        bistatic_config,
        geometry_data,
        progress_callback=lambda done, total: bistatic_callbacks.append((done, total)),
        progress_update_stride=2,
    )

    assert monostatic_callbacks
    assert bistatic_callbacks
    assert monostatic_callbacks[-1][0] == monostatic_callbacks[-1][1]
    assert bistatic_callbacks[-1][0] == bistatic_callbacks[-1][1]
    assert np.isfinite(monostatic_result.rcs_theta_db).all()
    assert np.isfinite(monostatic_result.rcs_phi_db).all()
    assert np.isfinite(bistatic_result.rcs_theta_db).all()
    assert np.isfinite(bistatic_result.rcs_phi_db).all()
