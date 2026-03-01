from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

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

REPO_ROOT = Path(__file__).resolve().parents[1]
PLATE_STL = REPO_ROOT / "stl_models" / "plate.stl"
DUMMY_MATERIAL_PATH = str(REPO_ROOT / "matrl.txt")


def _assert_rcs_db_parity(
    reference_db: np.ndarray,
    candidate_db: np.ndarray,
    *,
    strong_signal_floor_db: float = -200.0,
    strong_rtol: float = 1e-6,
    strong_atol: float = 1e-6,
    weak_atol_db: float = 3.0,
) -> None:
    """Assert parity while handling log-scale sensitivity near numerical floor."""
    assert reference_db.shape == candidate_db.shape

    strong_mask = (reference_db > strong_signal_floor_db) | (candidate_db > strong_signal_floor_db)
    weak_mask = ~strong_mask

    if np.any(strong_mask):
        assert np.allclose(
            candidate_db[strong_mask],
            reference_db[strong_mask],
            rtol=strong_rtol,
            atol=strong_atol,
        )

    if np.any(weak_mask):
        weak_diff_db = np.abs(candidate_db[weak_mask] - reference_db[weak_mask])
        assert np.all(weak_diff_db <= weak_atol_db)


def _make_angle_sweep() -> AngleSweep:
    return AngleSweep(
        phi_start_deg=0.0,
        phi_stop_deg=90.0,
        phi_step_deg=45.0,
        theta_start_deg=0.0,
        theta_stop_deg=90.0,
        theta_step_deg=45.0,
    )


def _make_monostatic_config(*, use_numba: bool) -> MonostaticSimulationConfig:
    return MonostaticSimulationConfig(
        input_model="plate.stl",
        frequency_hz=10.0e9,
        correlation_distance_m=0.0,
        standard_deviation_m=0.0,
        incident_polarization=0,
        angle_sweep=_make_angle_sweep(),
        material=MaterialConfig(
            resistivity_mode=0.2,
            material_path=DUMMY_MATERIAL_PATH,
        ),
        use_numba=use_numba,
    )


def _make_bistatic_config(*, use_numba: bool) -> BistaticSimulationConfig:
    return BistaticSimulationConfig(
        input_model="plate.stl",
        frequency_hz=10.0e9,
        correlation_distance_m=0.0,
        standard_deviation_m=0.0,
        incident_polarization=0,
        angle_sweep=_make_angle_sweep(),
        incident_theta_deg=35.0,
        incident_phi_deg=40.0,
        material=MaterialConfig(
            resistivity_mode=0.2,
            material_path=DUMMY_MATERIAL_PATH,
        ),
        use_numba=use_numba,
    )


def test_build_geometry_from_stl_consistent_shapes() -> None:
    geometry_data = build_geometry_from_stl(PLATE_STL, rs_value=0.2)

    assert geometry_data.n_vertices > 0
    assert geometry_data.n_triangles > 0
    assert geometry_data.vertex_coordinates.shape == (geometry_data.n_vertices, 3)
    assert geometry_data.vertex_indices.shape == (geometry_data.n_triangles, 3)
    assert int(np.min(geometry_data.vertex_indices)) >= 1
    assert int(np.max(geometry_data.vertex_indices)) <= geometry_data.n_vertices
    assert geometry_data.resistivity_values.shape[0] == geometry_data.n_triangles
    assert np.allclose(geometry_data.resistivity_values, 0.2)


def test_simulate_monostatic_returns_finite_rcs_grid() -> None:
    config = _make_monostatic_config(use_numba=False)
    geometry_data = build_geometry_from_stl(PLATE_STL, config.material.resistivity_mode)
    result = simulate_monostatic(config, geometry_data)

    assert result.mode == "Monostatic"
    assert result.rcs_theta_db.shape == (3, 3)
    assert result.rcs_phi_db.shape == (3, 3)
    assert np.isfinite(result.rcs_theta_db).all()
    assert np.isfinite(result.rcs_phi_db).all()


def test_simulate_bistatic_returns_finite_rcs_grid() -> None:
    config = _make_bistatic_config(use_numba=False)
    geometry_data = build_geometry_from_stl(PLATE_STL, config.material.resistivity_mode)
    result = simulate_bistatic(config, geometry_data)

    assert result.mode == "Bistatic"
    assert result.rcs_theta_db.shape == (3, 3)
    assert result.rcs_phi_db.shape == (3, 3)
    assert np.isfinite(result.rcs_theta_db).all()
    assert np.isfinite(result.rcs_phi_db).all()


@pytest.mark.skipif(not rf.NUMBA_AVAILABLE, reason="Numba is not available in this environment.")
def test_monostatic_numba_matches_python_path() -> None:
    python_config = _make_monostatic_config(use_numba=False)
    numba_config = _make_monostatic_config(use_numba=True)
    geometry_data = build_geometry_from_stl(PLATE_STL, python_config.material.resistivity_mode)

    python_result = simulate_monostatic(python_config, geometry_data)
    numba_result = simulate_monostatic(numba_config, geometry_data)

    _assert_rcs_db_parity(python_result.rcs_theta_db, numba_result.rcs_theta_db)
    _assert_rcs_db_parity(python_result.rcs_phi_db, numba_result.rcs_phi_db)


@pytest.mark.skipif(not rf.NUMBA_AVAILABLE, reason="Numba is not available in this environment.")
def test_bistatic_numba_matches_python_path() -> None:
    python_config = _make_bistatic_config(use_numba=False)
    numba_config = _make_bistatic_config(use_numba=True)
    geometry_data = build_geometry_from_stl(PLATE_STL, python_config.material.resistivity_mode)

    python_result = simulate_bistatic(python_config, geometry_data)
    numba_result = simulate_bistatic(numba_config, geometry_data)

    _assert_rcs_db_parity(python_result.rcs_theta_db, numba_result.rcs_theta_db)
    _assert_rcs_db_parity(python_result.rcs_phi_db, numba_result.rcs_phi_db)
