from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from open_rcs import rcs_functions as rf


def test_polarization_and_angle_helper_functions() -> None:
    label_tm, electric_theta_tm, electric_phi_tm = rf.get_polarization(0)
    label_te, electric_theta_te, electric_phi_te = rf.get_polarization(1)
    assert label_tm == "TM-z"
    assert electric_theta_tm == 1 + 0j
    assert electric_phi_tm == 0 + 0j
    assert label_te == "TE-z"
    assert electric_theta_te == 0 + 0j
    assert electric_phi_te == 1 + 0j
    with pytest.raises(ValueError):
        rf.get_polarization(99)

    theta, phi = rf.spherical_angles(0.2, 0.3, 0.9)
    assert np.isfinite(theta)
    assert np.isfinite(phi)

    (
        bistatic_theta,
        bistatic_phi,
        cosine_phi,
        sine_phi,
        sine_theta,
        cosine_theta,
    ) = rf.bi_spherical_angles(0.2, -0.3, 0.8)
    assert np.isfinite(bistatic_theta)
    assert np.isfinite(bistatic_phi)
    assert np.isclose(cosine_phi, math.cos(bistatic_phi))
    assert np.isclose(sine_phi, math.sin(bistatic_phi))
    assert np.isclose(cosine_theta, math.sqrt(max(0.0, 1.0 - sine_theta**2)))


def test_sampling_and_global_angle_builders() -> None:
    triangle_areas, alpha, beta, normals, edges, phi_count, theta_count = rf.calculate_values(
        pstart=0.0,
        pstop=0.0,
        delp=0.0,
        tstart=0.0,
        tstop=0.0,
        delt=0.0,
        ntria=2,
        rad=np.pi / 180.0,
    )
    assert phi_count == 1
    assert theta_count == 1
    assert triangle_areas.shape == (2,)
    assert alpha.shape == (2,)
    assert beta.shape == (2,)
    assert normals.shape == (2, 3)
    assert edges.shape == (2, 3)

    (
        _area,
        _alpha,
        _beta,
        _normals,
        _edges,
        bistatic_phi_count,
        bistatic_theta_count,
        *_,
    ) = rf.bi_calculate_values(
        pstart=0.0,
        pstop=0.0,
        delp=0.0,
        tstart=0.0,
        tstop=0.0,
        delt=0.0,
        ntria=2,
        rad=np.pi / 180.0,
        fii=10.0,
        thetai=20.0,
    )
    assert bistatic_phi_count == 1
    assert bistatic_theta_count == 1

    u_grid = np.zeros((1, 1), dtype=float)
    v_grid = np.zeros((1, 1), dtype=float)
    w_grid = np.zeros((1, 1), dtype=float)
    (
        u_grid,
        v_grid,
        w_grid,
        direction_vector,
        theta_projection_u,
        theta_projection_v,
        theta_projection_w,
        direction_u,
        direction_v,
        direction_w,
    ) = rf.global_angles(
        u_grid=u_grid,
        v_grid=v_grid,
        w_grid=w_grid,
        theta_radians=0.4,
        phi_radians=0.2,
        phi_index=0,
        theta_index=0,
    )
    assert np.isclose(direction_vector[0], direction_u)
    assert np.isclose(direction_vector[1], direction_v)
    assert np.isclose(direction_vector[2], direction_w)
    assert np.isclose(u_grid[0, 0], direction_u)
    assert np.isclose(v_grid[0, 0], direction_v)
    assert np.isclose(w_grid[0, 0], direction_w)
    assert np.isfinite(theta_projection_u)
    assert np.isfinite(theta_projection_v)
    assert np.isfinite(theta_projection_w)


def test_reflection_coefficients_and_area_integral_branches() -> None:
    reflection_perpendicular, reflection_parallel = rf.reflection_coefficients(
        rs=0.2,
        index=0,
        th2=0.3,
        thri=0.1,
        phrii=0.2,
        alpha=0.0,
        beta=0.0,
        freq=10.0e9,
        matrl=[],
    )
    assert np.isfinite(reflection_perpendicular)
    assert np.isfinite(reflection_parallel)

    n_terms = 5
    taylor_threshold = 1e-5
    area = 0.5
    incident_amplitude = 1.0
    test_cases = [
        (1e-8, 2e-2),
        (1e-8, 2e-8),
        (2e-2, 1e-8),
        (2e-2, 2e-2 + 1e-8),
        (2e-2, -3e-2),
    ]
    for phase_p, phase_q in test_cases:
        area_integral = rf.calculate_ic(
            phase_p=phase_p,
            phase_q=phase_q,
            phase_origin=0.03,
            taylor_terms=n_terms,
            triangle_area=area,
            incident_amplitude=incident_amplitude,
            taylor_threshold=taylor_threshold,
        )
        assert np.isfinite(np.real(area_integral))
        assert np.isfinite(np.imag(area_integral))


def test_refl_coeff_total_internal_reflection_branch() -> None:
    _gamma_parallel, _gamma_perpendicular, _theta_transmitted, tir_flag = rf.refl_coeff(
        er1=4.0,
        mr1=1.0,
        er2=1.0,
        mr2=1.0,
        thetai=1.0,
    )
    assert tir_flag == 1


def test_final_plot_single_sample_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rf, "RESULTS_DIR", tmp_path)
    theta_grid_deg = np.array([[15.0]], dtype=float)
    phi_grid_deg = np.array([[20.0]], dtype=float)
    rcs_theta_db = np.array([[1.0]], dtype=float)
    rcs_phi_db = np.array([[0.5]], dtype=float)
    direction_u = np.array([[0.0]], dtype=float)
    direction_v = np.array([[0.0]], dtype=float)

    plot_path = rf.final_plot(
        phi_sample_count=1,
        theta_sample_count=1,
        phi_grid_deg=phi_grid_deg,
        wavelength_m=0.03,
        theta_grid_deg=theta_grid_deg,
        rcs_min_db=-10.0,
        rcs_max_db=10.0,
        rcs_theta_db=rcs_theta_db,
        rcs_phi_db=rcs_phi_db,
        direction_cosine_u_grid=direction_u,
        direction_cosine_v_grid=direction_v,
        timestamp="unit-test",
        input_model="plate.stl",
        mode="Monostatic",
    )
    assert Path(plot_path).exists()
