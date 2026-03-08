"""Tests for notebook plotting helpers."""

from __future__ import annotations

import numpy as np
import pytest

from open_rcs.model_types import GeometryData, RcsComputationResult
from open_rcs.notebook_ui import _build_2d_cuts, _expand_full_polar_if_needed, build_plotly_figures


def _geometry_stub() -> GeometryData:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    triangle_indices = np.array([[0, 1, 2]], dtype=int)
    return GeometryData(
        x=vertices[:, 0].copy(),
        y=vertices[:, 1].copy(),
        z=vertices[:, 2].copy(),
        x_points=vertices[:, 0].copy(),
        y_points=vertices[:, 1].copy(),
        z_points=vertices[:, 2].copy(),
        n_vertices=3,
        facet_numbers=np.array([1], dtype=int),
        node1=np.array([1], dtype=int),
        node2=np.array([2], dtype=int),
        node3=np.array([3], dtype=int),
        illumination_flag_mode=0,
        illumination_flags=np.array([1], dtype=int),
        resistivity_values=np.array([0.0], dtype=float),
        n_triangles=1,
        vertex_indices=triangle_indices.copy(),
        vertex_coordinates=vertices.copy(),
    )


def _simulation_result_stub() -> RcsComputationResult:
    phi_values = np.array([0.0, 180.0, 360.0], dtype=float)
    theta_values = np.array([60.0, 90.0, 120.0], dtype=float)
    phi_grid_deg = np.repeat(phi_values[:, None], theta_values.size, axis=1)
    theta_grid_deg = np.repeat(theta_values[None, :], phi_values.size, axis=0)
    rcs_theta_db = np.arange(phi_values.size * theta_values.size, dtype=float).reshape(
        phi_values.size, theta_values.size
    )
    rcs_phi_db = rcs_theta_db + 100.0
    zero_grid = np.zeros_like(phi_grid_deg)
    return RcsComputationResult(
        mode="Monostatic",
        input_model="cube.stl",
        wavelength_m=0.03,
        phi_grid_deg=phi_grid_deg,
        theta_grid_deg=theta_grid_deg,
        rcs_theta_db=rcs_theta_db,
        rcs_phi_db=rcs_phi_db,
        direction_cosine_u_grid=zero_grid.copy(),
        direction_cosine_v_grid=zero_grid.copy(),
        direction_cosine_w_grid=zero_grid.copy(),
        geometry=_geometry_stub(),
    )


def test_build_2d_cuts_returns_distinct_theta_and_phi_sweeps() -> None:
    """Validate canonical theta and phi cuts for canonical sample selection."""
    simulation_result = _simulation_result_stub()

    theta_cut, phi_cut = _build_2d_cuts(simulation_result)

    np.testing.assert_allclose(theta_cut.angle_values_deg, np.array([60.0, 90.0, 120.0]))
    np.testing.assert_allclose(theta_cut.rcs_theta_db, np.array([0.0, 1.0, 2.0]))
    assert theta_cut.fixed_axis == "phi"
    assert theta_cut.fixed_angle_deg == pytest.approx(0.0)

    np.testing.assert_allclose(phi_cut.angle_values_deg, np.array([0.0, 180.0, 360.0]))
    np.testing.assert_allclose(phi_cut.rcs_theta_db, np.array([1.0, 4.0, 7.0]))
    assert phi_cut.fixed_axis == "theta"
    assert phi_cut.fixed_angle_deg == pytest.approx(90.0)


def test_expand_full_polar_only_mirrors_full_theta_half_plane() -> None:
    """Verify theta polar expansion only applies to full 0-180 theta sweeps."""
    theta_angles = np.array([0.0, 90.0, 180.0], dtype=float)
    theta_values = np.array([1.0, 2.0, 3.0], dtype=float)
    phi_values = np.array([4.0, 5.0, 6.0], dtype=float)

    expanded_angles, expanded_theta, expanded_phi = _expand_full_polar_if_needed(
        "theta",
        theta_angles,
        theta_values,
        phi_values,
    )
    np.testing.assert_allclose(expanded_angles, np.array([0.0, 90.0, 180.0, 270.0]))
    np.testing.assert_allclose(expanded_theta, np.array([1.0, 2.0, 3.0, 2.0]))
    np.testing.assert_allclose(expanded_phi, np.array([4.0, 5.0, 6.0, 5.0]))

    partial_angles = np.array([60.0, 90.0, 120.0], dtype=float)
    unchanged_angles, _, _ = _expand_full_polar_if_needed("theta", partial_angles, theta_values, phi_values)
    np.testing.assert_allclose(unchanged_angles, partial_angles)

    phi_sweep_angles = np.array([0.0, 180.0, 360.0], dtype=float)
    unchanged_phi_angles, _, _ = _expand_full_polar_if_needed("phi", phi_sweep_angles, theta_values, phi_values)
    np.testing.assert_allclose(unchanged_phi_angles, phi_sweep_angles)


def test_build_plotly_figures_uses_two_polar_subplots_for_two_axis_sweeps(tmp_path) -> None:
    """Verify polar plot uses two subplots when both theta and phi sweeps vary."""
    pytest.importorskip("plotly")
    simulation_result = _simulation_result_stub()
    dummy_stl_path = tmp_path / "dummy.stl"

    fig_2d, _fig_3d = build_plotly_figures(
        simulation_result,
        dummy_stl_path,
        chart_mode="polar",
        mesh_vertices=simulation_result.geometry.vertex_coordinates,
        mesh_faces=simulation_result.geometry.vertex_indices,
    )

    assert len(fig_2d.data) == 4
    assert fig_2d.data[0]["theta"][-1] == pytest.approx(120.0)
    assert fig_2d.data[2]["theta"][-1] == pytest.approx(360.0)
    assert fig_2d.data[0]["line"]["color"] == "#1f77b4"
    assert fig_2d.data[1]["line"]["color"] == "#d62728"
    assert fig_2d.data[2]["line"]["color"] == "#1f77b4"
    assert fig_2d.data[3]["line"]["color"] == "#d62728"
    assert hasattr(fig_2d.layout, "polar2")
