"""Tests for roughness and correlation guidance utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from open_rcs import build_geometry_from_stl, estimate_roughness_correlation_guidance
from open_rcs.model_types import GeometryData

REPO_ROOT = Path(__file__).resolve().parents[1]
PLATE_STL = REPO_ROOT / "stl_models" / "plate.stl"


def test_estimate_roughness_correlation_guidance_returns_bounded_values() -> None:
    """Verify guidance outputs remain within computed physical bounds."""
    geometry_data = build_geometry_from_stl(PLATE_STL, rs_value=0.0)
    guidance = estimate_roughness_correlation_guidance(
        frequency_hz=10.0e9,
        geometry_data=geometry_data,
    )

    assert guidance.wavelength_m > 0
    assert guidance.electrical_size_l_over_lambda >= 0
    assert guidance.median_edge_length_m > 0
    assert guidance.standard_deviation_bounds_m[0] <= guidance.suggested_standard_deviation_m
    assert guidance.suggested_standard_deviation_m <= guidance.standard_deviation_bounds_m[1]
    assert guidance.correlation_distance_bounds_m[0] <= guidance.suggested_correlation_distance_m
    assert guidance.suggested_correlation_distance_m <= guidance.correlation_distance_bounds_m[1]
    assert len(guidance.notes) >= 2


def test_estimate_roughness_correlation_guidance_invalid_frequency_raises() -> None:
    """Verify non-positive frequency inputs raise a validation error."""
    geometry_data = build_geometry_from_stl(PLATE_STL, rs_value=0.0)
    try:
        estimate_roughness_correlation_guidance(0.0, geometry_data)
    except ValueError as exc:
        assert "frequency_hz must be positive" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError for non-positive frequency.")


def test_estimate_roughness_correlation_guidance_small_target_note() -> None:
    """Verify notes include electrically-small guidance for tiny targets."""
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.0, 0.01, 0.0],
        ],
        dtype=float,
    )
    geometry_data = GeometryData(
        x=vertices[:, 0].copy(),
        y=vertices[:, 1].copy(),
        z=vertices[:, 2].copy(),
        x_points=vertices[:, 0].copy(),
        y_points=vertices[:, 1].copy(),
        z_points=vertices[:, 2].copy(),
        n_vertices=3,
        facet_numbers=np.array([1.0], dtype=float),
        node1=np.array([1], dtype=int),
        node2=np.array([2], dtype=int),
        node3=np.array([3], dtype=int),
        illumination_flag_mode=0,
        illumination_flags=np.array([1.0], dtype=float),
        resistivity_values=np.array([0.0], dtype=float),
        n_triangles=1,
        vertex_indices=np.array([[1, 2, 3]], dtype=np.int64),
        vertex_coordinates=vertices.copy(),
    )
    guidance = estimate_roughness_correlation_guidance(1.0e9, geometry_data)
    combined_notes = " ".join(guidance.notes)
    assert "electrically small" in combined_notes
    assert "relatively fine" in combined_notes
