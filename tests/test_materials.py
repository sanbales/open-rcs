from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from open_rcs import rcs_functions as rf


def test_material_roundtrip_and_pec_reflection() -> None:
    rows = [
        "PEC,plate\n",
        "Composite,skin,2.0,1.0,0.01,0.02,0.03\n",
    ]
    table = rf.convert_material_textlist_to_list(rows)

    assert table[0][rf.TYPE] == "PEC"
    assert table[1][rf.TYPE] == "Composite"
    assert table[1][rf.LAYERS] == [2.0, 1.0, 0.01, 0.02, 0.03]

    reflection_perpendicular, reflection_parallel = rf.reflection_coefficients(
        rs=float(rf.MATERIAL_SPECIFIC),
        index=0,
        th2=0.1,
        thri=0.2,
        phrii=0.3,
        alpha=0.0,
        beta=0.0,
        freq=10.0e9,
        matrl=table,
    )
    assert reflection_perpendicular == -1
    assert reflection_parallel == -1


def test_get_entries_from_material_file_validates_count(tmp_path: Path) -> None:
    material_path = tmp_path / "materials.txt"
    rf.save_list_in_file([["PEC", "facet_0"]], str(material_path))

    entries = rf.get_entries_from_material_file(1, str(material_path))
    assert len(entries) == 1

    with pytest.raises(ValueError):
        rf.get_entries_from_material_file(2, str(material_path))


def test_reflection_models_return_finite_values() -> None:
    test_cases = [
        ["Composite", "single", [2.5, 0.02, 1.0, 0.01, 1.5]],
        ["Composite Layer on PEC", "single-pec", [2.5, 0.02, 1.0, 0.01, 1.5]],
        [
            "Multiple Layers",
            "multi",
            [2.2, 0.01, 1.0, 0.00, 1.0],
            [3.1, 0.02, 1.2, 0.01, 0.5],
        ],
        [
            "Multiple Layers on PEC",
            "multi-pec",
            [2.2, 0.01, 1.0, 0.00, 1.0],
            [3.1, 0.02, 1.2, 0.01, 0.5],
        ],
    ]
    for material_entry in test_cases:
        reflection_perpendicular, reflection_parallel = rf.get_reflection_coeff_from_material(
            thri=0.35,
            phrii=0.45,
            alpha=0.1,
            beta=-0.2,
            freq=10.0e9,
            matrlLine=material_entry,
        )
        assert np.isfinite(np.real(reflection_perpendicular))
        assert np.isfinite(np.imag(reflection_perpendicular))
        assert np.isfinite(np.real(reflection_parallel))
        assert np.isfinite(np.imag(reflection_parallel))


def test_coordinate_transforms_and_phase_helpers_are_consistent() -> None:
    spherical = np.array([1.0, 0.6, -0.4], dtype=float)
    cartesian = rf.spher2cart(spherical)
    reconstructed_spherical = rf.cart2spher(cartesian)
    assert np.allclose(reconstructed_spherical, spherical, atol=1e-12)

    transform = rf.rotation_transform_matrix(alpha=0.2, beta=-0.1)
    local_spherical = rf.spherical_global_to_local(spherical, transform)
    assert local_spherical.shape == (3,)
    assert np.isfinite(local_spherical).all()

    gamma_parallel, gamma_perpendicular, theta_transmitted, tir_flag = rf.refl_coeff(
        er1=1.0,
        mr1=1.0,
        er2=2.4 - 0.05j,
        mr2=1.1 - 0.02j,
        thetai=0.4,
    )
    assert np.isfinite(np.real(gamma_parallel))
    assert np.isfinite(np.real(gamma_perpendicular))
    assert np.isfinite(theta_transmitted)
    assert tir_flag in (0, 1)

    x = np.array([0.0, 1.0, 0.0], dtype=float)
    y = np.array([0.0, 0.0, 1.0], dtype=float)
    z = np.array([0.0, 0.0, 0.0], dtype=float)
    vind = np.array([[1, 2, 3]], dtype=np.int64)
    wave_number = 2.0
    direction_u, direction_v, direction_w = 0.2, 0.3, 0.9
    incident_u, incident_v, incident_w = -0.1, 0.4, 0.8
    direct_phase = rf.phase_vertex_triangle(
        x, y, z, vind, wave_number, 0, direction_u, direction_v, direction_w
    )
    precomputed_phase = rf.phase_vertex_triangle_precomputed(
        wave_number=wave_number,
        phase_p_vectors=np.array([[0.0, -1.0, 0.0]], dtype=float),
        phase_q_vectors=np.array([[1.0, -1.0, 0.0]], dtype=float),
        phase_origin_vectors=np.array([[0.0, 1.0, 0.0]], dtype=float),
        triangle_index=0,
        direction_u=direction_u,
        direction_v=direction_v,
        direction_w=direction_w,
    )
    assert np.allclose(precomputed_phase, direct_phase, atol=1e-12)

    direct_bistatic_phase = rf.bi_phase_vertex_triangle(
        x,
        y,
        z,
        vind,
        wave_number,
        0,
        direction_u,
        direction_v,
        direction_w,
        incident_u,
        incident_v,
        incident_w,
    )
    precomputed_bistatic_phase = rf.bi_phase_vertex_triangle_precomputed(
        wave_number=wave_number,
        phase_p_vectors=np.array([[0.0, -1.0, 0.0]], dtype=float),
        phase_q_vectors=np.array([[1.0, -1.0, 0.0]], dtype=float),
        phase_origin_vectors=np.array([[0.0, 1.0, 0.0]], dtype=float),
        triangle_index=0,
        observation_direction_u=direction_u,
        observation_direction_v=direction_v,
        observation_direction_w=direction_w,
        incident_direction_u=incident_u,
        incident_direction_v=incident_v,
        incident_direction_w=incident_w,
    )
    assert np.allclose(precomputed_bistatic_phase, direct_bistatic_phase, atol=1e-12)
