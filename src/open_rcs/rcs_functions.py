from __future__ import annotations

import cmath
import math
from pathlib import Path
from typing import Final, cast

import numpy as np

from .constants import (
    RESULTS_DIR,
    SPECIFIC_MATERIAL,
    MaterialCode,
    MaterialEntryIndex,
    MaterialType,
)
from .materials import (
    cart2spher,
    compile_material_lookup_arrays,
    convert_material_textlist_to_list,
    get_entries_from_material_file,
    get_material_properties_from_file,
    get_reflection_coeff_from_material,
    load_material_catalog,
    refl_coeff,
    reflection_coefficients,
    rotation_transform_matrix,
    save_list_in_file,
    spher2cart,
    spherical_global_to_local,
)
from .model_types import GeometryData
from .plotting import (
    final_plot,
    generate_result_files,
    plot_limits,
    plot_parameters,
    plot_triangle_model,
    set_font_option,
)
from .stl_module import convert_stl

try:
    from numba import njit
except ImportError:  # pragma: no cover - optional acceleration dependency
    njit = None
NUMBA_AVAILABLE: Final[bool] = njit is not None
_NUMBA_PLACEHOLDER_INT_ARRAY = np.zeros(1, dtype=np.int32)
_NUMBA_PLACEHOLDER_FLOAT_ARRAY = np.zeros((1, 1), dtype=np.float64)

_MAT_CODE_PEC: Final[int] = int(MaterialCode.PEC)
_MAT_CODE_COMPOSITE: Final[int] = int(MaterialCode.COMPOSITE)
_MAT_CODE_COMPOSITE_ON_PEC: Final[int] = int(MaterialCode.COMPOSITE_ON_PEC)
_MAT_CODE_MULTI_LAYER_ON_PEC: Final[int] = int(MaterialCode.MULTI_LAYER_ON_PEC)

__all__ = [
    "cart2spher",
    "compile_material_lookup_arrays",
    "convert_material_textlist_to_list",
    "final_plot",
    "generate_result_files",
    "get_entries_from_material_file",
    "get_material_properties_from_file",
    "get_reflection_coeff_from_material",
    "load_material_catalog",
    "MaterialEntryIndex",
    "MaterialType",
    "plot_limits",
    "plot_parameters",
    "plot_triangle_model",
    "refl_coeff",
    "reflection_coefficients",
    "RESULTS_DIR",
    "rotation_transform_matrix",
    "save_list_in_file",
    "set_font_option",
    "SPECIFIC_MATERIAL",
    "spher2cart",
    "spherical_global_to_local",
]


def get_polarization(incident_polarization: int) -> tuple[str, complex, complex]:
    """Return polarization label and electric-field components."""
    if incident_polarization == 0:
        return "TM-z", 1 + 0j, 0 + 0j
    if incident_polarization == 1:
        return "TE-z", 0 + 0j, 1 + 0j
    raise ValueError("Invalid polarization value. Use 0 (TM-z) or 1 (TE-z).")


def get_standard_deviation(
    delstd: float, corel: float, wave: float
) -> tuple[float, float, float, float, float, int]:
    """Compute roughness-related constants used during scattering integration."""
    delsq = float(delstd) ** 2
    bk = 2 * np.pi / wave
    cfac1 = np.exp(-4 * bk**2 * delsq)
    cfac2 = 4 * np.pi * (bk * corel) ** 2 * delsq
    rad = np.pi / 180
    lt = 1e-5  # Taylor-series region
    nt = 5  # Number of terms in Taylor expansion
    return bk, cfac1, cfac2, rad, lt, nt


def extract_coordinates_data(rs_value: float) -> GeometryData:  # pragma: no cover
    """Load previously converted coordinates/facets and derive geometry arrays."""
    x, y, z, xpts, ypts, zpts, nverts = read_coordinates()
    nfc, node1, node2, node3, iflag, ilum, rs, ntria = read_facets(rs_value)
    vind = create_vind(node1, node2, node3)
    r = calculate_r(x, y, z, nverts)
    return GeometryData(
        x=x,
        y=y,
        z=z,
        x_points=xpts,
        y_points=ypts,
        z_points=zpts,
        n_vertices=nverts,
        facet_numbers=nfc,
        node1=node1,
        node2=node2,
        node3=node3,
        illumination_flag_mode=iflag,
        illumination_flags=ilum,
        resistivity_values=rs,
        n_triangles=ntria,
        vertex_indices=vind,
        vertex_coordinates=r,
    )


def build_geometry_from_stl(stl_path: str | Path, rs_value: float) -> GeometryData:
    """Build solver geometry directly from STL in memory."""
    coordinates, facets = convert_stl(stl_path, coordinates_output=None, facets_output=None)

    xpts = coordinates[:, 0]
    ypts = coordinates[:, 1]
    zpts = coordinates[:, 2]
    x = xpts.copy()
    y = ypts.copy()
    z = zpts.copy()
    nverts = len(xpts)

    nfc = facets[:, 0]
    node1 = facets[:, 1].astype(int)
    node2 = facets[:, 2].astype(int)
    node3 = facets[:, 3].astype(int)
    iflag = 0
    ilum = facets[:, 4]
    rs_values = np.full(ilum.shape, rs_value, dtype=float)
    ntria = len(node3)
    vind = create_vind(node1, node2, node3)
    r = calculate_r(x, y, z, nverts)
    return GeometryData(
        x=x,
        y=y,
        z=z,
        x_points=xpts,
        y_points=ypts,
        z_points=zpts,
        n_vertices=nverts,
        facet_numbers=nfc,
        node1=node1,
        node2=node2,
        node3=node3,
        illumination_flag_mode=iflag,
        illumination_flags=ilum,
        resistivity_values=rs_values,
        n_triangles=ntria,
        vertex_indices=vind,
        vertex_coordinates=r,
    )


def read_coordinates(path: str | Path = "./coordinates.txt"):  # pragma: no cover
    """Read coordinates file produced by :func:`convert_stl`."""
    coordinates = np.loadtxt(path)
    xpts = coordinates[:, 0]
    ypts = coordinates[:, 1]
    zpts = coordinates[:, 2]

    x = xpts.copy()
    y = ypts.copy()
    z = zpts.copy()
    nverts = len(xpts)
    return x, y, z, xpts, ypts, zpts, nverts


def read_facets(rs: float, path: str | Path = "./facets.txt"):  # pragma: no cover
    """Read facets and inject the selected resistivity for all faces."""
    facets = np.loadtxt(path)
    nfc = facets[:, 0]
    node1 = facets[:, 1].astype(int)
    node2 = facets[:, 2].astype(int)
    node3 = facets[:, 3].astype(int)
    iflag = 0
    ilum = facets[:, 4]
    rs_values = np.full(facets[:, 4].shape, rs, dtype=float)
    ntria = len(node3)
    return nfc, node1, node2, node3, iflag, ilum, rs_values, ntria


def create_vind(node1: np.ndarray, node2: np.ndarray, node3: np.ndarray) -> np.ndarray:
    """Create one-based triangle vertex index matrix."""
    return np.column_stack((node1, node2, node3)).astype(np.int64, copy=False)


def calculate_r(x: np.ndarray, y: np.ndarray, z: np.ndarray, nverts: int) -> np.ndarray:
    """Build Nx3 coordinates array used by geometric kernels."""
    if nverts != len(x):
        raise ValueError("nverts must match the length of x/y/z arrays.")
    return np.column_stack((x, y, z)).astype(np.double, copy=False)


def precompute_rotation_matrices(
    surface_alpha_angles: np.ndarray,
    surface_beta_angles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute per-triangle rotation matrices used by local/global transforms."""
    triangle_count = len(surface_alpha_angles)
    transform_z_all = np.zeros((triangle_count, 3, 3), dtype=float)
    transform_y_all = np.zeros((triangle_count, 3, 3), dtype=float)

    cos_alpha = np.cos(surface_alpha_angles)
    sin_alpha = np.sin(surface_alpha_angles)
    cos_beta = np.cos(surface_beta_angles)
    sin_beta = np.sin(surface_beta_angles)

    transform_z_all[:, 0, 0] = cos_alpha
    transform_z_all[:, 0, 1] = sin_alpha
    transform_z_all[:, 1, 0] = -sin_alpha
    transform_z_all[:, 1, 1] = cos_alpha
    transform_z_all[:, 2, 2] = 1.0

    transform_y_all[:, 0, 0] = cos_beta
    transform_y_all[:, 0, 2] = -sin_beta
    transform_y_all[:, 1, 1] = 1.0
    transform_y_all[:, 2, 0] = sin_beta
    transform_y_all[:, 2, 2] = cos_beta
    return transform_z_all, transform_y_all


def direction_cosines_from_precomputed(
    transform_z_all: np.ndarray,
    transform_y_all: np.ndarray,
    global_direction_vector: np.ndarray,
    triangle_index: int,
):
    """Compute local direction cosines using precomputed triangle transforms."""
    transform_z = transform_z_all[triangle_index]
    transform_y = transform_y_all[triangle_index]
    global_u = float(global_direction_vector[0])
    global_v = float(global_direction_vector[1])
    global_w = float(global_direction_vector[2])
    cosine_alpha = float(transform_z[0, 0])
    sine_alpha = float(transform_z[0, 1])
    cosine_beta = float(transform_y[0, 0])
    sine_beta = float(transform_y[2, 0])
    rotated_u = cosine_alpha * global_u + sine_alpha * global_v
    rotated_v = -sine_alpha * global_u + cosine_alpha * global_v
    local_u = cosine_beta * rotated_u - sine_beta * global_w
    local_v = rotated_v
    local_w = sine_beta * rotated_u + cosine_beta * global_w
    return local_u, local_v, local_w, transform_z, transform_y


def calculate_values(pstart, pstop, delp, tstart, tstop, delt, ntria, rad):
    def calculate_ip():
        if delp == 0:
            return int((pstop - pstart) + 1)
        else:
            return int((pstop - pstart) / delp + 1)

    def calculate_it():
        if delt == 0:
            return int((tstop - tstart) + 1)
        else:
            return int((tstop - tstart) / delt + 1)

    ip = calculate_ip()
    it = calculate_it()

    Area = np.empty(ntria, np.double)
    alpha = np.empty(ntria, np.double)
    beta = np.empty(ntria, np.double)
    N = np.empty([ntria, 3], np.double)
    d = np.empty([ntria, 3], np.double)

    return Area, alpha, beta, N, d, ip, it


def bi_calculate_values(pstart, pstop, delp, tstart, tstop, delt, ntria, rad, fii, thetai):
    # Compute trigonometric function values
    cpi = np.cos(fii * np.pi / 180.0)
    spi = np.sin(fii * np.pi / 180.0)
    sti = np.sin(thetai * np.pi / 180.0)
    cti = np.cos(thetai * np.pi / 180.0)

    # Compute vector values
    ui = sti * cpi
    vi = sti * spi
    wi = cti
    D0i = np.array([ui, vi, wi])

    uui = cti * cpi
    vvi = cti * spi
    wwi = -sti
    Ri = np.array([ui, vi, wi])

    def calculate_ip():
        if delp == 0:
            return int(pstop - pstart) + 1
        else:
            return int((pstop - pstart) / delp) + 1

    def calculate_it():
        if delt == 0:
            return int(tstop - tstart) + 1
        else:
            return int((tstop - tstart) / delt) + 1

    ip = calculate_ip()
    it = calculate_it()

    area = np.empty(ntria, np.double)
    alpha = np.empty(ntria, np.double)
    beta = np.empty(ntria, np.double)
    N = np.empty([ntria, 3], np.double)
    d = np.empty([ntria, 3], np.double)

    return area, alpha, beta, N, d, ip, it, cpi, spi, sti, cti, ui, vi, wi, D0i, uui, vvi, wwi, Ri


def global_angles(
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    w_grid: np.ndarray,
    theta_radians: float,
    phi_radians: float,
    phi_index: int,
    theta_index: int,
):
    direction_u = math.sin(theta_radians) * math.cos(phi_radians)
    direction_v = math.sin(theta_radians) * math.sin(phi_radians)
    direction_w = math.cos(theta_radians)
    u_grid[phi_index, theta_index] = direction_u
    v_grid[phi_index, theta_index] = direction_v
    w_grid[phi_index, theta_index] = direction_w
    direction_vector = np.array([direction_u, direction_v, direction_w])
    spherical_theta_projection_u = math.cos(theta_radians) * math.cos(phi_radians)
    spherical_theta_projection_v = math.cos(theta_radians) * math.sin(phi_radians)
    spherical_theta_projection_w = -math.sin(theta_radians)
    return (
        u_grid,
        v_grid,
        w_grid,
        direction_vector,
        spherical_theta_projection_u,
        spherical_theta_projection_v,
        spherical_theta_projection_w,
        direction_u,
        direction_v,
        direction_w,
    )


def incident_field_cartesian(
    theta_projection_u: float,
    theta_projection_v: float,
    theta_projection_w: float,
    incident_field_components: np.ndarray,
    electric_theta_component: complex,
    phi_radians: float,
    electric_phi_component: complex,
):
    incident_field_components[0] = (
        theta_projection_u * electric_theta_component - np.sin(phi_radians) * electric_phi_component
    )
    incident_field_components[1] = (
        theta_projection_v * electric_theta_component
        + math.cos(phi_radians) * electric_phi_component
    )
    incident_field_components[2] = theta_projection_w * electric_theta_component
    return incident_field_components


def bi_incident_field_cartesian(
    theta_projection_u: float,
    theta_projection_v: float,
    theta_projection_w: float,
    cosine_incident_phi: float,
    sine_incident_phi: float,
    electric_theta_component: complex,
    electric_phi_component: complex,
    incident_field_components: np.ndarray,
):
    incident_field_components[0] = (
        theta_projection_u * electric_theta_component - sine_incident_phi * electric_phi_component
    )
    incident_field_components[1] = (
        theta_projection_v * electric_theta_component + cosine_incident_phi * electric_phi_component
    )
    incident_field_components[2] = theta_projection_w * electric_theta_component
    return incident_field_components


def spherical_angles(u2, v2, w2):
    radial_xy = math.hypot(u2, v2)
    signed_sine_theta = radial_xy if w2 >= 0 else -radial_xy
    signed_sine_theta = max(-1.0, min(1.0, signed_sine_theta))
    th2 = math.asin(signed_sine_theta)
    phi2 = 0.0 if radial_xy <= 1e-12 else math.atan2(v2, u2)
    return th2, phi2


def bi_spherical_angles(ui2, vi2, wi2):
    radial_xy = math.hypot(ui2, vi2)
    sti2 = radial_xy if wi2 >= 0 else -radial_xy
    sti2 = max(-1.0, min(1.0, sti2))
    cti2 = math.sqrt(max(0.0, 1.0 - sti2**2))
    thi2 = math.acos(cti2)
    phii2 = 0.0 if radial_xy <= 1e-12 else math.atan2(vi2, ui2)
    return thi2, phii2, np.cos(phii2), np.sin(phii2), sti2, cti2


def phase_vertex_triangle(x, y, z, vind, bk, m, u, v, w):

    Dp = (
        2
        * bk
        * (
            (x[vind[m, 0] - 1] - x[vind[m, 2] - 1]) * u
            + (y[vind[m, 0] - 1] - y[vind[m, 2] - 1]) * v
            + (z[vind[m, 0] - 1] - z[vind[m, 2] - 1]) * w
        )
    )
    Dq = (
        2
        * bk
        * (
            (x[vind[m, 1] - 1] - x[vind[m, 2] - 1]) * u
            + (y[vind[m, 1] - 1] - y[vind[m, 2] - 1]) * v
            + (z[vind[m, 1] - 1] - z[vind[m, 2] - 1]) * w
        )
    )
    Do = 2 * bk * (x[vind[m, 2] - 1] * u + y[vind[m, 2] - 1] * v + z[vind[m, 2] - 1] * w)
    return (Dp, Dq, Do)


def precompute_phase_geometry(
    vertex_coordinates: np.ndarray,
    vertex_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute per-triangle vertex vectors used by phase equations."""
    zero_based_indices = vertex_indices.astype(np.int64) - 1
    p0 = vertex_coordinates[zero_based_indices[:, 0]]
    p1 = vertex_coordinates[zero_based_indices[:, 1]]
    p2 = vertex_coordinates[zero_based_indices[:, 2]]
    return p0 - p2, p1 - p2, p2


def phase_vertex_triangle_precomputed(
    wave_number: float,
    phase_p_vectors: np.ndarray,
    phase_q_vectors: np.ndarray,
    phase_origin_vectors: np.ndarray,
    triangle_index: int,
    direction_u: float,
    direction_v: float,
    direction_w: float,
) -> tuple[float, float, float]:
    """Monostatic phase terms with cached per-triangle geometry."""
    phase_p_vector = phase_p_vectors[triangle_index]
    phase_q_vector = phase_q_vectors[triangle_index]
    phase_origin_vector = phase_origin_vectors[triangle_index]
    dp = (
        2.0
        * wave_number
        * (
            phase_p_vector[0] * direction_u
            + phase_p_vector[1] * direction_v
            + phase_p_vector[2] * direction_w
        )
    )
    dq = (
        2.0
        * wave_number
        * (
            phase_q_vector[0] * direction_u
            + phase_q_vector[1] * direction_v
            + phase_q_vector[2] * direction_w
        )
    )
    do = (
        2.0
        * wave_number
        * (
            phase_origin_vector[0] * direction_u
            + phase_origin_vector[1] * direction_v
            + phase_origin_vector[2] * direction_w
        )
    )
    return dp, dq, do


def bi_phase_vertex_triangle(x, y, z, vind, bk, m, u, v, w, ui, vi, wi):
    Dp = bk * (
        (x[vind[m, 0] - 1] - x[vind[m, 2] - 1]) * (u + ui)
        + (y[vind[m, 0] - 1] - y[vind[m, 2] - 1]) * (v + vi)
        + (z[vind[m, 0] - 1] - z[vind[m, 2] - 1]) * (w + wi)
    )
    Dq = bk * (
        (x[vind[m, 1] - 1] - x[vind[m, 2] - 1]) * (u + ui)
        + (y[vind[m, 1] - 1] - y[vind[m, 2] - 1]) * (v + vi)
        + (z[vind[m, 1] - 1] - z[vind[m, 2] - 1]) * (w + wi)
    )
    Do = bk * (
        x[vind[m, 2] - 1] * (u + ui) + y[vind[m, 2] - 1] * (v + vi) + z[vind[m, 2] - 1] * (w + wi)
    )
    return (Dp, Dq, Do)


def bi_phase_vertex_triangle_precomputed(
    wave_number: float,
    phase_p_vectors: np.ndarray,
    phase_q_vectors: np.ndarray,
    phase_origin_vectors: np.ndarray,
    triangle_index: int,
    observation_direction_u: float,
    observation_direction_v: float,
    observation_direction_w: float,
    incident_direction_u: float,
    incident_direction_v: float,
    incident_direction_w: float,
) -> tuple[float, float, float]:
    """Bistatic phase terms with cached per-triangle geometry."""
    combined_u = observation_direction_u + incident_direction_u
    combined_v = observation_direction_v + incident_direction_v
    combined_w = observation_direction_w + incident_direction_w
    phase_p_vector = phase_p_vectors[triangle_index]
    phase_q_vector = phase_q_vectors[triangle_index]
    phase_origin_vector = phase_origin_vectors[triangle_index]
    dp = wave_number * (
        phase_p_vector[0] * combined_u
        + phase_p_vector[1] * combined_v
        + phase_p_vector[2] * combined_w
    )
    dq = wave_number * (
        phase_q_vector[0] * combined_u
        + phase_q_vector[1] * combined_v
        + phase_q_vector[2] * combined_w
    )
    do = wave_number * (
        phase_origin_vector[0] * combined_u
        + phase_origin_vector[1] * combined_v
        + phase_origin_vector[2] * combined_w
    )
    return dp, dq, do


def taylor_g(n, w):
    jw = 1j * w
    exp_jw = np.exp(jw)
    g = (exp_jw - 1) / jw
    if n > 0:
        for m in range(1, n + 1):
            go = g
            g = (exp_jw - m * go) / jw
    return g


def calculate_ic(
    phase_p: float,
    phase_q: float,
    phase_origin: float,
    taylor_terms: int,
    triangle_area: float,
    incident_amplitude: float,
    taylor_threshold: float,
) -> complex:
    area_scale = 2.0 * triangle_area
    phase_difference = phase_q - phase_p
    abs_phase_p = abs(phase_p)
    abs_phase_q = abs(phase_q)
    abs_phase_difference = abs(phase_difference)
    exp_phase_origin = cmath.exp(1j * phase_origin)
    # special case 1
    if abs_phase_p < taylor_threshold and abs_phase_q >= taylor_threshold:
        exp_phase_q = cmath.exp(1j * phase_q)
        series_integral: complex = 0.0 + 0.0j
        for n in range(taylor_terms + 1):
            series_integral = series_integral + (1j * phase_p) ** n / math.factorial(n) * (
                -incident_amplitude / (n + 1)
                + exp_phase_q * (incident_amplitude * taylor_g(n, -phase_q))
            )
        area_integral_value = series_integral * area_scale * exp_phase_origin / (1j * phase_q)
    # special case 2
    elif abs_phase_p < taylor_threshold and abs_phase_q < taylor_threshold:
        series_integral = 0.0 + 0.0j
        for n in range(taylor_terms + 1):
            for nn in range(taylor_terms):
                series_integral = (
                    series_integral
                    + (1j * phase_p) ** n
                    * (1j * phase_q) ** nn
                    / math.factorial(nn + n + 2)
                    * incident_amplitude
                )
        area_integral_value = series_integral * area_scale * exp_phase_origin
    # special case 3
    elif abs_phase_p >= taylor_threshold and abs_phase_q < taylor_threshold:
        exp_phase_p = cmath.exp(1j * phase_p)
        series_integral = 0.0 + 0.0j
        for n in range(taylor_terms + 1):
            series_integral = series_integral + (1j * phase_q) ** n / math.factorial(
                n
            ) * incident_amplitude * (taylor_g(n + 1, -phase_p) / (n + 1))
        area_integral_value = series_integral * area_scale * exp_phase_origin * exp_phase_p
    # special case 4
    elif (
        abs_phase_p >= taylor_threshold
        and abs_phase_q >= taylor_threshold
        and abs_phase_difference < taylor_threshold
    ):
        exp_phase_q = cmath.exp(1j * phase_q)
        series_integral = 0.0 + 0.0j
        for n in range(taylor_terms + 1):
            series_integral = series_integral + (1j * phase_difference) ** n / math.factorial(n) * (
                -incident_amplitude * taylor_g(n, phase_q)
                + exp_phase_q * incident_amplitude / (n + 1)
            )
        area_integral_value = series_integral * area_scale * exp_phase_origin / (1j * phase_q)
    else:
        exp_phase_p = cmath.exp(1j * phase_p)
        exp_phase_q = cmath.exp(1j * phase_q)
        area_integral_value = (
            area_scale
            * exp_phase_origin
            * (
                exp_phase_p * incident_amplitude / (phase_p * phase_difference)
                - exp_phase_q * incident_amplitude / (phase_q * phase_difference)
                - incident_amplitude / (phase_p * phase_q)
            )
        )
    return area_integral_value


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _factorial_numba(value: int) -> int:  # pragma: no cover
        result = 1
        for index in range(2, value + 1):
            result *= index
        return result

    @njit(cache=True)
    def _taylor_g_numba(n: int, w: float) -> complex:  # pragma: no cover
        jw = 1j * w
        exp_jw = np.exp(jw)
        g = (exp_jw - 1) / jw
        if n > 0:
            for m in range(1, n + 1):
                go = g
                g = (exp_jw - m * go) / jw
        return complex(g)

    @njit(cache=True)
    def _calculate_ic_numba(
        phase_p: float,
        phase_q: float,
        phase_origin: float,
        taylor_terms: int,
        triangle_area: float,
        incident_amplitude: float,
        taylor_threshold: float,
    ) -> complex:  # pragma: no cover
        area_scale = 2.0 * triangle_area
        phase_difference = phase_q - phase_p
        abs_phase_p = abs(phase_p)
        abs_phase_q = abs(phase_q)
        abs_phase_difference = abs(phase_difference)
        exp_phase_origin = np.exp(1j * phase_origin)

        if abs_phase_p < taylor_threshold and abs_phase_q >= taylor_threshold:
            exp_phase_q = np.exp(1j * phase_q)
            series_integral = 0.0 + 0.0j
            for n in range(taylor_terms + 1):
                series_integral = series_integral + (1j * phase_p) ** n / _factorial_numba(n) * (
                    -incident_amplitude / (n + 1)
                    + exp_phase_q * (incident_amplitude * _taylor_g_numba(n, -phase_q))
                )
            return complex(series_integral * area_scale * exp_phase_origin / (1j * phase_q))

        if abs_phase_p < taylor_threshold and abs_phase_q < taylor_threshold:
            series_integral = 0.0 + 0.0j
            for n in range(taylor_terms + 1):
                for nn in range(taylor_terms):
                    series_integral = (
                        series_integral
                        + (1j * phase_p) ** n
                        * (1j * phase_q) ** nn
                        / _factorial_numba(nn + n + 2)
                        * incident_amplitude
                    )
            return complex(series_integral * area_scale * exp_phase_origin)

        if abs_phase_p >= taylor_threshold and abs_phase_q < taylor_threshold:
            exp_phase_p = np.exp(1j * phase_p)
            series_integral = 0.0 + 0.0j
            for n in range(taylor_terms + 1):
                series_integral = series_integral + (1j * phase_q) ** n / _factorial_numba(
                    n
                ) * incident_amplitude * (_taylor_g_numba(n + 1, -phase_p) / (n + 1))
            return complex(series_integral * area_scale * exp_phase_origin * exp_phase_p)

        if (
            abs_phase_p >= taylor_threshold
            and abs_phase_q >= taylor_threshold
            and abs_phase_difference < taylor_threshold
        ):
            exp_phase_q = np.exp(1j * phase_q)
            series_integral = 0.0 + 0.0j
            for n in range(taylor_terms + 1):
                series_integral = series_integral + (1j * phase_difference) ** n / _factorial_numba(
                    n
                ) * (
                    -incident_amplitude * _taylor_g_numba(n, phase_q)
                    + exp_phase_q * incident_amplitude / (n + 1)
                )
            return complex(series_integral * area_scale * exp_phase_origin / (1j * phase_q))

        exp_phase_p = np.exp(1j * phase_p)
        exp_phase_q = np.exp(1j * phase_q)
        return complex(
            area_scale
            * exp_phase_origin
            * (
                exp_phase_p * incident_amplitude / (phase_p * phase_difference)
                - exp_phase_q * incident_amplitude / (phase_q * phase_difference)
                - incident_amplitude / (phase_p * phase_q)
            )
        )

    @njit(cache=True)
    def _mat2_mul_numba(
        a00: complex,
        a01: complex,
        a10: complex,
        a11: complex,
        b00: complex,
        b01: complex,
        b10: complex,
        b11: complex,
    ) -> tuple[complex, complex, complex, complex]:  # pragma: no cover
        return (
            a00 * b00 + a01 * b10,
            a00 * b01 + a01 * b11,
            a10 * b00 + a11 * b10,
            a10 * b01 + a11 * b11,
        )

    @njit(cache=True)
    def _apply_transfer_matrix_numba(
        m00: complex,
        m01: complex,
        m10: complex,
        m11: complex,
        gamma: complex,
        phase: complex,
    ) -> tuple[complex, complex, complex, complex]:  # pragma: no cover
        exp_phase = np.exp(1j * phase)
        exp_neg_phase = np.exp(-1j * phase)
        t00 = exp_phase
        t01 = gamma * exp_neg_phase
        t10 = gamma * exp_phase
        t11 = exp_neg_phase
        return (
            m00 * t00 + m01 * t10,
            m00 * t01 + m01 * t11,
            m10 * t00 + m11 * t10,
            m10 * t01 + m11 * t11,
        )

    @njit(cache=True)
    def _refl_coeff_numba(
        er1: complex,
        mr1: complex,
        er2: complex,
        mr2: complex,
        theta_incident: float,
    ) -> tuple[complex, complex, float, int]:  # pragma: no cover
        permeability_vacuum = 4 * math.pi * 1e-7
        permittivity_vacuum = 8.854e-12
        sinthetat = math.sin(theta_incident) * math.sqrt(
            (np.real(er1) * np.real(mr1)) / (np.real(er2) * np.real(mr2))
        )
        tir_flag = 0
        if sinthetat > 1.0:
            tir_flag = 1
            theta_transmitted = math.pi / 2
        else:
            theta_transmitted = math.asin(sinthetat)
        n1 = np.sqrt(mr1 * permeability_vacuum / (er1 * permittivity_vacuum))
        n2 = np.sqrt(mr2 * permeability_vacuum / (er2 * permittivity_vacuum))
        cosine_incident = math.cos(theta_incident)
        cosine_transmitted = math.cos(theta_transmitted)
        gamma_perpendicular = (n2 * cosine_incident - n1 * cosine_transmitted) / (
            n2 * cosine_incident + n1 * cosine_transmitted
        )
        gamma_parallel = (n2 * cosine_transmitted - n1 * cosine_incident) / (
            n2 * cosine_transmitted + n1 * cosine_incident
        )
        return gamma_parallel, gamma_perpendicular, theta_transmitted, tir_flag

    @njit(cache=True)
    def _material_reflection_coeff_numba(
        local_theta: float,
        frequency_hz: float,
        triangle_index: int,
        material_type_codes: np.ndarray,
        material_layer_count: np.ndarray,
        epsilon_r: np.ndarray,
        loss_tangent: np.ndarray,
        mu_r_real: np.ndarray,
        mu_r_imag: np.ndarray,
        thickness_m: np.ndarray,
    ) -> tuple[complex, complex]:  # pragma: no cover
        material_code = material_type_codes[triangle_index]
        if material_code == _MAT_CODE_PEC:
            return -1.0 + 0.0j, -1.0 + 0.0j

        if material_code == _MAT_CODE_COMPOSITE:
            eps = epsilon_r[triangle_index, 0]
            loss = loss_tangent[triangle_index, 0]
            mu_r = mu_r_real[triangle_index, 0]
            mu_i = mu_r_imag[triangle_index, 0]
            thickness = thickness_m[triangle_index, 0]
            epsilon_complex = complex(eps, -loss * eps)
            mu_complex = complex(mu_r, -mu_i)
            gamma_parallel_1, gamma_perpendicular_1, _theta_t, _tir = _refl_coeff_numba(
                1.0 + 0.0j,
                1.0 + 0.0j,
                epsilon_complex,
                mu_complex,
                local_theta,
            )
            gamma_parallel_2 = -gamma_parallel_1
            gamma_perpendicular_2 = -gamma_perpendicular_1
            wave_speed = 3e8 / math.sqrt(np.real(epsilon_complex) * np.real(mu_complex))
            phase = 2.0 * math.pi * thickness / (wave_speed / frequency_hz)

            m00p, m01p, m10p, m11p = _apply_transfer_matrix_numba(
                1.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                1.0 + 0.0j,
                gamma_parallel_1,
                phase,
            )
            m00s, m01s, m10s, m11s = _apply_transfer_matrix_numba(
                1.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                1.0 + 0.0j,
                gamma_perpendicular_1,
                phase,
            )
            m00p, m01p, m10p, m11p = _mat2_mul_numba(
                m00p,
                m01p,
                m10p,
                m11p,
                1.0 + 0.0j,
                gamma_parallel_2,
                gamma_parallel_2,
                1.0 + 0.0j,
            )
            m00s, m01s, m10s, m11s = _mat2_mul_numba(
                m00s,
                m01s,
                m10s,
                m11s,
                1.0 + 0.0j,
                gamma_perpendicular_2,
                gamma_perpendicular_2,
                1.0 + 0.0j,
            )
            return m10s / m00s, m10p / m00p

        layer_count = material_layer_count[triangle_index]
        sine_incidence = math.sin(local_theta)
        cosine_incidence = math.cos(local_theta)
        wave = 3e8 / frequency_hz
        beta_0 = 2.0 * math.pi / wave
        z_0 = 1.0 + 0.0j

        w00p, w01p, w10p, w11p = 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j
        w00s, w01s, w10s, w11s = 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j
        previous_z_parallel = 0.0 + 0.0j
        previous_z_perpendicular = 0.0 + 0.0j

        if material_code in (
            _MAT_CODE_COMPOSITE_ON_PEC,
            _MAT_CODE_MULTI_LAYER_ON_PEC,
        ):
            for layer_index in range(layer_count):
                eps = epsilon_r[triangle_index, layer_index]
                loss = loss_tangent[triangle_index, layer_index]
                mu_r = mu_r_real[triangle_index, layer_index]
                mu_i = mu_r_imag[triangle_index, layer_index]
                thickness = thickness_m[triangle_index, layer_index]
                epsilon_complex = complex(eps, -loss * eps)
                mu_complex = complex(mu_r, -mu_i)
                impedance_ratio = epsilon_complex / mu_complex
                root_term = np.sqrt(impedance_ratio - sine_incidence**2)
                z_parallel = root_term / (impedance_ratio * cosine_incidence)
                z_perpendicular = cosine_incidence / root_term

                if layer_index == 0:
                    gamma_parallel = (z_parallel - z_0) / (z_parallel + z_0)
                    gamma_perpendicular = (z_perpendicular - z_0) / (z_perpendicular + z_0)
                else:
                    gamma_parallel = (z_parallel - previous_z_parallel) / (
                        z_parallel + previous_z_parallel
                    )
                    gamma_perpendicular = (z_perpendicular - previous_z_perpendicular) / (
                        z_perpendicular + previous_z_perpendicular
                    )

                previous_z_parallel = z_parallel
                previous_z_perpendicular = z_perpendicular
                tau_parallel = 1.0 + gamma_parallel
                tau_perpendicular = 1.0 + gamma_perpendicular
                phase = (
                    beta_0 * thickness * np.sqrt(epsilon_complex * mu_complex - sine_incidence**2)
                )

                w00p, w01p, w10p, w11p = _apply_transfer_matrix_numba(
                    w00p,
                    w01p,
                    w10p,
                    w11p,
                    gamma_parallel,
                    phase,
                )
                w00s, w01s, w10s, w11s = _apply_transfer_matrix_numba(
                    w00s,
                    w01s,
                    w10s,
                    w11s,
                    gamma_perpendicular,
                    phase,
                )
                inverse_tau_parallel = 1.0 / tau_parallel
                inverse_tau_perpendicular = 1.0 / tau_perpendicular
                w00p *= inverse_tau_parallel
                w01p *= inverse_tau_parallel
                w10p *= inverse_tau_parallel
                w11p *= inverse_tau_parallel
                w00s *= inverse_tau_perpendicular
                w01s *= inverse_tau_perpendicular
                w10s *= inverse_tau_perpendicular
                w11s *= inverse_tau_perpendicular

            reflection_parallel = (w10p - w11p) / (w00p - w01p)
            reflection_perpendicular = (w10s - w11s) / (w00s - w01s)
            return reflection_perpendicular, reflection_parallel

        # MaterialCode.MULTI_LAYER
        previous_epsilon = 1.0 + 0.0j
        previous_mu = 1.0 + 0.0j
        previous_theta = local_theta
        last_phase = 0.0 + 0.0j

        for layer_index in range(layer_count):
            eps = epsilon_r[triangle_index, layer_index]
            loss = loss_tangent[triangle_index, layer_index]
            mu_r = mu_r_real[triangle_index, layer_index]
            mu_i = mu_r_imag[triangle_index, layer_index]
            thickness = thickness_m[triangle_index, layer_index]
            epsilon_complex = complex(eps, -loss * eps)
            mu_complex = complex(mu_r, -mu_i)
            if layer_index == 0:
                gamma_parallel, gamma_perpendicular, theta_transmitted, _tir = _refl_coeff_numba(
                    1.0 + 0.0j,
                    1.0 + 0.0j,
                    epsilon_complex,
                    mu_complex,
                    local_theta,
                )
            else:
                gamma_parallel, gamma_perpendicular, theta_transmitted, _tir = _refl_coeff_numba(
                    previous_epsilon,
                    previous_mu,
                    epsilon_complex,
                    mu_complex,
                    previous_theta,
                )
            wave_speed = 3e8 / math.sqrt(np.real(epsilon_complex) * np.real(mu_complex))
            last_phase = 2.0 * math.pi * thickness / (wave_speed / frequency_hz)
            w00p, w01p, w10p, w11p = _apply_transfer_matrix_numba(
                w00p,
                w01p,
                w10p,
                w11p,
                gamma_parallel,
                last_phase,
            )
            w00s, w01s, w10s, w11s = _apply_transfer_matrix_numba(
                w00s,
                w01s,
                w10s,
                w11s,
                gamma_perpendicular,
                last_phase,
            )
            previous_epsilon = epsilon_complex
            previous_mu = mu_complex
            previous_theta = theta_transmitted

        gamma_parallel_exit, gamma_perpendicular_exit, _theta_exit, _tir = _refl_coeff_numba(
            previous_epsilon,
            previous_mu,
            1.0 + 0.0j,
            1.0 + 0.0j,
            previous_theta,
        )
        w00p, w01p, w10p, w11p = _apply_transfer_matrix_numba(
            w00p,
            w01p,
            w10p,
            w11p,
            gamma_parallel_exit,
            last_phase,
        )
        w00s, w01s, w10s, w11s = _apply_transfer_matrix_numba(
            w00s,
            w01s,
            w10s,
            w11s,
            gamma_perpendicular_exit,
            last_phase,
        )
        return w10s / w00s, w10p / w00p

    @njit(cache=True)
    def _accumulate_monostatic_sample_numba(
        illumination_flag_mode: int,
        illumination_flags: np.ndarray,
        resistivity_values: np.ndarray,
        triangle_areas: np.ndarray,
        surface_alpha_cos: np.ndarray,
        surface_alpha_sin: np.ndarray,
        surface_beta_cos: np.ndarray,
        surface_beta_sin: np.ndarray,
        surface_normal_x: np.ndarray,
        surface_normal_y: np.ndarray,
        surface_normal_z: np.ndarray,
        phase_p_x: np.ndarray,
        phase_p_y: np.ndarray,
        phase_p_z: np.ndarray,
        phase_q_x: np.ndarray,
        phase_q_y: np.ndarray,
        phase_q_z: np.ndarray,
        phase_o_x: np.ndarray,
        phase_o_y: np.ndarray,
        phase_o_z: np.ndarray,
        direction_u: float,
        direction_v: float,
        direction_w: float,
        theta_projection_u: float,
        theta_projection_v: float,
        theta_projection_w: float,
        sine_phi: float,
        cosine_phi: float,
        incident_field_x: complex,
        incident_field_y: complex,
        incident_field_z: complex,
        two_wave_number: float,
        roughness_factor_secondary: float,
        normalized_correlation_distance: float,
        wavelength_m: float,
        incident_amplitude: float,
        taylor_terms: int,
        taylor_threshold: float,
        material_mode: int,
        frequency_hz: float,
        material_type_codes: np.ndarray,
        material_layer_count: np.ndarray,
        material_epsilon_r: np.ndarray,
        material_loss_tangent: np.ndarray,
        material_mu_r_real: np.ndarray,
        material_mu_r_imag: np.ndarray,
        material_thickness_m: np.ndarray,
    ) -> tuple[complex, complex, float, float]:  # pragma: no cover
        theta_component = 0.0 + 0.0j
        phi_component = 0.0 + 0.0j
        diffuse_theta = 0.0
        diffuse_phi = 0.0

        triangle_count = triangle_areas.shape[0]
        for triangle_index in range(triangle_count):
            normal_dot_observer = (
                surface_normal_x[triangle_index] * direction_u
                + surface_normal_y[triangle_index] * direction_v
                + surface_normal_z[triangle_index] * direction_w
            )
            if illumination_flag_mode == 0:
                illumination_flag = illumination_flags[triangle_index]
                if not (
                    (illumination_flag == 1 and normal_dot_observer >= 1e-5)
                    or illumination_flag == 0
                ):
                    continue

            cosine_alpha = surface_alpha_cos[triangle_index]
            sine_alpha = surface_alpha_sin[triangle_index]
            cosine_beta = surface_beta_cos[triangle_index]
            sine_beta = surface_beta_sin[triangle_index]

            rotated_direction_u = cosine_alpha * direction_u + sine_alpha * direction_v
            local_u = cosine_beta * rotated_direction_u - sine_beta * direction_w
            local_v = -sine_alpha * direction_u + cosine_alpha * direction_v
            local_w = sine_beta * rotated_direction_u + cosine_beta * direction_w

            phase_p = two_wave_number * (
                phase_p_x[triangle_index] * direction_u
                + phase_p_y[triangle_index] * direction_v
                + phase_p_z[triangle_index] * direction_w
            )
            phase_q = two_wave_number * (
                phase_q_x[triangle_index] * direction_u
                + phase_q_y[triangle_index] * direction_v
                + phase_q_z[triangle_index] * direction_w
            )
            phase_origin = two_wave_number * (
                phase_o_x[triangle_index] * direction_u
                + phase_o_y[triangle_index] * direction_v
                + phase_o_z[triangle_index] * direction_w
            )

            local_radial = math.hypot(local_u, local_v)
            local_sine_theta = local_radial if local_w >= 0 else -local_radial
            if local_sine_theta > 1.0:
                local_sine_theta = 1.0
            elif local_sine_theta < -1.0:
                local_sine_theta = -1.0
            local_cosine_theta = abs(local_w)
            if local_radial <= 1e-12:
                local_cosine_phi = 1.0
                local_sine_phi = 0.0
            else:
                inverse_local_radial = 1.0 / local_radial
                local_cosine_phi = local_u * inverse_local_radial
                local_sine_phi = local_v * inverse_local_radial

            rotated_x = cosine_alpha * incident_field_x + sine_alpha * incident_field_y
            rotated_y = -sine_alpha * incident_field_x + cosine_alpha * incident_field_y
            local_field_x = cosine_beta * rotated_x - sine_beta * incident_field_z
            local_field_y = rotated_y
            local_field_z = sine_beta * rotated_x + cosine_beta * incident_field_z

            local_theta_field = (
                local_field_x * local_cosine_theta * local_cosine_phi
                + local_field_y * local_cosine_theta * local_sine_phi
                - local_field_z * local_sine_theta
            )
            local_phi_field = -local_field_x * local_sine_phi + local_field_y * local_cosine_phi

            if material_mode == 1:
                local_theta = math.asin(local_sine_theta)
                reflection_perpendicular, reflection_parallel = _material_reflection_coeff_numba(
                    local_theta,
                    frequency_hz,
                    triangle_index,
                    material_type_codes,
                    material_layer_count,
                    material_epsilon_r,
                    material_loss_tangent,
                    material_mu_r_real,
                    material_mu_r_imag,
                    material_thickness_m,
                )
            else:
                resistivity = resistivity_values[triangle_index]
                reflection_perpendicular = -1.0 / (2.0 * resistivity * local_cosine_theta + 1.0)
                reflection_parallel = 0.0 + 0.0j
                reflection_parallel_denominator = 2.0 * resistivity + local_cosine_theta
                if reflection_parallel_denominator != 0.0:
                    reflection_parallel = -local_cosine_theta / reflection_parallel_denominator

            local_surface_current_x = (
                -local_theta_field * local_cosine_phi * reflection_parallel
                + local_phi_field * local_sine_phi * reflection_perpendicular * local_cosine_theta
            )
            local_surface_current_y = (
                -local_theta_field * local_sine_phi * reflection_parallel
                - local_phi_field * local_cosine_phi * reflection_perpendicular * local_cosine_theta
            )

            triangle_area = triangle_areas[triangle_index]
            area_integral_value = _calculate_ic_numba(
                phase_p,
                phase_q,
                phase_origin,
                taylor_terms,
                triangle_area,
                incident_amplitude,
                taylor_threshold,
            )

            diffuse_scale = roughness_factor_secondary * triangle_area * (local_cosine_theta**2)
            diffuse_exponent = -(
                (normalized_correlation_distance * math.pi * local_sine_theta / wavelength_m) ** 2
            )
            diffuse_component = diffuse_scale * math.exp(diffuse_exponent)

            scattered_local_x = local_surface_current_x * area_integral_value
            scattered_local_y = local_surface_current_y * area_integral_value
            diffuse_local_x = local_surface_current_x * diffuse_component
            diffuse_local_y = local_surface_current_y * diffuse_component

            scattered_global_x = (
                cosine_alpha * cosine_beta * scattered_local_x - sine_alpha * scattered_local_y
            )
            scattered_global_y = (
                sine_alpha * cosine_beta * scattered_local_x + cosine_alpha * scattered_local_y
            )
            scattered_global_z = -sine_beta * scattered_local_x

            diffuse_global_x = (
                cosine_alpha * cosine_beta * diffuse_local_x - sine_alpha * diffuse_local_y
            )
            diffuse_global_y = (
                sine_alpha * cosine_beta * diffuse_local_x + cosine_alpha * diffuse_local_y
            )
            diffuse_global_z = -sine_beta * diffuse_local_x

            theta_component += (
                theta_projection_u * scattered_global_x
                + theta_projection_v * scattered_global_y
                + theta_projection_w * scattered_global_z
            )
            phi_component += -sine_phi * scattered_global_x + cosine_phi * scattered_global_y
            diffuse_theta += abs(
                theta_projection_u * diffuse_global_x
                + theta_projection_v * diffuse_global_y
                + theta_projection_w * diffuse_global_z
            )
            diffuse_phi += abs(-sine_phi * diffuse_global_x + cosine_phi * diffuse_global_y)

        return theta_component, phi_component, diffuse_theta, diffuse_phi

    @njit(cache=True)
    def _accumulate_bistatic_sample_numba(
        illumination_flag_mode: int,
        illumination_flags: np.ndarray,
        resistivity_values: np.ndarray,
        triangle_areas: np.ndarray,
        surface_alpha_cos: np.ndarray,
        surface_alpha_sin: np.ndarray,
        surface_beta_cos: np.ndarray,
        surface_beta_sin: np.ndarray,
        surface_normal_x: np.ndarray,
        surface_normal_y: np.ndarray,
        surface_normal_z: np.ndarray,
        phase_p_x: np.ndarray,
        phase_p_y: np.ndarray,
        phase_p_z: np.ndarray,
        phase_q_x: np.ndarray,
        phase_q_y: np.ndarray,
        phase_q_z: np.ndarray,
        phase_o_x: np.ndarray,
        phase_o_y: np.ndarray,
        phase_o_z: np.ndarray,
        incident_direction_u: float,
        incident_direction_v: float,
        incident_direction_w: float,
        observation_direction_u: float,
        observation_direction_v: float,
        observation_direction_w: float,
        theta_projection_u: float,
        theta_projection_v: float,
        theta_projection_w: float,
        sine_phi: float,
        cosine_phi: float,
        incident_field_x: complex,
        incident_field_y: complex,
        incident_field_z: complex,
        wave_number: float,
        roughness_factor_secondary: float,
        normalized_correlation_distance: float,
        wavelength_m: float,
        incident_amplitude: float,
        taylor_terms: int,
        taylor_threshold: float,
        material_mode: int,
        frequency_hz: float,
        material_type_codes: np.ndarray,
        material_layer_count: np.ndarray,
        material_epsilon_r: np.ndarray,
        material_loss_tangent: np.ndarray,
        material_mu_r_real: np.ndarray,
        material_mu_r_imag: np.ndarray,
        material_thickness_m: np.ndarray,
    ) -> tuple[complex, complex, float, float]:  # pragma: no cover
        theta_component = 0.0 + 0.0j
        phi_component = 0.0 + 0.0j
        diffuse_theta = 0.0
        diffuse_phi = 0.0

        combined_direction_u = observation_direction_u + incident_direction_u
        combined_direction_v = observation_direction_v + incident_direction_v
        combined_direction_w = observation_direction_w + incident_direction_w
        triangle_count = triangle_areas.shape[0]

        for triangle_index in range(triangle_count):
            incident_normal_dot = (
                surface_normal_x[triangle_index] * incident_direction_u
                + surface_normal_y[triangle_index] * incident_direction_v
                + surface_normal_z[triangle_index] * incident_direction_w
            )
            if illumination_flag_mode == 0:
                illumination_flag = illumination_flags[triangle_index]
                if not (
                    (illumination_flag == 1 and incident_normal_dot >= 0.0)
                    or illumination_flag == 0
                ):
                    continue

            cosine_alpha = surface_alpha_cos[triangle_index]
            sine_alpha = surface_alpha_sin[triangle_index]
            cosine_beta = surface_beta_cos[triangle_index]
            sine_beta = surface_beta_sin[triangle_index]

            rotated_incident_u = (
                cosine_alpha * incident_direction_u + sine_alpha * incident_direction_v
            )
            incident_local_u = cosine_beta * rotated_incident_u - sine_beta * incident_direction_w
            incident_local_v = (
                -sine_alpha * incident_direction_u + cosine_alpha * incident_direction_v
            )
            incident_local_w = sine_beta * rotated_incident_u + cosine_beta * incident_direction_w

            incident_local_radial = math.hypot(incident_local_u, incident_local_v)
            incident_local_sine_theta = (
                incident_local_radial if incident_local_w >= 0 else -incident_local_radial
            )
            if incident_local_sine_theta > 1.0:
                incident_local_sine_theta = 1.0
            elif incident_local_sine_theta < -1.0:
                incident_local_sine_theta = -1.0
            incident_local_cosine_theta = abs(incident_local_w)
            if incident_local_radial <= 1e-12:
                cosine_incident_phi = 1.0
                sine_incident_phi = 0.0
            else:
                inverse_incident_local_radial = 1.0 / incident_local_radial
                cosine_incident_phi = incident_local_u * inverse_incident_local_radial
                sine_incident_phi = incident_local_v * inverse_incident_local_radial

            rotated_observation_u = (
                cosine_alpha * observation_direction_u + sine_alpha * observation_direction_v
            )
            observation_local_u = (
                cosine_beta * rotated_observation_u - sine_beta * observation_direction_w
            )
            observation_local_v = (
                -sine_alpha * observation_direction_u + cosine_alpha * observation_direction_v
            )
            observation_local_w = (
                sine_beta * rotated_observation_u + cosine_beta * observation_direction_w
            )

            observation_local_radial = math.hypot(observation_local_u, observation_local_v)
            observation_local_sine_theta = (
                observation_local_radial if observation_local_w >= 0 else -observation_local_radial
            )
            if observation_local_sine_theta > 1.0:
                observation_local_sine_theta = 1.0
            elif observation_local_sine_theta < -1.0:
                observation_local_sine_theta = -1.0
            observation_local_cosine_theta = abs(observation_local_w)

            phase_p = wave_number * (
                phase_p_x[triangle_index] * combined_direction_u
                + phase_p_y[triangle_index] * combined_direction_v
                + phase_p_z[triangle_index] * combined_direction_w
            )
            phase_q = wave_number * (
                phase_q_x[triangle_index] * combined_direction_u
                + phase_q_y[triangle_index] * combined_direction_v
                + phase_q_z[triangle_index] * combined_direction_w
            )
            phase_origin = wave_number * (
                phase_o_x[triangle_index] * combined_direction_u
                + phase_o_y[triangle_index] * combined_direction_v
                + phase_o_z[triangle_index] * combined_direction_w
            )

            rotated_field_x = cosine_alpha * incident_field_x + sine_alpha * incident_field_y
            rotated_field_y = -sine_alpha * incident_field_x + cosine_alpha * incident_field_y
            local_field_x = cosine_beta * rotated_field_x - sine_beta * incident_field_z
            local_field_y = rotated_field_y
            local_field_z = sine_beta * rotated_field_x + cosine_beta * incident_field_z

            local_theta_field = (
                local_field_x * incident_local_cosine_theta * cosine_incident_phi
                + local_field_y * incident_local_cosine_theta * sine_incident_phi
                - local_field_z * incident_local_sine_theta
            )
            local_phi_field = (
                -local_field_x * sine_incident_phi + local_field_y * cosine_incident_phi
            )

            if material_mode == 1:
                observation_local_theta = math.acos(observation_local_cosine_theta)
                reflection_perpendicular, reflection_parallel = _material_reflection_coeff_numba(
                    observation_local_theta,
                    frequency_hz,
                    triangle_index,
                    material_type_codes,
                    material_layer_count,
                    material_epsilon_r,
                    material_loss_tangent,
                    material_mu_r_real,
                    material_mu_r_imag,
                    material_thickness_m,
                )
            else:
                resistivity = resistivity_values[triangle_index]
                reflection_perpendicular = -1.0 / (
                    2.0 * resistivity * observation_local_cosine_theta + 1.0
                )
                reflection_parallel = 0.0 + 0.0j
                reflection_parallel_denominator = 2.0 * resistivity + observation_local_cosine_theta
                if reflection_parallel_denominator != 0.0:
                    reflection_parallel = (
                        -observation_local_cosine_theta / reflection_parallel_denominator
                    )

            local_surface_current_x = (
                -local_theta_field * cosine_incident_phi * reflection_parallel
                + local_phi_field
                * sine_incident_phi
                * reflection_perpendicular
                * incident_local_cosine_theta
            )
            local_surface_current_y = (
                -local_theta_field * sine_incident_phi * reflection_parallel
                - local_phi_field
                * cosine_incident_phi
                * reflection_perpendicular
                * incident_local_cosine_theta
            )

            triangle_area = triangle_areas[triangle_index]
            area_integral_value = _calculate_ic_numba(
                phase_p,
                phase_q,
                phase_origin,
                taylor_terms,
                triangle_area,
                incident_amplitude,
                taylor_threshold,
            )

            diffuse_scale = (
                roughness_factor_secondary * triangle_area * (observation_local_cosine_theta**2)
            )
            diffuse_exponent = -(
                (
                    normalized_correlation_distance
                    * math.pi
                    * observation_local_sine_theta
                    / wavelength_m
                )
                ** 2
            )
            diffuse_component = diffuse_scale * math.exp(diffuse_exponent)

            scattered_local_x = local_surface_current_x * area_integral_value
            scattered_local_y = local_surface_current_y * area_integral_value
            diffuse_local_x = local_surface_current_x * diffuse_component
            diffuse_local_y = local_surface_current_y * diffuse_component

            scattered_global_x = (
                cosine_alpha * cosine_beta * scattered_local_x - sine_alpha * scattered_local_y
            )
            scattered_global_y = (
                sine_alpha * cosine_beta * scattered_local_x + cosine_alpha * scattered_local_y
            )
            scattered_global_z = -sine_beta * scattered_local_x

            diffuse_global_x = (
                cosine_alpha * cosine_beta * diffuse_local_x - sine_alpha * diffuse_local_y
            )
            diffuse_global_y = (
                sine_alpha * cosine_beta * diffuse_local_x + cosine_alpha * diffuse_local_y
            )
            diffuse_global_z = -sine_beta * diffuse_local_x

            theta_component += (
                theta_projection_u * scattered_global_x
                + theta_projection_v * scattered_global_y
                + theta_projection_w * scattered_global_z
            )
            phi_component += -sine_phi * scattered_global_x + cosine_phi * scattered_global_y
            diffuse_theta += abs(
                theta_projection_u * diffuse_global_x
                + theta_projection_v * diffuse_global_y
                + theta_projection_w * diffuse_global_z
            )
            diffuse_phi += abs(-sine_phi * diffuse_global_x + cosine_phi * diffuse_global_y)

        return theta_component, phi_component, diffuse_theta, diffuse_phi


else:

    def _accumulate_monostatic_sample_numba(*_args, **_kwargs):  # pragma: no cover
        raise RuntimeError("Numba acceleration requested but numba is not installed.")

    def _accumulate_bistatic_sample_numba(*_args, **_kwargs):  # pragma: no cover
        raise RuntimeError("Numba acceleration requested but numba is not installed.")


def accumulate_monostatic_sample_numba(
    *,
    illumination_flag_mode: int,
    illumination_flags: np.ndarray,
    resistivity_values: np.ndarray,
    triangle_areas: np.ndarray,
    surface_alpha_cos: np.ndarray,
    surface_alpha_sin: np.ndarray,
    surface_beta_cos: np.ndarray,
    surface_beta_sin: np.ndarray,
    surface_normal_x: np.ndarray,
    surface_normal_y: np.ndarray,
    surface_normal_z: np.ndarray,
    phase_p_x: np.ndarray,
    phase_p_y: np.ndarray,
    phase_p_z: np.ndarray,
    phase_q_x: np.ndarray,
    phase_q_y: np.ndarray,
    phase_q_z: np.ndarray,
    phase_o_x: np.ndarray,
    phase_o_y: np.ndarray,
    phase_o_z: np.ndarray,
    direction_u: float,
    direction_v: float,
    direction_w: float,
    theta_projection_u: float,
    theta_projection_v: float,
    theta_projection_w: float,
    sine_phi: float,
    cosine_phi: float,
    incident_field_x: complex,
    incident_field_y: complex,
    incident_field_z: complex,
    two_wave_number: float,
    roughness_factor_secondary: float,
    normalized_correlation_distance: float,
    wavelength_m: float,
    incident_amplitude: float,
    taylor_terms: int,
    taylor_threshold: float,
) -> tuple[complex, complex, float, float]:
    """Numba-accelerated monostatic accumulation for one (phi, theta) sample."""
    return cast(
        tuple[complex, complex, float, float],
        _accumulate_monostatic_sample_numba(
            illumination_flag_mode,
            illumination_flags,
            resistivity_values,
            triangle_areas,
            surface_alpha_cos,
            surface_alpha_sin,
            surface_beta_cos,
            surface_beta_sin,
            surface_normal_x,
            surface_normal_y,
            surface_normal_z,
            phase_p_x,
            phase_p_y,
            phase_p_z,
            phase_q_x,
            phase_q_y,
            phase_q_z,
            phase_o_x,
            phase_o_y,
            phase_o_z,
            direction_u,
            direction_v,
            direction_w,
            theta_projection_u,
            theta_projection_v,
            theta_projection_w,
            sine_phi,
            cosine_phi,
            incident_field_x,
            incident_field_y,
            incident_field_z,
            two_wave_number,
            roughness_factor_secondary,
            normalized_correlation_distance,
            wavelength_m,
            incident_amplitude,
            taylor_terms,
            taylor_threshold,
            0,
            0.0,
            _NUMBA_PLACEHOLDER_INT_ARRAY,
            _NUMBA_PLACEHOLDER_INT_ARRAY,
            _NUMBA_PLACEHOLDER_FLOAT_ARRAY,
            _NUMBA_PLACEHOLDER_FLOAT_ARRAY,
            _NUMBA_PLACEHOLDER_FLOAT_ARRAY,
            _NUMBA_PLACEHOLDER_FLOAT_ARRAY,
            _NUMBA_PLACEHOLDER_FLOAT_ARRAY,
        ),
    )


def accumulate_monostatic_sample_numba_material(
    *,
    illumination_flag_mode: int,
    illumination_flags: np.ndarray,
    resistivity_values: np.ndarray,
    triangle_areas: np.ndarray,
    surface_alpha_cos: np.ndarray,
    surface_alpha_sin: np.ndarray,
    surface_beta_cos: np.ndarray,
    surface_beta_sin: np.ndarray,
    surface_normal_x: np.ndarray,
    surface_normal_y: np.ndarray,
    surface_normal_z: np.ndarray,
    phase_p_x: np.ndarray,
    phase_p_y: np.ndarray,
    phase_p_z: np.ndarray,
    phase_q_x: np.ndarray,
    phase_q_y: np.ndarray,
    phase_q_z: np.ndarray,
    phase_o_x: np.ndarray,
    phase_o_y: np.ndarray,
    phase_o_z: np.ndarray,
    direction_u: float,
    direction_v: float,
    direction_w: float,
    theta_projection_u: float,
    theta_projection_v: float,
    theta_projection_w: float,
    sine_phi: float,
    cosine_phi: float,
    incident_field_x: complex,
    incident_field_y: complex,
    incident_field_z: complex,
    two_wave_number: float,
    roughness_factor_secondary: float,
    normalized_correlation_distance: float,
    wavelength_m: float,
    incident_amplitude: float,
    taylor_terms: int,
    taylor_threshold: float,
    frequency_hz: float,
    material_type_codes: np.ndarray,
    material_layer_count: np.ndarray,
    material_epsilon_r: np.ndarray,
    material_loss_tangent: np.ndarray,
    material_mu_r_real: np.ndarray,
    material_mu_r_imag: np.ndarray,
    material_thickness_m: np.ndarray,
) -> tuple[complex, complex, float, float]:
    """Numba-accelerated monostatic accumulation with material-library reflections."""
    return cast(
        tuple[complex, complex, float, float],
        _accumulate_monostatic_sample_numba(
            illumination_flag_mode,
            illumination_flags,
            resistivity_values,
            triangle_areas,
            surface_alpha_cos,
            surface_alpha_sin,
            surface_beta_cos,
            surface_beta_sin,
            surface_normal_x,
            surface_normal_y,
            surface_normal_z,
            phase_p_x,
            phase_p_y,
            phase_p_z,
            phase_q_x,
            phase_q_y,
            phase_q_z,
            phase_o_x,
            phase_o_y,
            phase_o_z,
            direction_u,
            direction_v,
            direction_w,
            theta_projection_u,
            theta_projection_v,
            theta_projection_w,
            sine_phi,
            cosine_phi,
            incident_field_x,
            incident_field_y,
            incident_field_z,
            two_wave_number,
            roughness_factor_secondary,
            normalized_correlation_distance,
            wavelength_m,
            incident_amplitude,
            taylor_terms,
            taylor_threshold,
            1,
            frequency_hz,
            material_type_codes,
            material_layer_count,
            material_epsilon_r,
            material_loss_tangent,
            material_mu_r_real,
            material_mu_r_imag,
            material_thickness_m,
        ),
    )


def accumulate_bistatic_sample_numba(
    *,
    illumination_flag_mode: int,
    illumination_flags: np.ndarray,
    resistivity_values: np.ndarray,
    triangle_areas: np.ndarray,
    surface_alpha_cos: np.ndarray,
    surface_alpha_sin: np.ndarray,
    surface_beta_cos: np.ndarray,
    surface_beta_sin: np.ndarray,
    surface_normal_x: np.ndarray,
    surface_normal_y: np.ndarray,
    surface_normal_z: np.ndarray,
    phase_p_x: np.ndarray,
    phase_p_y: np.ndarray,
    phase_p_z: np.ndarray,
    phase_q_x: np.ndarray,
    phase_q_y: np.ndarray,
    phase_q_z: np.ndarray,
    phase_o_x: np.ndarray,
    phase_o_y: np.ndarray,
    phase_o_z: np.ndarray,
    incident_direction_u: float,
    incident_direction_v: float,
    incident_direction_w: float,
    observation_direction_u: float,
    observation_direction_v: float,
    observation_direction_w: float,
    theta_projection_u: float,
    theta_projection_v: float,
    theta_projection_w: float,
    sine_phi: float,
    cosine_phi: float,
    incident_field_x: complex,
    incident_field_y: complex,
    incident_field_z: complex,
    wave_number: float,
    roughness_factor_secondary: float,
    normalized_correlation_distance: float,
    wavelength_m: float,
    incident_amplitude: float,
    taylor_terms: int,
    taylor_threshold: float,
) -> tuple[complex, complex, float, float]:
    """Numba-accelerated bistatic accumulation for one (phi, theta) sample."""
    return cast(
        tuple[complex, complex, float, float],
        _accumulate_bistatic_sample_numba(
            illumination_flag_mode,
            illumination_flags,
            resistivity_values,
            triangle_areas,
            surface_alpha_cos,
            surface_alpha_sin,
            surface_beta_cos,
            surface_beta_sin,
            surface_normal_x,
            surface_normal_y,
            surface_normal_z,
            phase_p_x,
            phase_p_y,
            phase_p_z,
            phase_q_x,
            phase_q_y,
            phase_q_z,
            phase_o_x,
            phase_o_y,
            phase_o_z,
            incident_direction_u,
            incident_direction_v,
            incident_direction_w,
            observation_direction_u,
            observation_direction_v,
            observation_direction_w,
            theta_projection_u,
            theta_projection_v,
            theta_projection_w,
            sine_phi,
            cosine_phi,
            incident_field_x,
            incident_field_y,
            incident_field_z,
            wave_number,
            roughness_factor_secondary,
            normalized_correlation_distance,
            wavelength_m,
            incident_amplitude,
            taylor_terms,
            taylor_threshold,
            0,
            0.0,
            _NUMBA_PLACEHOLDER_INT_ARRAY,
            _NUMBA_PLACEHOLDER_INT_ARRAY,
            _NUMBA_PLACEHOLDER_FLOAT_ARRAY,
            _NUMBA_PLACEHOLDER_FLOAT_ARRAY,
            _NUMBA_PLACEHOLDER_FLOAT_ARRAY,
            _NUMBA_PLACEHOLDER_FLOAT_ARRAY,
            _NUMBA_PLACEHOLDER_FLOAT_ARRAY,
        ),
    )


def accumulate_bistatic_sample_numba_material(
    *,
    illumination_flag_mode: int,
    illumination_flags: np.ndarray,
    resistivity_values: np.ndarray,
    triangle_areas: np.ndarray,
    surface_alpha_cos: np.ndarray,
    surface_alpha_sin: np.ndarray,
    surface_beta_cos: np.ndarray,
    surface_beta_sin: np.ndarray,
    surface_normal_x: np.ndarray,
    surface_normal_y: np.ndarray,
    surface_normal_z: np.ndarray,
    phase_p_x: np.ndarray,
    phase_p_y: np.ndarray,
    phase_p_z: np.ndarray,
    phase_q_x: np.ndarray,
    phase_q_y: np.ndarray,
    phase_q_z: np.ndarray,
    phase_o_x: np.ndarray,
    phase_o_y: np.ndarray,
    phase_o_z: np.ndarray,
    incident_direction_u: float,
    incident_direction_v: float,
    incident_direction_w: float,
    observation_direction_u: float,
    observation_direction_v: float,
    observation_direction_w: float,
    theta_projection_u: float,
    theta_projection_v: float,
    theta_projection_w: float,
    sine_phi: float,
    cosine_phi: float,
    incident_field_x: complex,
    incident_field_y: complex,
    incident_field_z: complex,
    wave_number: float,
    roughness_factor_secondary: float,
    normalized_correlation_distance: float,
    wavelength_m: float,
    incident_amplitude: float,
    taylor_terms: int,
    taylor_threshold: float,
    frequency_hz: float,
    material_type_codes: np.ndarray,
    material_layer_count: np.ndarray,
    material_epsilon_r: np.ndarray,
    material_loss_tangent: np.ndarray,
    material_mu_r_real: np.ndarray,
    material_mu_r_imag: np.ndarray,
    material_thickness_m: np.ndarray,
) -> tuple[complex, complex, float, float]:
    """Numba-accelerated bistatic accumulation with material-library reflections."""
    return cast(
        tuple[complex, complex, float, float],
        _accumulate_bistatic_sample_numba(
            illumination_flag_mode,
            illumination_flags,
            resistivity_values,
            triangle_areas,
            surface_alpha_cos,
            surface_alpha_sin,
            surface_beta_cos,
            surface_beta_sin,
            surface_normal_x,
            surface_normal_y,
            surface_normal_z,
            phase_p_x,
            phase_p_y,
            phase_p_z,
            phase_q_x,
            phase_q_y,
            phase_q_z,
            phase_o_x,
            phase_o_y,
            phase_o_z,
            incident_direction_u,
            incident_direction_v,
            incident_direction_w,
            observation_direction_u,
            observation_direction_v,
            observation_direction_w,
            theta_projection_u,
            theta_projection_v,
            theta_projection_w,
            sine_phi,
            cosine_phi,
            incident_field_x,
            incident_field_y,
            incident_field_z,
            wave_number,
            roughness_factor_secondary,
            normalized_correlation_distance,
            wavelength_m,
            incident_amplitude,
            taylor_terms,
            taylor_threshold,
            1,
            frequency_hz,
            material_type_codes,
            material_layer_count,
            material_epsilon_r,
            material_loss_tangent,
            material_mu_r_real,
            material_mu_r_imag,
            material_thickness_m,
        ),
    )


def calculate_fields(
    triangle_area: float,
    roughness_factor_secondary: float,
    normalized_correlation_distance: float,
    local_sine_theta: float,
    local_cosine_theta: float,
    wavelength_m: float,
    local_surface_current_y: complex,
    area_integral_value: complex,
    theta_projection_u: float,
    theta_projection_v: float,
    theta_projection_w: float,
    sine_phi: float,
    cosine_phi: float,
    accumulated_theta_field: complex,
    accumulated_phi_field: complex,
    accumulated_diffuse_theta: float,
    accumulated_diffuse_phi: float,
    local_surface_current_x: complex,
    transform_z: np.ndarray,
    transform_y: np.ndarray,
):
    cosine_alpha = float(transform_z[0, 0])
    sine_alpha = float(transform_z[0, 1])
    cosine_beta = float(transform_y[0, 0])
    sine_beta = float(transform_y[2, 0])

    Edif = (
        roughness_factor_secondary
        * triangle_area
        * (local_cosine_theta**2)
        * np.exp(
            -((normalized_correlation_distance * np.pi * local_sine_theta / wavelength_m) ** 2)
        )
    )

    es_local_x = local_surface_current_x * area_integral_value
    es_local_y = local_surface_current_y * area_integral_value
    ed_local_x = local_surface_current_x * Edif
    ed_local_y = local_surface_current_y * Edif

    es_global_x = cosine_alpha * cosine_beta * es_local_x - sine_alpha * es_local_y
    es_global_y = sine_alpha * cosine_beta * es_local_x + cosine_alpha * es_local_y
    es_global_z = -sine_beta * es_local_x

    ed_global_x = cosine_alpha * cosine_beta * ed_local_x - sine_alpha * ed_local_y
    ed_global_y = sine_alpha * cosine_beta * ed_local_x + cosine_alpha * ed_local_y
    ed_global_z = -sine_beta * ed_local_x

    scattered_theta = (
        theta_projection_u * es_global_x
        + theta_projection_v * es_global_y
        + theta_projection_w * es_global_z
    )
    scattered_phi = -sine_phi * es_global_x + cosine_phi * es_global_y
    diffuse_theta = (
        theta_projection_u * ed_global_x
        + theta_projection_v * ed_global_y
        + theta_projection_w * ed_global_z
    )
    diffuse_phi = -sine_phi * ed_global_x + cosine_phi * ed_global_y

    accumulated_theta_field += scattered_theta
    accumulated_diffuse_theta += abs(diffuse_theta)
    accumulated_phi_field += scattered_phi
    accumulated_diffuse_phi += abs(diffuse_phi)
    return (
        accumulated_theta_field,
        accumulated_phi_field,
        accumulated_diffuse_phi,
        accumulated_diffuse_theta,
    )


def calculate_sth_sph(
    roughness_factor_primary: float,
    accumulated_theta_field: complex,
    accumulated_phi_field: complex,
    accumulated_diffuse_theta: float,
    wavelength_m: float,
    rcs_theta_db: np.ndarray,
    rcs_phi_db: np.ndarray,
    phi_index: int,
    theta_index: int,
    accumulated_diffuse_phi: float,
):
    floor_power = np.finfo(np.float64).tiny
    theta_power = (
        4
        * np.pi
        * roughness_factor_primary
        * (
            np.abs(accumulated_theta_field) ** 2
            + np.sqrt(1 - roughness_factor_primary**2) * accumulated_diffuse_theta
        )
        / wavelength_m**2
    )
    phi_power = (
        4
        * np.pi
        * roughness_factor_primary
        * (
            np.abs(accumulated_phi_field) ** 2
            + np.sqrt(1 - roughness_factor_primary**2) * accumulated_diffuse_phi
        )
        / wavelength_m**2
    )
    rcs_theta_db[phi_index, theta_index] = 10 * np.log10(np.maximum(theta_power, floor_power))
    rcs_phi_db[phi_index, theta_index] = 10 * np.log10(np.maximum(phi_power, floor_power))


def product_vector(
    n_triangles: int,
    surface_normals: np.ndarray,
    vertex_coordinates: np.ndarray,
    triangle_edge_lengths: np.ndarray,
    triangle_areas: np.ndarray,
    surface_alpha_angles: np.ndarray,
    surface_beta_angles: np.ndarray,
    vertex_indices: np.ndarray,
):
    """Vectorized triangle geometry preparation (normals, edge lengths, area)."""
    indices = vertex_indices.astype(np.int64) - 1
    p0 = vertex_coordinates[indices[:, 0]]
    p1 = vertex_coordinates[indices[:, 1]]
    p2 = vertex_coordinates[indices[:, 2]]

    a_vec = p1 - p0
    b_vec = p2 - p1
    c_vec = p0 - p2

    normals = -np.cross(b_vec, a_vec)
    computed_edge_lengths = np.column_stack(
        (
            np.linalg.norm(a_vec, axis=1),
            np.linalg.norm(b_vec, axis=1),
            np.linalg.norm(c_vec, axis=1),
        )
    )
    ss = 0.5 * computed_edge_lengths.sum(axis=1)
    area_expr = (
        ss
        * (ss - computed_edge_lengths[:, 0])
        * (ss - computed_edge_lengths[:, 1])
        * (ss - computed_edge_lengths[:, 2])
    )
    areas = np.sqrt(np.clip(area_expr, a_min=0.0, a_max=None))

    normal_norms = np.linalg.norm(normals, axis=1)
    valid = normal_norms > 0
    normals[valid] = normals[valid] / normal_norms[valid, None]

    beta_vals = np.arccos(np.clip(normals[:, 2], -1.0, 1.0))
    alpha_vals = np.arctan2(normals[:, 1], normals[:, 0])
    alpha_vals[(normals[:, 1] == 0) & (normals[:, 0] == 0)] = 0.0

    surface_normals[:] = normals
    triangle_edge_lengths[:] = computed_edge_lengths
    triangle_areas[:] = areas
    surface_beta_angles[:] = beta_vals
    surface_alpha_angles[:] = alpha_vals
    return (
        surface_normals,
        triangle_edge_lengths,
        triangle_areas,
        surface_beta_angles,
        surface_alpha_angles,
    )


def other_vector_components(phi_sample_count: int, theta_sample_count: int):
    phi = np.zeros((phi_sample_count, theta_sample_count), dtype=np.double)
    theta = np.zeros((phi_sample_count, theta_sample_count), dtype=np.double)
    U = np.zeros((phi_sample_count, theta_sample_count), dtype=np.double)
    V = np.zeros((phi_sample_count, theta_sample_count), dtype=np.double)
    W = np.zeros((phi_sample_count, theta_sample_count), dtype=np.double)
    e0 = np.zeros(3, dtype=complex)
    Sth = np.zeros((phi_sample_count, theta_sample_count), dtype=np.double)
    Sph = np.zeros((phi_sample_count, theta_sample_count), dtype=np.double)
    return phi, theta, U, V, W, e0, Sth, Sph
