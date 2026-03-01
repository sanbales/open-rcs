from __future__ import annotations

import cmath
import math
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - optional acceleration dependency
    njit = None

from .model_types import (
    AngleSweep,
    BistaticSimulationConfig,
    GeometryData,
    MaterialConfig,
    MonostaticSimulationConfig,
)
from .stl_module import convert_stl

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
INPUT_MODEL = 0
FREQUENCY = 1
STANDART_DEVIATION = 3
RESISTIVITY = 5
MATERIALESPECIFICO = 1
TYPE = 0
THETA = 1
NTRIA = 14
DESCRIPTION = 1
LAYERS = 2
RESULTS_DIR = Path("./results")
NUMBA_AVAILABLE = njit is not None


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


def set_font_option(
    font_size: int = SMALL_SIZE,
    axes_title: int = MEDIUM_SIZE,
    axes_label: int = SMALL_SIZE,
    xtick_label: int = SMALL_SIZE,
    ytick_label: int = SMALL_SIZE,
    legend_size: int = SMALL_SIZE,
    figure_title: int = BIGGER_SIZE,
) -> None:
    """Configure global matplotlib font sizes."""
    plt.rc("font", size=font_size)
    plt.rc("axes", titlesize=axes_title)
    plt.rc("axes", labelsize=axes_label)
    plt.rc("xtick", labelsize=xtick_label)
    plt.rc("ytick", labelsize=ytick_label)
    plt.rc("legend", fontsize=legend_size)
    plt.rc("figure", titlesize=figure_title)


def extract_coordinates_data(rs_value: float) -> GeometryData:
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


def _parse_param(value: str) -> float | str:
    value = value.strip()
    try:
        return float(value)
    except ValueError:
        return value


def get_params_from_file(
    method: str,
) -> MonostaticSimulationConfig | BistaticSimulationConfig:
    """Read input files and return a typed simulation configuration."""
    input_data_file = Path(f"./input_files/input_data_file_{method}.dat")
    param_list: list[float | str] = []
    with input_data_file.open("r", encoding="utf-8") as params:
        for row in params:
            row = row.strip()
            if row and not row.startswith("#"):
                param_list.append(_parse_param(row))

    param_list[FREQUENCY] = float(param_list[FREQUENCY]) * 1e9
    convert_stl(Path("./stl_models") / str(param_list[INPUT_MODEL]))

    if int(param_list[RESISTIVITY]) != MATERIALESPECIFICO:
        param_list[-1] = "matrl.txt"

    angle_sweep = AngleSweep(
        phi_start_deg=float(param_list[6]),
        phi_stop_deg=float(param_list[7]),
        phi_step_deg=float(param_list[8]),
        theta_start_deg=float(param_list[9]),
        theta_stop_deg=float(param_list[10]),
        theta_step_deg=float(param_list[11]),
    )
    material = MaterialConfig(
        resistivity_mode=int(param_list[RESISTIVITY]),
        material_path=str(param_list[-1]),
    )

    if method == "monostatic":
        return MonostaticSimulationConfig(
            input_model=str(param_list[INPUT_MODEL]),
            frequency_hz=float(param_list[FREQUENCY]),
            correlation_distance_m=float(param_list[2]),
            standard_deviation_m=float(param_list[3]),
            incident_polarization=int(param_list[4]),
            angle_sweep=angle_sweep,
            material=material,
        )
    if method == "bistatic":
        return BistaticSimulationConfig(
            input_model=str(param_list[INPUT_MODEL]),
            frequency_hz=float(param_list[FREQUENCY]),
            correlation_distance_m=float(param_list[2]),
            standard_deviation_m=float(param_list[3]),
            incident_polarization=int(param_list[4]),
            angle_sweep=angle_sweep,
            incident_theta_deg=float(param_list[12]),
            incident_phi_deg=float(param_list[13]),
            material=material,
        )
    raise ValueError("method must be 'monostatic' or 'bistatic'.")


def read_coordinates(path: str | Path = "./coordinates.txt"):
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


def read_facets(rs: float, path: str | Path = "./facets.txt"):
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


def plot_triangle_model(
    input_model, vind, x, y, z, xpts, ypts, zpts, nverts, ntria, node1, node2, node3, nfc
):
    fig = plt.figure(1)
    fig.suptitle(f"Triangle Model of Target: {input_model}")
    ilabv = "n"
    ilabf = "n"  # label vertices and faces
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    for i in range(ntria):
        X = [x[vind[i, 0] - 1], x[vind[i, 1] - 1], x[vind[i, 2] - 1], x[vind[i, 0] - 1]]
        Y = [y[vind[i, 0] - 1], y[vind[i, 1] - 1], y[vind[i, 2] - 1], y[vind[i, 0] - 1]]
        Z = [z[vind[i, 0] - 1], z[vind[i, 1] - 1], z[vind[i, 2] - 1], z[vind[i, 0] - 1]]
        ax.plot(X, Y, Z)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    xmax = max(xpts)
    xmin = min(xpts)
    ymax = max(ypts)
    ymin = min(ypts)
    zmax = max(zpts)
    zmin = min(zpts)

    # Define the largest axis interval to preserve aspect ratio
    x_range = xmax - xmin
    y_range = ymax - ymin
    z_range = zmax - zmin
    max_range = max(x_range, y_range, z_range)

    # Adjust each axis limits to keep the same scale
    ax.set_xlim([xmin, xmin + max_range])
    ax.set_ylim([ymin, ymin + max_range])
    ax.set_zlim([zmin, zmin + max_range])

    # Keep equal scaling across x, y, and z axes
    ax.set_box_aspect([1, 1, 1])

    if ilabv == "y":
        for i in range(nverts):
            ax.text(x[i] - max(x) / 20, y[i] - max(y) / 20, z[i], str(i + 1))

    if ilabf == "y":
        for i in range(ntria):
            xav = (xpts[node1[i] - 1] + xpts[node2[i] - 1] + xpts[node3[i] - 1]) / 3
            yav = (ypts[node1[i] - 1] + ypts[node2[i] - 1] + ypts[node3[i] - 1]) / 3
            zav = (zpts[node1[i] - 1] + zpts[node2[i] - 1] + zpts[node3[i] - 1]) / 3
            ax.text(xav, yav, zav, str(nfc[i]))
    # return xmin, ymin, zmin, xmax, ymax, zmax

    # plot parameters
    # param = plot_parameters("Monostatic",freq,wave,corr,delstd, pol,ntria,pstart,pstop,delp,tstart,tstop,delt)

    # save plots
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    fig_name = str(RESULTS_DIR / f"temp_{now}.jpg")
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(fig_name, bbox_inches=extent)
    plt.close()

    return fig_name


def direction_cosines(
    surface_alpha_angles: np.ndarray,
    surface_beta_angles: np.ndarray,
    global_direction_vector: np.ndarray,
    triangle_index: int,
):
    transform_z = np.array(
        [
            [
                math.cos(surface_alpha_angles[triangle_index]),
                math.sin(surface_alpha_angles[triangle_index]),
                0,
            ],
            [
                -math.sin(surface_alpha_angles[triangle_index]),
                math.cos(surface_alpha_angles[triangle_index]),
                0,
            ],
            [0, 0, 1],
        ]
    )
    transform_y = np.array(
        [
            [
                math.cos(surface_beta_angles[triangle_index]),
                0,
                -math.sin(surface_beta_angles[triangle_index]),
            ],
            [0, 1, 0],
            [
                math.sin(surface_beta_angles[triangle_index]),
                0,
                math.cos(surface_beta_angles[triangle_index]),
            ],
        ]
    )
    local_direction_step1 = np.dot(transform_z, np.transpose(global_direction_vector))
    local_direction = np.dot(transform_y, local_direction_step1)
    local_u = local_direction[0]
    local_v = local_direction[1]
    local_w = local_direction[2]
    return local_u, local_v, local_w, transform_z, transform_y


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

    def calculate_phr0():
        if pstart == pstop:
            return pstart * rad

    def calculate_thr0():
        if tstart == tstop:
            return tstart * rad

    ip = calculate_ip()
    it = calculate_it()
    phr0 = calculate_phr0()
    thr0 = calculate_thr0()

    Area = np.empty(ntria, np.double)
    alpha = np.empty(ntria, np.double)
    beta = np.empty(ntria, np.double)
    N = np.empty([ntria, 3], np.double)
    d = np.empty([ntria, 3], np.double)

    return Area, alpha, beta, N, d, ip, it, cpi, spi, sti, cti, ui, vi, wi, D0i, uui, vvi, wwi, Ri


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


def save_list_in_file(especific_list: list, especific_file: str) -> None:
    especific_list_str = []
    for row in especific_list:
        entry_str = str(row[TYPE]) + "," + str(row[DESCRIPTION])

        for layer in row[LAYERS:]:
            for i in range(len(layer)):
                entry_str = entry_str + "," + str(layer[i])

        especific_list_str.append(entry_str)

    with open(especific_file, "w") as file:
        for row in especific_list_str:
            file.write(row + "\n")


def get_entries_from_material_file(ntria: int, matrlpath: str) -> list:
    try:
        with open(matrlpath) as file:
            matrl = get_material_properties_from_file(file)
    except Exception as exc:
        raise FileNotFoundError("Matrl file not found.") from exc

    if len(matrl) != ntria:
        raise ValueError("Number of entrys in matrl diferent from number of facets.")

    return matrl


def get_material_properties_from_file(filename) -> list:
    material_text_list = []
    for row in filename:
        material_text_list.append(row)
    return convert_material_textlist_to_list(material_text_list)


def convert_material_textlist_to_list(text_list: str) -> list:
    matrl = []
    for row in text_list:
        entrys = row.strip("\n")
        entrys = entrys.split(",")
        formatedEntrys = [entrys[TYPE], entrys[DESCRIPTION]]
        layer = []

        for index, entry in enumerate(entrys[LAYERS:]):
            layer.append(float(entry))
            if (index + 1) % 5 == 0:
                formatedEntrys.append(layer)
                layer = []

        matrl.append(formatedEntrys)
    return matrl


def erase_widges_from_table(table_frame):
    for widget in table_frame.winfo_children():
        try:
            widget.destroy()
        except Exception:
            pass


def get_surface_layers(facet_material_properties):
    layers = facet_material_properties[2:]
    new_data = []
    for i, layer in enumerate(layers):
        prop = [i + 1, facet_material_properties[0], facet_material_properties[1]]
        for x in layer:
            prop.append(x)
        new_data.append(prop)
    return new_data


def rotation_transform_matrix(alpha, beta):
    T1 = np.array(
        [[np.cos(alpha), np.sin(alpha), 0], [-np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]]
    )
    T2 = np.array([[np.cos(beta), 0, -np.sin(beta)], [0, 1, 0], [np.sin(beta), 0, np.cos(beta)]])
    return np.dot(T2, T1)


def refl_coeff(er1, mr1, er2, mr2, thetai):
    m0 = 4 * np.pi * 1e-7  # vacuum permeability
    e0 = 8.854e-12  # vacuum permittivity

    TIR = 0
    sinthetat = np.sin(thetai) * np.sqrt(
        np.real(er1) * np.real(mr1) / (np.real(er2) * np.real(mr2))
    )

    # Total internal reflection (TIR) check
    if sinthetat > 1:
        TIR = 1
        thetat = np.pi / 2  # critical angle
    else:
        thetat = np.arcsin(sinthetat)

    # Compute n1 and n2 (refractive indices)
    n1 = np.sqrt(mr1 * m0 / (er1 * e0))
    n2 = np.sqrt(mr2 * m0 / (er2 * e0))

    # Compute gammaperp and gammapar
    gammaperp = (n2 * np.cos(thetai) - n1 * np.cos(thetat)) / (
        n2 * np.cos(thetai) + n1 * np.cos(thetat)
    )
    gammapar = (n2 * np.cos(thetat) - n1 * np.cos(thetai)) / (
        n2 * np.cos(thetat) + n1 * np.cos(thetai)
    )

    return gammapar, gammaperp, thetat, TIR


def spher2cart(sphericalVector: np.array) -> np.array:
    R = sphericalVector[0]
    theta = sphericalVector[1]
    phi = sphericalVector[2]
    x = R * np.sin(theta) * np.cos(phi)
    y = R * np.sin(theta) * np.sin(phi)
    z = R * np.cos(theta)

    return np.array([x, y, z])


def cart2spher(cartVector: np.array) -> np.array:
    x = cartVector[0]
    y = cartVector[1]
    z = cartVector[2]
    R = np.sqrt(x**2 + y**2 + z**2)
    theta = math.atan2(np.sqrt(x**2 + y**2), z)
    phi = math.atan2(y, x)
    return np.array([R, theta, phi])


def spherical_global_to_local(
    spherical_vector: np.ndarray,
    transform_matrix_global_to_local: np.ndarray,
) -> np.ndarray:
    cartesian_vector = spher2cart(spherical_vector)
    cartesian_vector = np.dot(transform_matrix_global_to_local, cartesian_vector)
    return cart2spher(cartesian_vector)


def refl_coeff_composite(
    thri: float, phrii: float, alpha: float, beta: float, freq: float, matrlLine: list
) -> tuple[float, float]:
    layer = matrlLine[LAYERS]

    er = layer[0] - 1j * layer[1] * layer[0]
    mr = layer[2] - 1j * layer[3]
    t = layer[4] * 0.001

    T21 = rotation_transform_matrix(alpha, beta)

    sphericalVector = spherical_global_to_local(np.array([1, thri, phrii]), T21)

    G1par, G1perp, thetat, TIR = refl_coeff(1, 1, er, mr, sphericalVector[THETA])

    G2par = -G1par
    G2perp = -G1perp

    v = 3e8 / np.sqrt(np.real(er) * np.real(mr))
    wave = v / freq
    b1 = 2 * np.pi / wave
    phase = b1 * t

    M1par = np.array(
        [
            [np.exp(1j * phase), G1par * np.exp(-1j * phase)],
            [G1par * np.exp(1j * phase), np.exp(-1j * phase)],
        ]
    )

    M1perp = np.array(
        [
            [np.exp(1j * phase), G1perp * np.exp(-1j * phase)],
            [G1perp * np.exp(1j * phase), np.exp(-1j * phase)],
        ]
    )

    M2par = np.array([[1, G2par], [G2par, 1]])

    M2perp = np.array([[1, G2perp], [G2perp, 1]])

    Mpar = np.dot(M1par, M2par)
    Mperp = np.dot(M1perp, M2perp)

    RCpar = Mpar[1, 0] / Mpar[0, 0]
    RCperp = Mperp[1, 0] / Mperp[0, 0]

    return RCperp, RCpar


def refl_coeff_composite_layer_on_pec(
    thri: float, phrii: float, alpha: float, beta: float, freq: float, matrlLine: list
) -> tuple[float, float]:
    layers = matrlLine[LAYERS:]

    T21 = rotation_transform_matrix(alpha, beta)
    sphericalVector = spherical_global_to_local(np.array([1, thri, phrii]), T21)

    PEC = np.array([[1, 0], [-1, 0]])

    WMatrix_par = np.eye(2)
    WMatrix_perp = np.eye(2)
    Z0 = 1
    wave = 3e8 / freq
    B0 = 2 * np.pi / wave
    thinc = sphericalVector[THETA]

    Z_par = []
    Z_perp = []
    Beta = []
    gamma_par = []
    gamma_perp = []
    tau_par = []
    tau_perp = []

    for i, layer in enumerate(layers):
        erp = layer[0]
        erdp = erp * layer[1]
        erc = erp - 1j * erdp
        urp = layer[2]
        urdp = layer[3]
        urc = urp - 1j * urdp
        t = layer[4] * 1e-3

        Z_par.append(np.sqrt(erc / urc - np.sin(thinc) ** 2) / (erc / urc * np.cos(thinc)))
        Z_perp.append(np.cos(thinc) / np.sqrt(erc / urc - np.sin(thinc) ** 2))
        Beta.append(2 * np.pi / (wave / np.sqrt(np.real(erc) * np.real(urc))))

        if i == 0:
            gamma_par.append((Z_par[i] - Z0) / (Z_par[i] + Z0))
            gamma_perp.append((Z_perp[i] - Z0) / (Z_perp[i] + Z0))
        else:
            gamma_par.append((Z_par[i] - Z_par[i - 1]) / (Z_par[i] + Z_par[i - 1]))
            gamma_perp.append((Z_perp[i] - Z_perp[i - 1]) / (Z_perp[i] + Z_perp[i - 1]))

        tau_par.append(1 + gamma_par[i])
        tau_perp.append(1 + gamma_perp[i])
        phi_calc = B0 * t * np.sqrt(erc * urc - np.sin(thinc) ** 2)

        T_par = np.array(
            [
                [np.exp(1j * phi_calc), gamma_par[i] * np.exp(-1j * phi_calc)],
                [gamma_par[i] * np.exp(1j * phi_calc), np.exp(-1j * phi_calc)],
            ]
        )

        WMatrix_par = 1 / tau_par[i] * WMatrix_par @ T_par

        T_perp = np.array(
            [
                [np.exp(1j * phi_calc), gamma_perp[i] * np.exp(-1j * phi_calc)],
                [gamma_perp[i] * np.exp(1j * phi_calc), np.exp(-1j * phi_calc)],
            ]
        )

        WMatrix_perp = 1 / tau_perp[i] * WMatrix_perp @ T_perp

    WMatrix_par = WMatrix_par @ PEC
    WMatrix_perp = WMatrix_perp @ PEC

    RCperp = WMatrix_perp[1, 0] / WMatrix_perp[0, 0]
    RCpar = WMatrix_par[1, 0] / WMatrix_par[0, 0]
    return RCperp, RCpar


def refl_coeff_multi_layers(
    thri: float, phrii: float, alpha: float, beta: float, freq: float, matrlLine: list
) -> tuple[float, float]:

    T21 = rotation_transform_matrix(alpha, beta)
    sphericalVector = spherical_global_to_local(np.array([1, thri, phrii]), T21)
    layers = matrlLine[LAYERS:]
    Mpar = np.eye(2)
    Mperp = np.eye(2)

    er = []
    mr = []
    t = []
    thetat = []

    for i, layer in enumerate(layers):
        er.append(layer[0] - 1j * layer[1] * layer[0])
        mr.append(layer[2] - 1j * layer[3])
        t.append(layer[4] * 0.001)

        if i == 0:
            Gpar, Gperp, thetatI, TIR = refl_coeff(1, 1, er[i], mr[i], sphericalVector[THETA])
        else:
            Gpar, Gperp, thetatI, TIR = refl_coeff(
                er[i - 1], mr[i - 1], er[i], mr[i], thetat[i - 1]
            )

        thetat.append(thetatI)
        v = 3e8 / np.sqrt(np.real(er[i]) * np.real(mr[i]))
        wave = v / freq
        b1 = 2 * np.pi / wave
        phase = b1 * t[i]

        Mpar = Mpar @ np.array(
            [
                [np.exp(1j * phase), Gpar * np.exp(-1j * phase)],
                [Gpar * np.exp(1j * phase), np.exp(-1j * phase)],
            ]
        )
        Mperp = Mperp @ np.array(
            [
                [np.exp(1j * phase), Gperp * np.exp(-1j * phase)],
                [Gperp * np.exp(1j * phase), np.exp(-1j * phase)],
            ]
        )

    Gpar, Gperp, thetatdum, TIR = refl_coeff(er[-1], mr[-1], 1, 1, thetat[-1])

    Mpar = Mpar @ np.array(
        [
            [np.exp(1j * phase), Gpar * np.exp(-1j * phase)],
            [Gpar * np.exp(1j * phase), np.exp(-1j * phase)],
        ]
    )
    Mperp = Mperp @ np.array(
        [
            [np.exp(1j * phase), Gperp * np.exp(-1j * phase)],
            [Gperp * np.exp(1j * phase), np.exp(-1j * phase)],
        ]
    )

    RCpar = Mpar[1, 0] / Mpar[0, 0]
    RCperp = Mperp[1, 0] / Mperp[0, 0]
    return RCperp, RCpar


def refl_coeff_multi_layers_on_pec(
    thri: float, phrii: float, alpha: float, beta: float, freq: float, matrlLine: list
) -> tuple[float, float]:
    T21 = rotation_transform_matrix(alpha, beta)
    sphericalVector = spherical_global_to_local(np.array([1, thri, phrii]), T21)
    layers = matrlLine[LAYERS:]
    Mpar = np.eye(2)
    Mperp = np.eye(2)
    WMatrix_par = np.eye(2)
    WMatrix_perp = np.eye(2)

    PEC = np.array([[1, 0], [-1, 0]])

    Z0 = 1
    wave = 3e8 / freq
    B0 = 2 * np.pi / wave
    thinc = sphericalVector[THETA]

    Z_par = []
    Z_perp = []
    Beta = []
    gamma_par = []
    gamma_perp = []
    tau_par = []
    tau_perp = []

    for i, layer in enumerate(layers):
        erp = layer[0]
        erdp = erp * layer[1]
        erc = erp - 1j * erdp
        urp = layer[2]
        urdp = layer[3]
        urc = urp - 1j * urdp
        t = layer[4] * 1e-3

        Z_par.append(np.sqrt(erc / urc - np.sin(thinc) ** 2) / (erc / urc * np.cos(thinc)))
        Z_perp.append(np.cos(thinc) / np.sqrt(erc / urc - np.sin(thinc) ** 2))
        Beta.append(2 * np.pi / (wave / np.sqrt(np.real(erc) * np.real(urc))))

        if i == 0:
            gamma_par.append((Z_par[i] - Z0) / (Z_par[i] + Z0))
            gamma_perp.append((Z_perp[i] - Z0) / (Z_perp[i] + Z0))
        else:
            gamma_par.append((Z_par[i] - Z_par[i - 1]) / (Z_par[i] + Z_par[i - 1]))
            gamma_perp.append((Z_perp[i] - Z_perp[i - 1]) / (Z_perp[i] + Z_perp[i - 1]))

        tau_par.append(1 + gamma_par[i])
        tau_perp.append(1 + gamma_perp[i])
        phi_calc = B0 * t * (erc * urc - np.sin(thinc) ** 2) ** 0.5

        T_par = np.array(
            [
                [np.exp(1j * phi_calc), gamma_par[i] * np.exp(-1j * phi_calc)],
                [gamma_par[i] * np.exp(1j * phi_calc), np.exp(-1j * phi_calc)],
            ]
        )

        WMatrix_par = 1 / tau_par[i] * WMatrix_par @ T_par

        T_perp = np.array(
            [
                [np.exp(1j * phi_calc), gamma_perp[i] * np.exp(-1j * phi_calc)],
                [gamma_perp[i] * np.exp(1j * phi_calc), np.exp(-1j * phi_calc)],
            ]
        )

        WMatrix_perp = 1 / tau_perp[i] * WMatrix_perp @ T_perp

    WMatrix_par = WMatrix_par @ PEC
    WMatrix_perp = WMatrix_perp @ PEC

    RCpar = WMatrix_par[1, 0] / WMatrix_par[0, 0]
    RCperp = WMatrix_perp[1, 0] / WMatrix_perp[0, 0]

    return RCperp, RCpar


def get_reflection_coeff_from_material(
    thri: float, phrii: float, alpha: float, beta: float, freq: float, matrlLine: list
) -> tuple[float, float]:
    RCperp = 0
    RCpar = 0

    if matrlLine[TYPE] == "PEC":
        RCperp = -1
        RCpar = -1

    elif matrlLine[TYPE] == "Composito":
        RCperp, RCpar = refl_coeff_composite(thri, phrii, alpha, beta, freq, matrlLine)

    elif matrlLine[TYPE] == "Camada de Composito em PEC":
        RCperp, RCpar = refl_coeff_composite_layer_on_pec(thri, phrii, alpha, beta, freq, matrlLine)

    elif matrlLine[TYPE] == "Multiplas Camadas":
        RCperp, RCpar = refl_coeff_multi_layers(thri, phrii, alpha, beta, freq, matrlLine)

    elif matrlLine[TYPE] == "Multiplas Camadas em PEC":
        RCperp, RCpar = refl_coeff_multi_layers_on_pec(thri, phrii, alpha, beta, freq, matrlLine)

    return RCperp, RCpar


def reflection_coefficients(
    rs: int,
    index: int,
    th2: float,
    thri: float,
    phrii: float,
    alpha: float,
    beta: float,
    freq: float,
    matrl: list,
    local_cos_theta: float | None = None,
) -> tuple[float, float]:
    perp = 0
    para = 0

    if rs == MATERIALESPECIFICO:
        perp, para = get_reflection_coeff_from_material(
            thri, phrii, alpha, beta, freq, matrl[index]
        )
    else:
        cosine_local_theta = math.cos(th2) if local_cos_theta is None else local_cos_theta
        perp = -1 / (2 * rs * cosine_local_theta + 1)  # local TE polarization
        para = 0  # local TM polarization
        if (2 * rs + cosine_local_theta) != 0:
            para = -cosine_local_theta / (2 * rs + cosine_local_theta)

    return perp, para


def incident_field_spherical_coordinates(th2, e2, phi2):
    Et2 = (
        e2[0] * math.cos(th2) * math.cos(phi2)
        + e2[1] * math.cos(th2) * math.sin(phi2)
        - e2[2] * math.sin(th2)
    )
    Ep2 = -e2[0] * math.sin(phi2) + e2[1] * math.cos(phi2)
    return Et2, Ep2


def bi_incident_field_spherical_coordinates(cpi2, cti2, sti2, spi2, e2):
    Et2 = e2[0] * cti2 * cpi2 + e2[1] * cti2 * spi2 - e2[2] * sti2
    Ep2 = -e2[0] * spi2 + e2[1] * cpi2
    return Et2, Ep2


def final_plot(
    phi_sample_count: int,
    theta_sample_count: int,
    phi_grid_deg: np.ndarray,
    wavelength_m: float,
    theta_grid_deg: np.ndarray,
    rcs_min_db: float,
    rcs_max_db: float,
    rcs_theta_db: np.ndarray,
    rcs_phi_db: np.ndarray,
    direction_cosine_u_grid: np.ndarray,
    direction_cosine_v_grid: np.ndarray,
    timestamp: str,
    input_model: str,
    mode: str,
) -> str:
    if phi_sample_count == 1:
        plt.figure(1)
        plt.suptitle(f"RCS Simulation IR Signature - {mode}")
        plt.title(
            f"target: {input_model}   solid: theta     dashed: phi     "
            f"phi= {phi_grid_deg[0][0]}    wave (m): {round(wavelength_m, 6)}"
        )
        plt.xlabel(f"{mode} Angle, theta (deg)")
        plt.ylabel("RCS (dBsm)")
        plt.axis([np.min(theta_grid_deg), np.max(theta_grid_deg), rcs_min_db, rcs_max_db])
        plt.plot(theta_grid_deg[0], rcs_theta_db[0])
        plt.plot(theta_grid_deg[0], rcs_phi_db[0], linewidth=2, linestyle="dashed")
        plt.grid(True)

    if theta_sample_count == 1:
        plt.figure(1)
        plt.suptitle(f"RCS Simulation IR Signature - {mode}")
        plt.title(
            f"target: {input_model}   solid: theta     dashed: phi     "
            f"theta= {theta_grid_deg[0][0]}    wave (m): {round(wavelength_m, 6)}"
        )
        plt.xlabel(f"{mode} Angle, phi (deg)")
        plt.ylabel("RCS (dBsm)")
        plt.axis([np.min(phi_grid_deg), np.max(phi_grid_deg), rcs_min_db, rcs_max_db])
        plt.plot(phi_grid_deg, rcs_theta_db)
        plt.plot(phi_grid_deg, rcs_phi_db, linewidth=2, linestyle="dashed")
        plt.grid(True)

    if phi_sample_count > 1 and theta_sample_count > 1:
        contour_levels = [-20, 0]
        fig = plt.figure(1)
        fig.suptitle(f"RCS Simulation IR Signature - {mode}")

        ax = fig.add_subplot(2, 3, 2)
        if mode == "Monostatic":
            cp = ax.contour(direction_cosine_u_grid, direction_cosine_v_grid, rcs_theta_db)
        elif mode == "Bistatic":
            cp = ax.contour(
                direction_cosine_u_grid, direction_cosine_v_grid, rcs_theta_db, contour_levels
            )
        ax.set_title("RCS-theta")
        ax.set_xlabel("U")
        ax.set_ylabel("V")
        ax.axis("square")
        cbar = fig.colorbar(cp)
        cbar.set_label("RCS (dBsm)")

        bx = fig.add_subplot(2, 3, 5)
        if mode == "Monostatic":
            cp = bx.contour(direction_cosine_u_grid, direction_cosine_v_grid, rcs_phi_db)
        elif mode == "Bistatic":
            cp = bx.contour(
                direction_cosine_u_grid, direction_cosine_v_grid, rcs_phi_db, contour_levels
            )
        bx.set_title("RCS-phi")
        bx.set_xlabel("U")
        bx.set_ylabel("V")
        bx.axis("square")
        cbar = fig.colorbar(cp)
        cbar.set_label("RCS (dBsm)")

        fig.subplots_adjust(wspace=0)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_name = str(RESULTS_DIR / f"temp_{timestamp}.png")
    plt.savefig(plot_name)
    # plt.show()
    plt.close()
    return plot_name


def generate_result_files(
    theta_grid_deg: np.ndarray,
    rcs_theta_db: np.ndarray,
    phi_grid_deg: np.ndarray,
    rcs_phi_db: np.ndarray,
    parameter_text: str,
    phi_sample_count: int,
):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = str(RESULTS_DIR / f"temp_{now}.dat")

    with open(file_name, "w", encoding="utf-8") as result_file:
        result_file.write(f"RCS SIMULATOR RESULTS {now}\n")
        result_file.write("\nSimulation Parameters:\n" + parameter_text)
        result_file.write("\nSimulation Results IR Signature:")
        result_file.write("\nTheta (deg):\n")
        for i1 in range(phi_sample_count):
            result_file.write(f"{theta_grid_deg[i1]}\n")
        result_file.write("\nRCS Theta (dBsm):\n")
        for i1 in range(phi_sample_count):
            result_file.write(f"{rcs_theta_db[i1]}\n")
        result_file.write("\nPhi (deg):\n")
        for i1 in range(phi_sample_count):
            result_file.write(f"{phi_grid_deg[i1]}\n")
        result_file.write("\nRCS Phi (dBsm):\n")
        for i1 in range(phi_sample_count):
            result_file.write(f"{rcs_phi_db[i1]}\n")

    return now, file_name


def calculate_ic(
    phase_p: float,
    phase_q: float,
    phase_origin: float,
    taylor_terms: int,
    triangle_area: float,
    incident_amplitude: float,
    taylor_threshold: float,
):
    area_scale = 2.0 * triangle_area
    phase_difference = phase_q - phase_p
    abs_phase_p = abs(phase_p)
    abs_phase_q = abs(phase_q)
    abs_phase_difference = abs(phase_difference)
    exp_phase_origin = cmath.exp(1j * phase_origin)
    # special case 1
    if abs_phase_p < taylor_threshold and abs_phase_q >= taylor_threshold:
        exp_phase_q = cmath.exp(1j * phase_q)
        series_integral = 0.0
        for n in range(taylor_terms + 1):
            series_integral = series_integral + (1j * phase_p) ** n / math.factorial(n) * (
                -incident_amplitude / (n + 1)
                + exp_phase_q * (incident_amplitude * taylor_g(n, -phase_q))
            )
        area_integral_value = series_integral * area_scale * exp_phase_origin / (1j * phase_q)
    # special case 2
    elif abs_phase_p < taylor_threshold and abs_phase_q < taylor_threshold:
        series_integral = 0.0
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
        series_integral = 0.0
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
        series_integral = 0.0
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
    def _factorial_numba(value: int) -> int:
        result = 1
        for index in range(2, value + 1):
            result *= index
        return result

    @njit(cache=True)
    def _taylor_g_numba(n: int, w: float) -> complex:
        jw = 1j * w
        exp_jw = np.exp(jw)
        g = (exp_jw - 1) / jw
        if n > 0:
            for m in range(1, n + 1):
                go = g
                g = (exp_jw - m * go) / jw
        return g

    @njit(cache=True)
    def _calculate_ic_numba(
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
        exp_phase_origin = np.exp(1j * phase_origin)

        if abs_phase_p < taylor_threshold and abs_phase_q >= taylor_threshold:
            exp_phase_q = np.exp(1j * phase_q)
            series_integral = 0.0 + 0.0j
            for n in range(taylor_terms + 1):
                series_integral = series_integral + (1j * phase_p) ** n / _factorial_numba(n) * (
                    -incident_amplitude / (n + 1)
                    + exp_phase_q * (incident_amplitude * _taylor_g_numba(n, -phase_q))
                )
            return series_integral * area_scale * exp_phase_origin / (1j * phase_q)

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
            return series_integral * area_scale * exp_phase_origin

        if abs_phase_p >= taylor_threshold and abs_phase_q < taylor_threshold:
            exp_phase_p = np.exp(1j * phase_p)
            series_integral = 0.0 + 0.0j
            for n in range(taylor_terms + 1):
                series_integral = series_integral + (1j * phase_q) ** n / _factorial_numba(
                    n
                ) * incident_amplitude * (_taylor_g_numba(n + 1, -phase_p) / (n + 1))
            return series_integral * area_scale * exp_phase_origin * exp_phase_p

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
            return series_integral * area_scale * exp_phase_origin / (1j * phase_q)

        exp_phase_p = np.exp(1j * phase_p)
        exp_phase_q = np.exp(1j * phase_q)
        return (
            area_scale
            * exp_phase_origin
            * (
                exp_phase_p * incident_amplitude / (phase_p * phase_difference)
                - exp_phase_q * incident_amplitude / (phase_q * phase_difference)
                - incident_amplitude / (phase_p * phase_q)
            )
        )

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
    ) -> tuple[complex, complex, float, float]:
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

            resistivity = resistivity_values[triangle_index]
            reflection_perpendicular = -1.0 / (2.0 * resistivity * local_cosine_theta + 1.0)
            reflection_parallel = 0.0
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
    ) -> tuple[complex, complex, float, float]:
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

            resistivity = resistivity_values[triangle_index]
            reflection_perpendicular = -1.0 / (
                2.0 * resistivity * observation_local_cosine_theta + 1.0
            )
            reflection_parallel = 0.0
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

    def _accumulate_monostatic_sample_numba(*_args, **_kwargs):
        raise RuntimeError("Numba acceleration requested but numba is not installed.")

    def _accumulate_bistatic_sample_numba(*_args, **_kwargs):
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
    return _accumulate_monostatic_sample_numba(
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
    return _accumulate_bistatic_sample_numba(
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
    )


def plot_parameters(
    mode: str,
    frequency_hz: float,
    wavelength_m: float,
    correlation_distance_m: float,
    standard_deviation_m: float,
    polarization_label: str,
    triangle_count: int,
    phi_start_deg: float,
    phi_stop_deg: float,
    phi_step_deg: float,
    theta_start_deg: float,
    theta_stop_deg: float,
    theta_step_deg: float,
):
    param = f"    Mode: {mode}\n\
    Radar Frequency (GHz): {frequency_hz / 1e9}\n\
    Wavelength (m): {wavelength_m}\n\
    Correlation distance (m): {correlation_distance_m}\n\
    Standard Deviation (m): {standard_deviation_m}\n\
    Incident wave polarization: {polarization_label}\n\
    Start phi angle (degrees): {phi_start_deg}\n\
    Stop phi angle (degrees): {phi_stop_deg}\n\
    Phi increment step (degrees): {phi_step_deg}\n\
    Start theta angle (degrees): {theta_start_deg}\n\
    Stop theta angle (degrees): {theta_stop_deg}\n\
    Phi increment step (degrees): {theta_step_deg}\n"
    return param


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


def plot_limits(rcs_theta_db: np.ndarray, rcs_phi_db: np.ndarray):
    rcs_max_db = max(np.max(rcs_theta_db), np.max(rcs_phi_db))
    rounded_max_db = (np.floor(rcs_max_db / 5) + 1) * 5
    rcs_min_db = min(np.min(rcs_theta_db), np.min(rcs_phi_db))
    return rounded_max_db, rcs_min_db


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
