"""Matplotlib plotting helpers and result file output for Open RCS."""

from __future__ import annotations

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from .constants import RESULTS_DIR, FontSize


def _expanded_limits(min_value: float, max_value: float) -> tuple[float, float]:
    """Expand degenerate axis limits to avoid singular matplotlib transforms."""
    if np.isclose(min_value, max_value):
        padding = max(0.5, abs(min_value) * 0.01)
        return min_value - padding, max_value + padding
    return min_value, max_value


def set_font_option(
    font_size: int = int(FontSize.SMALL),
    axes_title: int = int(FontSize.MEDIUM),
    axes_label: int = int(FontSize.SMALL),
    xtick_label: int = int(FontSize.SMALL),
    ytick_label: int = int(FontSize.SMALL),
    legend_size: int = int(FontSize.SMALL),
    figure_title: int = int(FontSize.LARGE),
) -> None:
    """Configure global matplotlib font sizes."""
    plt.rc("font", size=font_size)
    plt.rc("axes", titlesize=axes_title)
    plt.rc("axes", labelsize=axes_label)
    plt.rc("xtick", labelsize=xtick_label)
    plt.rc("ytick", labelsize=ytick_label)
    plt.rc("legend", fontsize=legend_size)
    plt.rc("figure", titlesize=figure_title)


def plot_triangle_model(
    input_model,
    vind,
    x,
    y,
    z,
    xpts,
    ypts,
    zpts,
    nverts,
    ntria,
    node1,
    node2,
    node3,
    nfc,
):
    """Render and save a 3D triangle-wireframe view of the input geometry."""
    fig = plt.figure(1)
    fig.suptitle(f"Triangle Model of Target: {input_model}")
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

    x_range = xmax - xmin
    y_range = ymax - ymin
    z_range = zmax - zmin
    max_range = max(x_range, y_range, z_range)

    ax.set_xlim([xmin, xmin + max_range])
    ax.set_ylim([ymin, ymin + max_range])
    ax.set_zlim([zmin, zmin + max_range])
    ax.set_box_aspect([1, 1, 1])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    fig_name = str(RESULTS_DIR / f"temp_{now}.jpg")
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(fig_name, bbox_inches=extent)
    plt.close()

    return fig_name


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
    """Generate and save the final 2D RCS visualization for the selected sweep."""
    if phi_sample_count == 1:
        x_min, x_max = _expanded_limits(
            float(np.min(theta_grid_deg)),
            float(np.max(theta_grid_deg)),
        )
        y_min, y_max = _expanded_limits(float(rcs_min_db), float(rcs_max_db))
        plt.figure(1)
        plt.suptitle(f"RCS Simulation IR Signature - {mode}")
        plt.title(
            f"target: {input_model}   solid: theta     dashed: phi     "
            f"phi= {phi_grid_deg[0][0]}    wave (m): {round(wavelength_m, 6)}"
        )
        plt.xlabel(f"{mode} Angle, theta (deg)")
        plt.ylabel("RCS (dBsm)")
        plt.axis(
            (
                x_min,
                x_max,
                y_min,
                y_max,
            )
        )
        plt.plot(theta_grid_deg[0], rcs_theta_db[0])
        plt.plot(theta_grid_deg[0], rcs_phi_db[0], linewidth=2, linestyle="dashed")
        plt.grid(True)

    if theta_sample_count == 1:
        x_min, x_max = _expanded_limits(
            float(np.min(phi_grid_deg)),
            float(np.max(phi_grid_deg)),
        )
        y_min, y_max = _expanded_limits(float(rcs_min_db), float(rcs_max_db))
        plt.figure(1)
        plt.suptitle(f"RCS Simulation IR Signature - {mode}")
        plt.title(
            f"target: {input_model}   solid: theta     dashed: phi     "
            f"theta= {theta_grid_deg[0][0]}    wave (m): {round(wavelength_m, 6)}"
        )
        plt.xlabel(f"{mode} Angle, phi (deg)")
        plt.ylabel("RCS (dBsm)")
        plt.axis(
            (
                x_min,
                x_max,
                y_min,
                y_max,
            )
        )
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
            cp = ax.contour(direction_cosine_u_grid, direction_cosine_v_grid, rcs_theta_db, contour_levels)
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
            cp = bx.contour(direction_cosine_u_grid, direction_cosine_v_grid, rcs_phi_db, contour_levels)
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
    """Write simulation parameters and RCS arrays to a timestamped results file."""
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
    """Format simulation settings into a text block for report output."""
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


def plot_limits(rcs_theta_db: np.ndarray, rcs_phi_db: np.ndarray):
    """Compute display bounds from theta/phi RCS grids."""
    rcs_max_db = max(np.max(rcs_theta_db), np.max(rcs_phi_db))
    rounded_max_db = (np.floor(rcs_max_db / 5) + 1) * 5
    rcs_min_db = min(np.min(rcs_theta_db), np.min(rcs_phi_db))
    return rounded_max_db, rcs_min_db
