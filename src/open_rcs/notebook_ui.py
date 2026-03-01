"""Jupyter widget UI for interactive Open RCS simulation and visualization."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np
from stl import mesh

from . import rcs_functions as rf
from .guidance import estimate_roughness_correlation_guidance
from .model_types import (
    AngleSweep,
    BistaticSimulationConfig,
    GeometryData,
    MaterialConfig,
    MonostaticSimulationConfig,
    RadarBand,
    RcsComputationResult,
)
from .rcs_bistatic import simulate_bistatic
from .rcs_monostatic import simulate_monostatic


def _load_mesh_vertices_faces(stl_path: Path) -> tuple[np.ndarray, np.ndarray]:
    stl_mesh = mesh.Mesh.from_file(str(stl_path))
    triangles = np.asarray(stl_mesh.vectors, dtype=float)
    flat_vertices = triangles.reshape(-1, 3)
    unique_vertices, inverse_indices = np.unique(flat_vertices, axis=0, return_inverse=True)
    faces = inverse_indices.reshape(-1, 3)
    return unique_vertices, faces


def _rotation_matrix_xyz(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Build a right-handed XYZ rotation matrix from Euler angles in degrees."""
    roll_rad, pitch_rad, yaw_rad = np.deg2rad([roll_deg, pitch_deg, yaw_deg])

    rot_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(roll_rad), -np.sin(roll_rad)],
            [0.0, np.sin(roll_rad), np.cos(roll_rad)],
        ],
        dtype=float,
    )
    rot_y = np.array(
        [
            [np.cos(pitch_rad), 0.0, np.sin(pitch_rad)],
            [0.0, 1.0, 0.0],
            [-np.sin(pitch_rad), 0.0, np.cos(pitch_rad)],
        ],
        dtype=float,
    )
    rot_z = np.array(
        [
            [np.cos(yaw_rad), -np.sin(yaw_rad), 0.0],
            [np.sin(yaw_rad), np.cos(yaw_rad), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return cast(np.ndarray, rot_z @ rot_y @ rot_x)


def _rotate_vertices(
    vertices: np.ndarray, roll_deg: float, pitch_deg: float, yaw_deg: float
) -> np.ndarray:
    """Rotate vertices around the mesh centroid."""
    if vertices.size == 0:
        return vertices
    rotation = _rotation_matrix_xyz(roll_deg, pitch_deg, yaw_deg)
    centroid = np.mean(vertices, axis=0, keepdims=True)
    return cast(np.ndarray, (vertices - centroid) @ rotation.T + centroid)


def _build_rotated_geometry(
    geometry_data: GeometryData,
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
) -> GeometryData:
    """Return a rotated copy of geometry data for simulation."""
    rotated_vertices = _rotate_vertices(
        np.asarray(geometry_data.vertex_coordinates, dtype=float),
        roll_deg,
        pitch_deg,
        yaw_deg,
    )
    return GeometryData(
        x=rotated_vertices[:, 0].copy(),
        y=rotated_vertices[:, 1].copy(),
        z=rotated_vertices[:, 2].copy(),
        x_points=rotated_vertices[:, 0].copy(),
        y_points=rotated_vertices[:, 1].copy(),
        z_points=rotated_vertices[:, 2].copy(),
        n_vertices=geometry_data.n_vertices,
        facet_numbers=np.array(geometry_data.facet_numbers, copy=True),
        node1=np.array(geometry_data.node1, copy=True),
        node2=np.array(geometry_data.node2, copy=True),
        node3=np.array(geometry_data.node3, copy=True),
        illumination_flag_mode=geometry_data.illumination_flag_mode,
        illumination_flags=np.array(geometry_data.illumination_flags, copy=True),
        resistivity_values=np.array(geometry_data.resistivity_values, copy=True),
        n_triangles=geometry_data.n_triangles,
        vertex_indices=np.array(geometry_data.vertex_indices, copy=True),
        vertex_coordinates=rotated_vertices,
    )


def _select_rcs_component(simulation_result: RcsComputationResult, component: str) -> np.ndarray:
    if component == "theta":
        return simulation_result.rcs_theta_db
    if component == "phi":
        return simulation_result.rcs_phi_db
    return cast(
        np.ndarray, np.maximum(simulation_result.rcs_theta_db, simulation_result.rcs_phi_db)
    )


def _build_rcs_surface_xyz(
    simulation_result: RcsComputationResult,
    component: str,
    radial_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rcs_db = _select_rcs_component(simulation_result, component)
    phi_rad = np.deg2rad(simulation_result.phi_grid_deg)
    theta_rad = np.deg2rad(simulation_result.theta_grid_deg)

    mesh_vertices = simulation_result.geometry.vertex_coordinates
    mesh_center = mesh_vertices.mean(axis=0)
    mesh_radius = np.linalg.norm(mesh_vertices - mesh_center, axis=1).max()
    if mesh_radius <= 0:
        mesh_radius = 1.0

    rcs_linear = np.power(10.0, rcs_db / 10.0)
    linear_span = float(np.max(rcs_linear) - np.min(rcs_linear))
    if linear_span <= 1e-12:
        normalized_rcs = np.zeros_like(rcs_linear)
    else:
        normalized_rcs = (rcs_linear - np.min(rcs_linear)) / linear_span

    radial_distance = mesh_radius * (1.05 + radial_scale * normalized_rcs)
    surface_x = mesh_center[0] + radial_distance * np.sin(theta_rad) * np.cos(phi_rad)
    surface_y = mesh_center[1] + radial_distance * np.sin(theta_rad) * np.sin(phi_rad)
    surface_z = mesh_center[2] + radial_distance * np.cos(theta_rad)
    return surface_x, surface_y, surface_z, rcs_db


def _build_2d_cut(
    simulation_result: RcsComputationResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return angle-axis and theta/phi RCS cuts for 2D plotting."""
    phi_count, theta_count = simulation_result.phi_grid_deg.shape
    if phi_count == 1:
        angle_values = simulation_result.theta_grid_deg[0]
        return angle_values, simulation_result.rcs_theta_db[0], simulation_result.rcs_phi_db[0]
    if theta_count == 1:
        angle_values = simulation_result.phi_grid_deg[:, 0]
        return (
            angle_values,
            simulation_result.rcs_theta_db[:, 0],
            simulation_result.rcs_phi_db[:, 0],
        )

    mid_phi_index = phi_count // 2
    angle_values = simulation_result.theta_grid_deg[mid_phi_index]
    return (
        angle_values,
        simulation_result.rcs_theta_db[mid_phi_index],
        simulation_result.rcs_phi_db[mid_phi_index],
    )


def _expand_full_polar_if_needed(
    angle_values_deg: np.ndarray,
    theta_values_db: np.ndarray,
    phi_values_db: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror 0..180 cuts to a full 0..360 polar display."""
    if angle_values_deg.size < 2:
        return angle_values_deg, theta_values_db, phi_values_db
    if float(np.min(angle_values_deg)) < 0.0 or float(np.max(angle_values_deg)) > 180.0:
        return angle_values_deg, theta_values_db, phi_values_db

    mirrored_angles = 360.0 - angle_values_deg[-2::-1]
    mirrored_theta = theta_values_db[-2::-1]
    mirrored_phi = phi_values_db[-2::-1]
    return (
        np.concatenate([angle_values_deg, mirrored_angles]),
        np.concatenate([theta_values_db, mirrored_theta]),
        np.concatenate([phi_values_db, mirrored_phi]),
    )


def _sample_count(start_deg: float, stop_deg: float, step_deg: float) -> int:
    """Count inclusive samples produced by start/stop/step."""
    span_deg = max(0.0, float(stop_deg) - float(start_deg))
    step = max(1e-12, float(step_deg))
    return int(np.floor(span_deg / step) + 1)


def _sphere_surface(
    center_xyz: np.ndarray,
    radius: float,
    theta_start_deg: float,
    theta_stop_deg: float,
    phi_start_deg: float,
    phi_stop_deg: float,
    theta_resolution: int = 36,
    phi_resolution: int = 72,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build XYZ surface arrays for a spherical angular span."""
    if theta_stop_deg < theta_start_deg:
        theta_start_deg, theta_stop_deg = theta_stop_deg, theta_start_deg
    if phi_stop_deg < phi_start_deg:
        phi_start_deg, phi_stop_deg = phi_stop_deg, phi_start_deg

    theta_values = np.linspace(theta_start_deg, theta_stop_deg, max(2, theta_resolution))
    phi_values = np.linspace(phi_start_deg, phi_stop_deg, max(2, phi_resolution))
    phi_grid, theta_grid = np.meshgrid(np.deg2rad(phi_values), np.deg2rad(theta_values))

    x_surface = center_xyz[0] + radius * np.sin(theta_grid) * np.cos(phi_grid)
    y_surface = center_xyz[1] + radius * np.sin(theta_grid) * np.sin(phi_grid)
    z_surface = center_xyz[2] + radius * np.cos(theta_grid)
    return x_surface, y_surface, z_surface


def build_plotly_figures(
    simulation_result: RcsComputationResult,
    stl_model_path: str | Path,
    *,
    chart_mode: str = "xy",
    component: str = "max",
    radial_scale: float = 1.0,
    rcs_opacity: float = 0.55,
    mesh_vertices: np.ndarray | None = None,
    mesh_faces: np.ndarray | None = None,
):
    """Build in-memory Plotly figures for 2D RCS and 3D STL+RCS overlay."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover
        raise ImportError("build_plotly_figures requires plotly.") from exc

    angle_values, rcs_theta_cut, rcs_phi_cut = _build_2d_cut(simulation_result)
    fig_2d = go.Figure()
    if chart_mode == "polar":
        angle_values, rcs_theta_cut, rcs_phi_cut = _expand_full_polar_if_needed(
            angle_values,
            rcs_theta_cut,
            rcs_phi_cut,
        )
        fig_2d.add_trace(
            go.Scatterpolar(theta=angle_values, r=rcs_theta_cut, mode="lines", name="RCS Theta")
        )
        fig_2d.add_trace(
            go.Scatterpolar(theta=angle_values, r=rcs_phi_cut, mode="lines", name="RCS Phi")
        )
        fig_2d.update_layout(
            title="RCS Cut (Polar)",
            polar=dict(angularaxis=dict(direction="counterclockwise", rotation=90)),
        )
    else:
        fig_2d.add_trace(
            go.Scatter(x=angle_values, y=rcs_theta_cut, mode="lines", name="RCS Theta")
        )
        fig_2d.add_trace(go.Scatter(x=angle_values, y=rcs_phi_cut, mode="lines", name="RCS Phi"))
        fig_2d.update_layout(
            title="RCS Cut (X-Y)", xaxis_title="Angle (deg)", yaxis_title="RCS (dBsm)"
        )

    if mesh_vertices is None or mesh_faces is None:
        mesh_vertices, mesh_faces = _load_mesh_vertices_faces(Path(stl_model_path))
    surface_x, surface_y, surface_z, rcs_color = _build_rcs_surface_xyz(
        simulation_result,
        component,
        radial_scale,
    )
    fig_3d = go.Figure()
    fig_3d.add_trace(
        go.Mesh3d(
            x=mesh_vertices[:, 0],
            y=mesh_vertices[:, 1],
            z=mesh_vertices[:, 2],
            i=mesh_faces[:, 0],
            j=mesh_faces[:, 1],
            k=mesh_faces[:, 2],
            color="lightgray",
            opacity=0.5,
            name="STL Model",
            showscale=False,
        )
    )
    if min(surface_x.shape) > 1:
        fig_3d.add_trace(
            go.Surface(
                x=surface_x,
                y=surface_y,
                z=surface_z,
                surfacecolor=rcs_color,
                colorscale="Viridis",
                opacity=rcs_opacity,
                colorbar=dict(title="RCS (dBsm)"),
                name="RCS Surface",
            )
        )
    else:
        fig_3d.add_trace(
            go.Scatter3d(
                x=surface_x.flatten(),
                y=surface_y.flatten(),
                z=surface_z.flatten(),
                mode="lines",
                line=dict(color="royalblue", width=5),
                name="RCS Curve",
            )
        )
    fig_3d.update_layout(title="Interactive STL + RCS Overlay", scene=dict(aspectmode="data"))
    return fig_2d, fig_3d


def launch_rcs_widget(project_root: str | Path = "."):
    """Launch an interactive RCS widget for Jupyter notebooks."""
    try:
        import ipywidgets as widgets
        import plotly.graph_objects as go
        from IPython.display import display
    except ImportError as exc:  # pragma: no cover - notebook runtime guard
        raise ImportError(
            "Notebook UI requires 'ipywidgets' and 'plotly'. "
            "Install them with: pip install ipywidgets plotly"
        ) from exc

    project_path = Path(project_root).resolve()
    material_extension = ".rcsmat"
    model_dir = project_path / "stl_models"
    stl_models = sorted(p.name for p in model_dir.glob("*.stl"))
    if not stl_models:
        raise FileNotFoundError(f"No STL files found in: {model_dir}")

    material_candidates: list[Path] = []
    for base_dir in (project_path, project_path / "materials"):
        if base_dir.exists():
            material_candidates.extend(sorted(base_dir.glob(f"*{material_extension}")))
    legacy_material = project_path / "matrl.txt"
    if legacy_material.exists():
        material_candidates.append(legacy_material)

    unique_material_files: list[Path] = []
    seen_material_paths: set[str] = set()
    for material_file in material_candidates:
        resolved = str(material_file.resolve())
        if resolved in seen_material_paths:
            continue
        seen_material_paths.add(resolved)
        unique_material_files.append(material_file.resolve())

    material_options: list[tuple[str, str]] = []
    for material_file in unique_material_files:
        if material_file.is_relative_to(project_path):
            label = str(material_file.relative_to(project_path))
        else:
            label = str(material_file)
        material_options.append((label, str(material_file)))

    mode_widget = widgets.ToggleButtons(
        options=[("Monostatic", "monostatic"), ("Bistatic", "bistatic")],
        value="monostatic",
        description="Mode:",
        tooltip="Select monostatic or bistatic solver mode.",
    )
    model_widget = widgets.Dropdown(
        options=stl_models,
        value="sphere.stl" if "sphere.stl" in stl_models else stl_models[0],
        description="Model:",
        tooltip="STL model used for geometry and visualization.",
    )
    radar_bands = "\n".join(["", "Radar Bands:", *RadarBand.to_string_list()])
    freq_widget = widgets.FloatText(
        value=10.0,
        description="Freq GHz:",
        tooltip=f"Radar frequency in GHz.\n{radar_bands}",
    )
    corr_widget = widgets.FloatText(
        value=0.0,
        description="Surf. Corr. Dist.:",
        tooltip="Surface correlation distance (meters).",
    )
    std_widget = widgets.FloatText(
        value=0.0,
        description="Surf. Rough.:",
        tooltip="Surface roughness standard deviation (meters).",
    )
    pol_widget = widgets.Dropdown(
        options=[("TM-z", 0), ("TE-z", 1)],
        value=0,
        description="Polarization:",
        tooltip="Incident wave polarization.\nTM-z: Theta Polarization\nTE-z: Phi Polarization",
    )
    suggest_surface_params_button = widgets.Button(
        description="Suggest Surface Params",
        tooltip=("Estimate roughness and correlation distance from geometry scale and wavelength."),
        button_style="info",
    )
    surface_guidance_widget = widgets.HTML(
        value=(
            "<span style='color:#555;'>Use <b>Suggest Surface Params</b> for analyst "
            "starting values.</span>"
        )
    )
    use_material_file_widget = widgets.Checkbox(
        value=bool(material_options),
        description="Use material file",
        tooltip="Apply per-facet material properties from a material file.",
        disabled=not bool(material_options),
    )
    material_file_widget = widgets.Dropdown(
        options=material_options if material_options else [("(No material file found)", "")],
        value=material_options[0][1] if material_options else "",
        description="Material:",
        tooltip="Select a material file. Preferred extension: .rcsmat",
        disabled=not bool(material_options),
    )
    material_summary_widget = widgets.HTML(
        value="<span style='color:#555;'>Select a material file to view catalog details.</span>"
    )

    phi_range_widget = widgets.FloatRangeSlider(
        value=(0.0, 360.0),
        min=0.0,
        max=360.0,
        step=1.0,
        description="Phi range:",
        readout_format=".0f",
        tooltip="Azimuth sweep range in degrees.",
        layout=widgets.Layout(width="430px"),
    )
    theta_range_widget = widgets.FloatRangeSlider(
        value=(0.0, 180.0),
        min=0.0,
        max=180.0,
        step=1.0,
        description="Theta range:",
        readout_format=".0f",
        tooltip="Elevation sweep range in degrees.",
        layout=widgets.Layout(width="430px"),
    )
    phi_step_widget = widgets.IntSlider(
        value=5,
        min=1,
        max=5,
        step=1,
        description="Phi step:",
        tooltip="Azimuth step size (degrees).",
        layout=widgets.Layout(width="260px"),
    )
    theta_step_widget = widgets.IntSlider(
        value=5,
        min=1,
        max=5,
        step=1,
        description="Theta step:",
        tooltip="Elevation step size (degrees).",
        layout=widgets.Layout(width="260px"),
    )
    inc_theta_widget = widgets.FloatSlider(
        value=30.0,
        min=0.0,
        max=180.0,
        step=1.0,
        description="Inc theta:",
        readout_format=".0f",
        tooltip="Incident elevation angle for bistatic mode (degrees).",
        layout=widgets.Layout(width="320px"),
    )
    inc_phi_widget = widgets.FloatSlider(
        value=0.0,
        min=0.0,
        max=360.0,
        step=1.0,
        description="Inc phi:",
        readout_format=".0f",
        tooltip="Incident azimuth angle for bistatic mode (degrees).",
        layout=widgets.Layout(width="320px"),
    )
    incident_angles_row = widgets.HBox(
        [inc_theta_widget, inc_phi_widget],
        layout=widgets.Layout(display="none"),
    )

    roll_widget = widgets.FloatSlider(
        value=0.0,
        min=-180.0,
        max=180.0,
        step=1.0,
        description="Roll X:",
        tooltip="Rotate STL around X axis before simulation (degrees).",
    )
    pitch_widget = widgets.FloatSlider(
        value=0.0,
        min=-180.0,
        max=180.0,
        step=1.0,
        description="Pitch Y:",
        tooltip="Rotate STL around Y axis before simulation (degrees).",
    )
    yaw_widget = widgets.FloatSlider(
        value=0.0,
        min=-180.0,
        max=180.0,
        step=1.0,
        description="Yaw Z:",
        tooltip="Rotate STL around Z axis before simulation (degrees).",
    )

    chart_mode_widget = widgets.ToggleButtons(
        options=[("Polar", "polar"), ("X-Y", "xy")],
        value="polar",
        description="2D:",
        tooltip="Select Cartesian or polar RCS cut plot.",
    )
    component_widget = widgets.Dropdown(
        options=[("Max(theta,phi)", "max"), ("Theta", "theta"), ("Phi", "phi")],
        value="max",
        description="3D comp:",
        tooltip="RCS component used to color/shape the 3D envelope.",
    )
    scale_widget = widgets.FloatLogSlider(
        value=1.0,
        base=10,
        min=-2,
        max=2,
        step=0.05,
        description="3D scale:",
        readout_format=".2f",
        tooltip="Scale factor for the RCS envelope radius.",
    )
    opacity_widget = widgets.FloatSlider(
        value=0.55,
        min=0.1,
        max=1.0,
        step=0.05,
        description="3D alpha:",
        tooltip="Opacity of the RCS surface overlay.",
    )
    run_button = widgets.Button(
        description="Run Simulation",
        button_style="primary",
        tooltip="Run solver with current inputs and model orientation.",
    )
    sweep_info_widget = widgets.HTML()
    progress_bar = widgets.FloatProgress(
        value=0.0,
        min=0.0,
        max=1.0,
        description="Progress:",
        bar_style="",
        layout=widgets.Layout(width="520px"),
    )
    progress_text_widget = widgets.HTML("<span>Idle</span>")

    preview_output = widgets.Output()
    results_output = widgets.Output()
    state: dict[str, Any] = {
        "result": None,
        "mesh_vertices": None,
        "mesh_faces": None,
        "raw_vertices": None,
        "raw_faces": None,
        "loaded_model": None,
    }

    def _ensure_raw_mesh_loaded() -> None:
        if state["loaded_model"] == model_widget.value:
            return
        raw_vertices, raw_faces = _load_mesh_vertices_faces(model_dir / model_widget.value)
        state["raw_vertices"] = raw_vertices
        state["raw_faces"] = raw_faces
        state["loaded_model"] = model_widget.value

    def _current_rotation() -> tuple[float, float, float]:
        return float(roll_widget.value), float(pitch_widget.value), float(yaw_widget.value)

    def _current_sweep() -> tuple[float, float, float, float, int, int]:
        phi_start_deg, phi_stop_deg = [float(value) for value in phi_range_widget.value]
        theta_start_deg, theta_stop_deg = [float(value) for value in theta_range_widget.value]
        return (
            phi_start_deg,
            phi_stop_deg,
            theta_start_deg,
            theta_stop_deg,
            int(phi_step_widget.value),
            int(theta_step_widget.value),
        )

    def _incident_direction_unit() -> np.ndarray:
        incident_theta_rad = np.deg2rad(float(inc_theta_widget.value))
        incident_phi_rad = np.deg2rad(float(inc_phi_widget.value))
        incident_direction = np.array(
            [
                np.sin(incident_theta_rad) * np.cos(incident_phi_rad),
                np.sin(incident_theta_rad) * np.sin(incident_phi_rad),
                np.cos(incident_theta_rad),
            ],
            dtype=float,
        )
        norm = float(np.linalg.norm(incident_direction))
        if norm <= 1e-12:
            return np.array([0.0, 0.0, 1.0], dtype=float)
        return incident_direction / norm

    def _add_incident_wave_arrow(
        fig: Any, mesh_center: np.ndarray, reference_radius: float
    ) -> None:
        if mode_widget.value != "bistatic":
            return
        incident_direction = _incident_direction_unit()
        line_tail = mesh_center + incident_direction * (reference_radius * 1.95)
        cone_tail = mesh_center + incident_direction * (reference_radius * 1.35)
        cone_length = max(reference_radius * 0.24, 0.08)
        cone_vector = -incident_direction * cone_length
        fig.add_trace(
            go.Scatter3d(
                x=[line_tail[0], cone_tail[0]],
                y=[line_tail[1], cone_tail[1]],
                z=[line_tail[2], cone_tail[2]],
                mode="lines",
                line=dict(color="#e63946", width=8),
                name="Incident Wave",
                hovertemplate=(
                    "Incident wave direction<br>"
                    f"theta={float(inc_theta_widget.value):.0f} deg, "
                    f"phi={float(inc_phi_widget.value):.0f} deg<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Cone(
                x=[cone_tail[0]],
                y=[cone_tail[1]],
                z=[cone_tail[2]],
                u=[cone_vector[0]],
                v=[cone_vector[1]],
                w=[cone_vector[2]],
                anchor="tail",
                sizemode="absolute",
                sizeref=max(cone_length * 1.05, 0.05),
                colorscale=[[0.0, "#e63946"], [1.0, "#e63946"]],
                showscale=False,
                showlegend=False,
                name="Incident Wave Tip",
                hovertemplate=(
                    "Wave incoming toward model<br>"
                    f"theta={float(inc_theta_widget.value):.0f} deg, "
                    f"phi={float(inc_phi_widget.value):.0f} deg<extra></extra>"
                ),
            )
        )

    def _update_sweep_info(*_) -> None:
        (
            phi_start_deg,
            phi_stop_deg,
            theta_start_deg,
            theta_stop_deg,
            phi_step_deg,
            theta_step_deg,
        ) = _current_sweep()
        phi_samples = _sample_count(phi_start_deg, phi_stop_deg, phi_step_deg)
        theta_samples = _sample_count(theta_start_deg, theta_stop_deg, theta_step_deg)
        total_samples = phi_samples * theta_samples
        triangle_count = 0
        if state["raw_faces"] is not None:
            triangle_count = int(np.asarray(state["raw_faces"]).shape[0])
        interaction_count = total_samples * triangle_count if triangle_count else 0
        sweep_info_widget.value = (
            "<b>Sweep Info:</b> "
            f"phi samples={phi_samples}, theta samples={theta_samples}, "
            f"total angles={total_samples}, "
            f"triangle-angle evaluations~={interaction_count:,}"
        )
        progress_bar.max = float(max(1, total_samples))
        if progress_bar.value > progress_bar.max:
            progress_bar.value = progress_bar.max

    def _render_preview(*_) -> None:
        _ensure_raw_mesh_loaded()
        raw_vertices = np.asarray(state["raw_vertices"])
        raw_faces = np.asarray(state["raw_faces"])
        roll_deg, pitch_deg, yaw_deg = _current_rotation()
        rotated_vertices = _rotate_vertices(raw_vertices, roll_deg, pitch_deg, yaw_deg)
        state["mesh_vertices"] = rotated_vertices
        state["mesh_faces"] = raw_faces

        with preview_output:
            preview_output.clear_output(wait=True)
            fig_preview = go.Figure()
            fig_preview.add_trace(
                go.Mesh3d(
                    x=rotated_vertices[:, 0],
                    y=rotated_vertices[:, 1],
                    z=rotated_vertices[:, 2],
                    i=raw_faces[:, 0],
                    j=raw_faces[:, 1],
                    k=raw_faces[:, 2],
                    color="lightsteelblue",
                    opacity=0.75,
                    name="STL Preview",
                    showscale=False,
                )
            )
            mesh_center = rotated_vertices.mean(axis=0)
            mesh_radius = float(np.linalg.norm(rotated_vertices - mesh_center, axis=1).max())
            if mesh_radius <= 0:
                mesh_radius = 1.0
            envelope_radius = mesh_radius * 1.2
            (
                phi_start_deg,
                phi_stop_deg,
                theta_start_deg,
                theta_stop_deg,
                phi_step_deg,
                theta_step_deg,
            ) = _current_sweep()
            phi_samples = _sample_count(phi_start_deg, phi_stop_deg, phi_step_deg)
            theta_samples = _sample_count(theta_start_deg, theta_stop_deg, theta_step_deg)

            full_sphere_x, full_sphere_y, full_sphere_z = _sphere_surface(
                mesh_center,
                envelope_radius,
                0.0,
                180.0,
                0.0,
                360.0,
                theta_resolution=28,
                phi_resolution=56,
            )
            sweep_x, sweep_y, sweep_z = _sphere_surface(
                mesh_center,
                envelope_radius * 1.01,
                theta_start_deg,
                theta_stop_deg,
                phi_start_deg,
                phi_stop_deg,
                theta_resolution=max(4, min(64, theta_samples + 1)),
                phi_resolution=max(4, min(128, phi_samples + 1)),
            )
            fig_preview.add_trace(
                go.Surface(
                    x=full_sphere_x,
                    y=full_sphere_y,
                    z=full_sphere_z,
                    surfacecolor=np.zeros_like(full_sphere_x),
                    colorscale=[[0.0, "#6fa8dc"], [1.0, "#6fa8dc"]],
                    opacity=0.08,
                    showscale=False,
                    name="Full Angular Envelope",
                    hoverinfo="skip",
                )
            )
            fig_preview.add_trace(
                go.Surface(
                    x=sweep_x,
                    y=sweep_y,
                    z=sweep_z,
                    surfacecolor=np.ones_like(sweep_x),
                    colorscale=[[0.0, "#f4a261"], [1.0, "#f4a261"]],
                    opacity=0.28,
                    showscale=False,
                    name="Selected Sweep Span",
                    hovertemplate="Sweep surface<extra></extra>",
                )
            )
            is_bistatic = mode_widget.value == "bistatic"
            _add_incident_wave_arrow(fig_preview, mesh_center, envelope_radius)
            fig_preview.update_layout(
                title=(
                    "3D Model Preview (drag to rotate) "
                    f"| phi {phi_start_deg:.0f}-{phi_stop_deg:.0f} deg, "
                    f"theta {theta_start_deg:.0f}-{theta_stop_deg:.0f} deg"
                    + (
                        f" | Inc: theta={float(inc_theta_widget.value):.0f} deg, "
                        f"phi={float(inc_phi_widget.value):.0f} deg"
                        if is_bistatic
                        else ""
                    )
                ),
                scene=dict(aspectmode="data"),
                height=520,
                margin=dict(l=0, r=0, t=40, b=0),
            )
            fig_preview.show()
        _update_sweep_info()

    def _set_mode_controls(*_) -> None:
        is_bistatic = mode_widget.value == "bistatic"
        inc_theta_widget.disabled = not is_bistatic
        inc_phi_widget.disabled = not is_bistatic
        incident_angles_row.layout.display = "flex" if is_bistatic else "none"

    def _set_material_controls(*_) -> None:
        material_file_widget.disabled = (not bool(use_material_file_widget.value)) or (
            not bool(material_options)
        )

    def _update_material_summary(*_) -> None:
        if not bool(use_material_file_widget.value):
            material_summary_widget.value = (
                "<span style='color:#555;'>Material file usage is disabled.</span>"
            )
            return
        material_path = str(material_file_widget.value)
        if not material_path:
            material_summary_widget.value = (
                "<span style='color:#b22222;'>No material file selected.</span>"
            )
            return
        path = Path(material_path)
        if path.suffix.lower() not in {".rcsmat", ".yaml", ".yml"}:
            material_summary_widget.value = (
                "<span style='color:#555;'>Legacy per-facet text material file selected.</span>"
            )
            return
        try:
            material_catalog = rf.load_material_catalog(path)
            material_ids = ", ".join(sorted(material_catalog.keys()))
            material_summary_widget.value = (
                f"<span><b>Material catalog:</b> {len(material_catalog)} entries "
                f"({material_ids})</span>"
            )
        except Exception as exc:
            material_summary_widget.value = (
                f"<span style='color:#b22222;'>Invalid material library: {exc}</span>"
            )

    def _suggest_surface_parameters(*_) -> None:
        try:
            frequency_hz = float(freq_widget.value) * 1e9
            guidance_geometry = rf.build_geometry_from_stl(
                model_dir / model_widget.value,
                0.0,
            )
            guidance = estimate_roughness_correlation_guidance(frequency_hz, guidance_geometry)
            corr_widget.value = float(guidance.suggested_correlation_distance_m)
            std_widget.value = float(guidance.suggested_standard_deviation_m)
            notes_html = "".join(f"<li>{note}</li>" for note in guidance.notes)
            surface_guidance_widget.value = (
                "<div style='font-size:12px;'>"
                f"<b>Suggested sigma:</b> {guidance.suggested_standard_deviation_m:.3e} m "
                f"(range {guidance.standard_deviation_bounds_m[0]:.3e} - "
                f"{guidance.standard_deviation_bounds_m[1]:.3e})<br>"
                f"<b>Suggested correlation:</b> {guidance.suggested_correlation_distance_m:.3e} m "
                f"(range {guidance.correlation_distance_bounds_m[0]:.3e} - "
                f"{guidance.correlation_distance_bounds_m[1]:.3e})<br>"
                f"<b>Wavelength:</b> {guidance.wavelength_m:.3e} m, "
                f"<b>L/lambda:</b> {guidance.electrical_size_l_over_lambda:.2f}, "
                f"<b>Median edge:</b> {guidance.median_edge_length_m:.3e} m"
                f"<ul>{notes_html}</ul></div>"
            )
        except Exception as exc:
            surface_guidance_widget.value = (
                f"<span style='color:#b22222;'>Unable to estimate parameters: {exc}</span>"
            )

    def _run_simulation(*_) -> None:
        run_button.disabled = True
        progress_bar.bar_style = "info"
        progress_bar.value = 0.0
        progress_text_widget.value = "<span>Preparing simulation...</span>"
        _update_sweep_info()
        with results_output:
            results_output.clear_output(wait=True)

        try:
            (
                phi_start_deg,
                phi_stop_deg,
                theta_start_deg,
                theta_stop_deg,
                phi_step_deg,
                theta_step_deg,
            ) = _current_sweep()
            phi_samples = _sample_count(phi_start_deg, phi_stop_deg, phi_step_deg)
            theta_samples = _sample_count(theta_start_deg, theta_stop_deg, theta_step_deg)
            total_samples = max(1, phi_samples * theta_samples)
            progress_bar.max = float(total_samples)
            callback_stride = max(1, total_samples // 250)

            angle_sweep = AngleSweep(
                phi_start_deg=phi_start_deg,
                phi_stop_deg=phi_stop_deg,
                phi_step_deg=float(phi_step_deg),
                theta_start_deg=theta_start_deg,
                theta_stop_deg=theta_stop_deg,
                theta_step_deg=float(theta_step_deg),
            )
            use_material_file = bool(use_material_file_widget.value) and bool(material_options)
            material = MaterialConfig(
                resistivity_mode=float(rf.MATERIAL_SPECIFIC if use_material_file else 0),
                material_path=str(
                    material_file_widget.value
                    if use_material_file
                    else (project_path / "matrl.txt")
                ),
            )

            progress_text_widget.value = "<span>Building geometry...</span>"
            base_geometry = rf.build_geometry_from_stl(
                model_dir / model_widget.value,
                material.resistivity_mode,
            )
            roll_deg, pitch_deg, yaw_deg = _current_rotation()
            geometry_data = _build_rotated_geometry(base_geometry, roll_deg, pitch_deg, yaw_deg)

            start_time = perf_counter()

            def _progress_update(completed: int, total: int) -> None:
                progress_bar.value = float(completed)
                elapsed_seconds = max(0.0, perf_counter() - start_time)
                if completed > 0:
                    seconds_per_step = elapsed_seconds / float(completed)
                    remaining_seconds = max(0.0, (total - completed) * seconds_per_step)
                else:
                    remaining_seconds = 0.0
                progress_text_widget.value = (
                    f"<span>Computing: {completed}/{total} angles | "
                    f"elapsed {elapsed_seconds:.1f}s | ETA {remaining_seconds:.1f}s</span>"
                )

            _progress_update(0, total_samples)
            if mode_widget.value == "monostatic":
                monostatic_config = MonostaticSimulationConfig(
                    input_model=model_widget.value,
                    frequency_hz=float(freq_widget.value) * 1e9,
                    correlation_distance_m=float(corr_widget.value),
                    standard_deviation_m=float(std_widget.value),
                    incident_polarization=int(pol_widget.value),
                    angle_sweep=angle_sweep,
                    material=material,
                )
                simulation_result = simulate_monostatic(
                    monostatic_config,
                    geometry_data,
                    progress_callback=_progress_update,
                    progress_update_stride=callback_stride,
                )
            else:
                bistatic_config = BistaticSimulationConfig(
                    input_model=model_widget.value,
                    frequency_hz=float(freq_widget.value) * 1e9,
                    correlation_distance_m=float(corr_widget.value),
                    standard_deviation_m=float(std_widget.value),
                    incident_polarization=int(pol_widget.value),
                    angle_sweep=angle_sweep,
                    incident_theta_deg=float(inc_theta_widget.value),
                    incident_phi_deg=float(inc_phi_widget.value),
                    material=material,
                )
                simulation_result = simulate_bistatic(
                    bistatic_config,
                    geometry_data,
                    progress_callback=_progress_update,
                    progress_update_stride=callback_stride,
                )

            total_elapsed_seconds = perf_counter() - start_time
            progress_bar.value = float(total_samples)
            progress_bar.bar_style = "success"
            progress_text_widget.value = (
                f"<span>Completed in {total_elapsed_seconds:.1f}s "
                f"for {total_samples} angle samples.</span>"
            )

            if state["mesh_vertices"] is None or state["mesh_faces"] is None:
                _render_preview()

            state["result"] = simulation_result
            _render_results()
        except Exception:
            progress_bar.bar_style = "danger"
            progress_text_widget.value = "<span>Simulation failed. Check input settings.</span>"
            raise
        finally:
            run_button.disabled = False

    def _render_results(*_) -> None:
        simulation_result = state.get("result")
        mesh_vertices = state.get("mesh_vertices")
        mesh_faces = state.get("mesh_faces")
        if simulation_result is None or mesh_vertices is None or mesh_faces is None:
            return

        with results_output:
            results_output.clear_output(wait=True)
            angle_values, rcs_theta_cut, rcs_phi_cut = _build_2d_cut(simulation_result)

            fig_2d = go.Figure()
            if chart_mode_widget.value == "polar":
                angle_values, rcs_theta_cut, rcs_phi_cut = _expand_full_polar_if_needed(
                    angle_values,
                    rcs_theta_cut,
                    rcs_phi_cut,
                )
                fig_2d.add_trace(
                    go.Scatterpolar(
                        theta=angle_values, r=rcs_theta_cut, mode="lines", name="RCS Theta"
                    )
                )
                fig_2d.add_trace(
                    go.Scatterpolar(theta=angle_values, r=rcs_phi_cut, mode="lines", name="RCS Phi")
                )
                fig_2d.update_layout(
                    title="RCS Cut (Polar)",
                    polar=dict(angularaxis=dict(direction="counterclockwise", rotation=90)),
                )
            else:
                fig_2d.add_trace(
                    go.Scatter(x=angle_values, y=rcs_theta_cut, mode="lines", name="RCS Theta")
                )
                fig_2d.add_trace(
                    go.Scatter(x=angle_values, y=rcs_phi_cut, mode="lines", name="RCS Phi")
                )
                fig_2d.update_layout(
                    title="RCS Cut (X-Y)",
                    xaxis_title="Angle (deg)",
                    yaxis_title="RCS (dBsm)",
                )
            fig_2d.update_layout(height=620, autosize=True)

            surface_x, surface_y, surface_z, rcs_color = _build_rcs_surface_xyz(
                simulation_result,
                component_widget.value,
                float(scale_widget.value),
            )
            mesh_vertices_np = np.asarray(mesh_vertices)
            mesh_faces_np = np.asarray(mesh_faces)
            fig_3d = go.Figure()
            fig_3d.add_trace(
                go.Mesh3d(
                    x=mesh_vertices_np[:, 0],
                    y=mesh_vertices_np[:, 1],
                    z=mesh_vertices_np[:, 2],
                    i=mesh_faces_np[:, 0],
                    j=mesh_faces_np[:, 1],
                    k=mesh_faces_np[:, 2],
                    color="lightgray",
                    opacity=0.5,
                    name="STL Model",
                    showscale=False,
                )
            )
            if min(surface_x.shape) > 1:
                fig_3d.add_trace(
                    go.Surface(
                        x=surface_x,
                        y=surface_y,
                        z=surface_z,
                        surfacecolor=rcs_color,
                        colorscale="Viridis",
                        opacity=float(opacity_widget.value),
                        colorbar=dict(title="RCS (dBsm)"),
                        name="RCS Surface",
                    )
                )
            else:
                fig_3d.add_trace(
                    go.Scatter3d(
                        x=surface_x.flatten(),
                        y=surface_y.flatten(),
                        z=surface_z.flatten(),
                        mode="lines",
                        line=dict(color="royalblue", width=5),
                        name="RCS Curve",
                    )
                )
            mesh_center = mesh_vertices_np.mean(axis=0)
            mesh_radius = float(np.linalg.norm(mesh_vertices_np - mesh_center, axis=1).max())
            if mesh_radius <= 0:
                mesh_radius = 1.0
            _add_incident_wave_arrow(fig_3d, mesh_center, mesh_radius * 1.2)
            fig_3d.update_layout(
                title=(
                    "Interactive STL + RCS Overlay"
                    + (
                        f" | Inc: theta={float(inc_theta_widget.value):.0f} deg, "
                        f"phi={float(inc_phi_widget.value):.0f} deg"
                        if mode_widget.value == "bistatic"
                        else ""
                    )
                ),
                scene=dict(aspectmode="data"),
                legend=dict(orientation="h"),
                height=620,
                autosize=True,
            )

            selected_component_db = _select_rcs_component(simulation_result, component_widget.value)
            dynamic_range_db = float(np.max(selected_component_db) - np.min(selected_component_db))
            if dynamic_range_db < 1e-6:
                display(
                    widgets.HTML(
                        "<b>Note:</b> Selected 3D component is nearly constant. "
                        "Try switching polarization or use the 'Max(theta,phi)' component."
                    )
                )

            plot_2d_output = widgets.Output(layout=widgets.Layout(width="50%", min_width="520px"))
            plot_3d_output = widgets.Output(layout=widgets.Layout(width="50%", min_width="520px"))
            with plot_2d_output:
                display(fig_2d)
            with plot_3d_output:
                display(fig_3d)
            display(
                widgets.HBox(
                    [plot_2d_output, plot_3d_output],
                    layout=widgets.Layout(
                        width="100%",
                        justify_content="space-between",
                        align_items="stretch",
                        gap="12px",
                    ),
                )
            )

    run_button.on_click(_run_simulation)
    suggest_surface_params_button.on_click(_suggest_surface_parameters)
    mode_widget.observe(_set_mode_controls, names="value")
    mode_widget.observe(_render_preview, names="value")
    use_material_file_widget.observe(_set_material_controls, names="value")
    use_material_file_widget.observe(_update_material_summary, names="value")
    material_file_widget.observe(_update_material_summary, names="value")
    model_widget.observe(_render_preview, names="value")
    phi_range_widget.observe(_render_preview, names="value")
    theta_range_widget.observe(_render_preview, names="value")
    phi_step_widget.observe(_update_sweep_info, names="value")
    theta_step_widget.observe(_update_sweep_info, names="value")
    roll_widget.observe(_render_preview, names="value")
    pitch_widget.observe(_render_preview, names="value")
    yaw_widget.observe(_render_preview, names="value")
    inc_theta_widget.observe(_render_preview, names="value")
    inc_phi_widget.observe(_render_preview, names="value")

    chart_mode_widget.observe(_render_results, names="value")
    component_widget.observe(_render_results, names="value")
    scale_widget.observe(_render_results, names="value")
    opacity_widget.observe(_render_results, names="value")

    _set_mode_controls()
    _set_material_controls()
    _update_material_summary()
    _render_preview()

    controls = widgets.VBox(
        [
            widgets.HBox([mode_widget, model_widget, run_button]),
            widgets.HBox(
                [freq_widget, corr_widget, std_widget, pol_widget, suggest_surface_params_button]
            ),
            surface_guidance_widget,
            widgets.HBox([use_material_file_widget, material_file_widget]),
            material_summary_widget,
            widgets.HBox([phi_range_widget, phi_step_widget]),
            widgets.HBox([theta_range_widget, theta_step_widget]),
            incident_angles_row,
            widgets.HBox([roll_widget, pitch_widget, yaw_widget]),
            widgets.HBox([chart_mode_widget, component_widget, scale_widget, opacity_widget]),
            sweep_info_widget,
            widgets.HBox([progress_bar, progress_text_widget]),
        ],
        layout=widgets.Layout(width="58%"),
    )
    preview_panel = widgets.VBox(
        [
            widgets.HTML("<h4>Model Preview</h4>"),
            preview_output,
        ],
        layout=widgets.Layout(width="42%"),
    )
    top_row = widgets.HBox(
        [controls, preview_panel],
        layout=widgets.Layout(width="100%", align_items="flex-start", gap="12px"),
    )
    root = widgets.VBox(
        [
            top_row,
            widgets.HTML("<h4>Simulation Results</h4>"),
            results_output,
        ]
    )
    display(root)
    return root
