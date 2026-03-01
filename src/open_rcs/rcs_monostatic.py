"""Monostatic RCS solver."""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable

import numpy as np

from . import rcs_functions as rf
from .model_types import (
    FieldAccumulator,
    GeometryData,
    MonostaticSimulationConfig,
    RcsComputationResult,
    SolverResult,
)


def simulate_monostatic(
    simulation_config: MonostaticSimulationConfig,
    geometry_data: GeometryData,
    progress_callback: Callable[[int, int], None] | None = None,
    progress_update_stride: int = 1,
) -> RcsComputationResult:
    """Run monostatic simulation and return in-memory arrays (no disk output)."""
    material_entries: list = []
    if simulation_config.material.resistivity_mode == rf.MATERIALESPECIFICO:
        material_entries = rf.get_entries_from_material_file(
            int(geometry_data.n_triangles),
            simulation_config.material.material_path,
        )

    wavelength_m = 3e8 / simulation_config.frequency_hz
    normalized_correlation_distance = simulation_config.correlation_distance_m / wavelength_m
    (
        wave_number,
        roughness_factor_primary,
        roughness_factor_secondary,
        radian_factor,
        taylor_threshold,
        taylor_terms,
    ) = rf.get_standard_deviation(
        simulation_config.standard_deviation_m,
        normalized_correlation_distance,
        wavelength_m,
    )
    _, electric_theta_component, electric_phi_component = rf.get_polarization(
        simulation_config.incident_polarization
    )
    incident_amplitude = 1.0

    (
        triangle_areas,
        surface_alpha_angles,
        surface_beta_angles,
        surface_normals,
        edge_lengths,
        phi_count,
        theta_count,
    ) = rf.calculate_values(
        simulation_config.angle_sweep.phi_start_deg,
        simulation_config.angle_sweep.phi_stop_deg,
        simulation_config.angle_sweep.phi_step_deg,
        simulation_config.angle_sweep.theta_start_deg,
        simulation_config.angle_sweep.theta_stop_deg,
        simulation_config.angle_sweep.theta_step_deg,
        int(geometry_data.n_triangles),
        radian_factor,
    )
    (
        surface_normals,
        edge_lengths,
        triangle_areas,
        surface_beta_angles,
        surface_alpha_angles,
    ) = rf.product_vector(
        int(geometry_data.n_triangles),
        surface_normals,
        geometry_data.vertex_coordinates,
        edge_lengths,
        triangle_areas,
        surface_alpha_angles,
        surface_beta_angles,
        geometry_data.vertex_indices,
    )
    (
        phi_grid_deg,
        theta_grid_deg,
        direction_cosine_u_grid,
        direction_cosine_v_grid,
        direction_cosine_w_grid,
        incident_field_cartesian,
        rcs_theta_db,
        rcs_phi_db,
    ) = rf.other_vector_components(phi_count, theta_count)

    phi_values_deg = (
        simulation_config.angle_sweep.phi_start_deg
        + np.arange(phi_count, dtype=float) * simulation_config.angle_sweep.phi_step_deg
    )
    theta_values_deg = (
        simulation_config.angle_sweep.theta_start_deg
        + np.arange(theta_count, dtype=float) * simulation_config.angle_sweep.theta_step_deg
    )
    total_samples = phi_count * theta_count
    completed_samples = 0
    callback_stride = max(1, int(progress_update_stride))

    for phi_index, phi_deg in enumerate(phi_values_deg):
        phi_radians = phi_deg * radian_factor
        for theta_index, theta_deg in enumerate(theta_values_deg):
            theta_radians = theta_deg * radian_factor
            phi_grid_deg[phi_index, theta_index] = phi_deg
            theta_grid_deg[phi_index, theta_index] = theta_deg

            (
                direction_cosine_u_grid,
                direction_cosine_v_grid,
                direction_cosine_w_grid,
                global_direction_vector,
                theta_projection_u,
                theta_projection_v,
                theta_projection_w,
                direction_u,
                direction_v,
                direction_w,
            ) = rf.global_angles(
                direction_cosine_u_grid,
                direction_cosine_v_grid,
                direction_cosine_w_grid,
                theta_radians,
                phi_radians,
                phi_index,
                theta_index,
            )
            observation_direction_vector = np.array([direction_u, direction_v, direction_w], dtype=float)
            incident_field_cartesian = rf.incident_field_cartesian(
                theta_projection_u,
                theta_projection_v,
                theta_projection_w,
                incident_field_cartesian,
                electric_theta_component,
                phi_radians,
                electric_phi_component,
            )

            accumulated_fields = FieldAccumulator()
            for triangle_index in range(int(geometry_data.n_triangles)):
                normal_dot_observer = float(
                    np.dot(surface_normals[triangle_index, :], observation_direction_vector)
                )
                if geometry_data.illumination_flag_mode == 0 and not (
                    (
                        geometry_data.illumination_flags[triangle_index] == 1
                        and normal_dot_observer >= 1e-5
                    )
                    or geometry_data.illumination_flags[triangle_index] == 0
                ):
                    continue

                local_u, local_v, local_w, transform_z, transform_y = rf.direction_cosines(
                    surface_alpha_angles,
                    surface_beta_angles,
                    global_direction_vector,
                    triangle_index,
                )
                local_theta_angle, local_phi_angle = rf.spherical_angles(local_u, local_v, local_w)
                phase_p, phase_q, phase_origin = rf.phase_vertex_triangle(
                    geometry_data.x,
                    geometry_data.y,
                    geometry_data.z,
                    geometry_data.vertex_indices,
                    wave_number,
                    triangle_index,
                    direction_u,
                    direction_v,
                    direction_w,
                )

                local_field_step_1 = np.dot(transform_z, np.transpose(np.conj(incident_field_cartesian)))
                local_field_components = np.dot(transform_y, local_field_step_1)
                local_theta_field, local_phi_field = rf.incident_field_spherical_coordinates(
                    local_theta_angle,
                    local_field_components,
                    local_phi_angle,
                )

                reflection_perpendicular, reflection_parallel = rf.reflection_coefficients(
                    geometry_data.resistivity_values[triangle_index],
                    triangle_index,
                    local_theta_angle,
                    theta_radians,
                    phi_radians,
                    surface_alpha_angles[triangle_index],
                    surface_beta_angles[triangle_index],
                    simulation_config.frequency_hz,
                    material_entries,
                )
                local_surface_current_x = -local_theta_field * math.cos(local_phi_angle) * reflection_parallel + (
                    local_phi_field
                    * math.sin(local_phi_angle)
                    * reflection_perpendicular
                    * math.cos(local_theta_angle)
                )
                local_surface_current_y = -local_theta_field * math.sin(local_phi_angle) * reflection_parallel - (
                    local_phi_field
                    * math.cos(local_phi_angle)
                    * reflection_perpendicular
                    * math.cos(local_theta_angle)
                )

                phase_difference, exp_phase_origin, exp_phase_p, exp_phase_q = rf.area_integral(
                    phase_q,
                    phase_p,
                    phase_origin,
                )
                area_integral_value = rf.calculate_ic(
                    phase_p,
                    phase_q,
                    phase_origin,
                    surface_normals,
                    taylor_terms,
                    triangle_areas,
                    exp_phase_origin,
                    incident_amplitude,
                    taylor_threshold,
                    phase_difference,
                    exp_phase_q,
                    triangle_index,
                    exp_phase_p,
                )

                (
                    accumulated_fields.theta_component,
                    accumulated_fields.phi_component,
                    accumulated_fields.diffuse_phi,
                    accumulated_fields.diffuse_theta,
                ) = rf.calculate_fields(
                    triangle_areas,
                    roughness_factor_secondary,
                    normalized_correlation_distance,
                    local_theta_angle,
                    wavelength_m,
                    local_surface_current_y,
                    area_integral_value,
                    theta_projection_u,
                    theta_projection_v,
                    theta_projection_w,
                    phi_radians,
                    accumulated_fields.theta_component,
                    accumulated_fields.phi_component,
                    accumulated_fields.diffuse_theta,
                    accumulated_fields.diffuse_phi,
                    triangle_index,
                    local_surface_current_x,
                    transform_z,
                    transform_y,
                )

            rf.calculate_sth_sph(
                roughness_factor_primary,
                accumulated_fields.theta_component,
                accumulated_fields.phi_component,
                accumulated_fields.diffuse_theta,
                wavelength_m,
                rcs_theta_db,
                rcs_phi_db,
                phi_index,
                theta_index,
                accumulated_fields.diffuse_phi,
            )
            completed_samples += 1
            if progress_callback and (
                completed_samples % callback_stride == 0 or completed_samples == total_samples
            ):
                progress_callback(completed_samples, total_samples)

    return RcsComputationResult(
        mode="Monostatic",
        input_model=simulation_config.input_model,
        wavelength_m=wavelength_m,
        phi_grid_deg=phi_grid_deg,
        theta_grid_deg=theta_grid_deg,
        rcs_theta_db=rcs_theta_db,
        rcs_phi_db=rcs_phi_db,
        direction_cosine_u_grid=direction_cosine_u_grid,
        direction_cosine_v_grid=direction_cosine_v_grid,
        direction_cosine_w_grid=direction_cosine_w_grid,
        geometry=geometry_data,
    )


def run_monostatic(
    simulation_config: MonostaticSimulationConfig,
    geometry_data: GeometryData,
) -> SolverResult:
    """Run monostatic simulation and persist standard artifacts to disk."""
    simulation_result = simulate_monostatic(simulation_config, geometry_data)
    rcs_max_db, rcs_min_db = rf.plot_limits(simulation_result.rcs_theta_db, simulation_result.rcs_phi_db)
    polarization_label = rf.get_polarization(simulation_config.incident_polarization)[0]

    rf.set_font_option()
    warnings.filterwarnings("ignore")
    figure_path = rf.plot_triangle_model(
        simulation_config.input_model,
        geometry_data.vertex_indices,
        geometry_data.x,
        geometry_data.y,
        geometry_data.z,
        geometry_data.x_points,
        geometry_data.y_points,
        geometry_data.z_points,
        geometry_data.n_vertices,
        int(geometry_data.n_triangles),
        geometry_data.node1,
        geometry_data.node2,
        geometry_data.node3,
        geometry_data.facet_numbers,
    )
    params_text = rf.plot_parameters(
        "Monostatic",
        simulation_config.frequency_hz,
        simulation_result.wavelength_m,
        simulation_config.correlation_distance_m,
        simulation_config.standard_deviation_m,
        polarization_label,
        int(geometry_data.n_triangles),
        simulation_config.angle_sweep.phi_start_deg,
        simulation_config.angle_sweep.phi_stop_deg,
        simulation_config.angle_sweep.phi_step_deg,
        simulation_config.angle_sweep.theta_start_deg,
        simulation_config.angle_sweep.theta_stop_deg,
        simulation_config.angle_sweep.theta_step_deg,
    )
    timestamp, data_path = rf.generate_result_files(
        simulation_result.theta_grid_deg,
        simulation_result.rcs_theta_db,
        simulation_result.phi_grid_deg,
        simulation_result.rcs_phi_db,
        params_text,
        simulation_result.phi_grid_deg.shape[0],
    )
    plot_path = rf.final_plot(
        simulation_result.phi_grid_deg.shape[0],
        simulation_result.theta_grid_deg.shape[1],
        simulation_result.phi_grid_deg,
        simulation_result.wavelength_m,
        simulation_result.theta_grid_deg,
        rcs_min_db,
        rcs_max_db,
        simulation_result.rcs_theta_db,
        simulation_result.rcs_phi_db,
        simulation_result.direction_cosine_u_grid,
        simulation_result.direction_cosine_v_grid,
        timestamp,
        simulation_config.input_model,
        "Monostatic",
    )
    return SolverResult(plot_path=plot_path, figure_path=figure_path, data_path=data_path)
