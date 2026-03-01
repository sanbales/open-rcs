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
    use_material_lookup = simulation_config.material.resistivity_mode == rf.MATERIALESPECIFICO
    if use_material_lookup:
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
    transform_z_all, transform_y_all = rf.precompute_rotation_matrices(
        surface_alpha_angles,
        surface_beta_angles,
    )
    phase_p_vectors, phase_q_vectors, phase_origin_vectors = rf.precompute_phase_geometry(
        geometry_data.vertex_coordinates,
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
    resistivity_values = geometry_data.resistivity_values
    illumination_flags = geometry_data.illumination_flags
    triangle_count = int(geometry_data.n_triangles)
    two_wave_number = 2.0 * wave_number
    cosine_alpha_all = transform_z_all[:, 0, 0]
    sine_alpha_all = transform_z_all[:, 0, 1]
    cosine_beta_all = transform_y_all[:, 0, 0]
    sine_beta_all = transform_y_all[:, 2, 0]
    surface_normal_x = surface_normals[:, 0]
    surface_normal_y = surface_normals[:, 1]
    surface_normal_z = surface_normals[:, 2]
    phase_p_x = phase_p_vectors[:, 0]
    phase_p_y = phase_p_vectors[:, 1]
    phase_p_z = phase_p_vectors[:, 2]
    phase_q_x = phase_q_vectors[:, 0]
    phase_q_y = phase_q_vectors[:, 1]
    phase_q_z = phase_q_vectors[:, 2]
    phase_o_x = phase_origin_vectors[:, 0]
    phase_o_y = phase_origin_vectors[:, 1]
    phase_o_z = phase_origin_vectors[:, 2]
    use_numba_kernel = (
        simulation_config.use_numba and rf.NUMBA_AVAILABLE and not use_material_lookup
    )

    for phi_index, phi_deg in enumerate(phi_values_deg):
        phi_radians = phi_deg * radian_factor
        sine_phi = math.sin(phi_radians)
        cosine_phi = math.cos(phi_radians)
        for theta_index, theta_deg in enumerate(theta_values_deg):
            theta_radians = theta_deg * radian_factor
            phi_grid_deg[phi_index, theta_index] = phi_deg
            theta_grid_deg[phi_index, theta_index] = theta_deg

            (
                direction_cosine_u_grid,
                direction_cosine_v_grid,
                direction_cosine_w_grid,
                _global_direction_vector,
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
            incident_field_cartesian = rf.incident_field_cartesian(
                theta_projection_u,
                theta_projection_v,
                theta_projection_w,
                incident_field_cartesian,
                electric_theta_component,
                phi_radians,
                electric_phi_component,
            )
            incident_field_x = incident_field_cartesian[0].conjugate()
            incident_field_y = incident_field_cartesian[1].conjugate()
            incident_field_z = incident_field_cartesian[2].conjugate()

            accumulated_fields = FieldAccumulator()
            if use_numba_kernel:
                (
                    accumulated_fields.theta_component,
                    accumulated_fields.phi_component,
                    accumulated_fields.diffuse_theta,
                    accumulated_fields.diffuse_phi,
                ) = rf.accumulate_monostatic_sample_numba(
                    illumination_flag_mode=geometry_data.illumination_flag_mode,
                    illumination_flags=illumination_flags,
                    resistivity_values=resistivity_values,
                    triangle_areas=triangle_areas,
                    surface_alpha_cos=cosine_alpha_all,
                    surface_alpha_sin=sine_alpha_all,
                    surface_beta_cos=cosine_beta_all,
                    surface_beta_sin=sine_beta_all,
                    surface_normal_x=surface_normal_x,
                    surface_normal_y=surface_normal_y,
                    surface_normal_z=surface_normal_z,
                    phase_p_x=phase_p_x,
                    phase_p_y=phase_p_y,
                    phase_p_z=phase_p_z,
                    phase_q_x=phase_q_x,
                    phase_q_y=phase_q_y,
                    phase_q_z=phase_q_z,
                    phase_o_x=phase_o_x,
                    phase_o_y=phase_o_y,
                    phase_o_z=phase_o_z,
                    direction_u=direction_u,
                    direction_v=direction_v,
                    direction_w=direction_w,
                    theta_projection_u=theta_projection_u,
                    theta_projection_v=theta_projection_v,
                    theta_projection_w=theta_projection_w,
                    sine_phi=sine_phi,
                    cosine_phi=cosine_phi,
                    incident_field_x=incident_field_x,
                    incident_field_y=incident_field_y,
                    incident_field_z=incident_field_z,
                    two_wave_number=two_wave_number,
                    roughness_factor_secondary=roughness_factor_secondary,
                    normalized_correlation_distance=normalized_correlation_distance,
                    wavelength_m=wavelength_m,
                    incident_amplitude=incident_amplitude,
                    taylor_terms=taylor_terms,
                    taylor_threshold=taylor_threshold,
                )
            else:
                for triangle_index in range(triangle_count):
                    normal_dot_observer = (
                        surface_normal_x[triangle_index] * direction_u
                        + surface_normal_y[triangle_index] * direction_v
                        + surface_normal_z[triangle_index] * direction_w
                    )
                    if geometry_data.illumination_flag_mode == 0 and not (
                        (illumination_flags[triangle_index] == 1 and normal_dot_observer >= 1e-5)
                        or illumination_flags[triangle_index] == 0
                    ):
                        continue

                    cosine_alpha = float(cosine_alpha_all[triangle_index])
                    sine_alpha = float(sine_alpha_all[triangle_index])
                    cosine_beta = float(cosine_beta_all[triangle_index])
                    sine_beta = float(sine_beta_all[triangle_index])

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
                    local_phi_field = (
                        -local_field_x * local_sine_phi + local_field_y * local_cosine_phi
                    )

                    local_theta_angle = math.asin(local_sine_theta)
                    reflection_perpendicular, reflection_parallel = rf.reflection_coefficients(
                        int(resistivity_values[triangle_index]),
                        triangle_index,
                        local_theta_angle,
                        theta_radians,
                        phi_radians,
                        surface_alpha_angles[triangle_index],
                        surface_beta_angles[triangle_index],
                        simulation_config.frequency_hz,
                        material_entries,
                        local_cos_theta=local_cosine_theta,
                    )

                    local_surface_current_x = (
                        -local_theta_field * local_cosine_phi * reflection_parallel
                        + local_phi_field
                        * local_sine_phi
                        * reflection_perpendicular
                        * local_cosine_theta
                    )
                    local_surface_current_y = (
                        -local_theta_field * local_sine_phi * reflection_parallel
                        - local_phi_field
                        * local_cosine_phi
                        * reflection_perpendicular
                        * local_cosine_theta
                    )

                    area_integral_value = rf.calculate_ic(
                        phase_p,
                        phase_q,
                        phase_origin,
                        taylor_terms,
                        float(triangle_areas[triangle_index]),
                        incident_amplitude,
                        taylor_threshold,
                    )

                    triangle_area = float(triangle_areas[triangle_index])
                    diffuse_scale = (
                        roughness_factor_secondary * triangle_area * (local_cosine_theta**2)
                    )
                    diffuse_exponent = -(
                        (
                            normalized_correlation_distance
                            * math.pi
                            * local_sine_theta
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
                        cosine_alpha * cosine_beta * scattered_local_x
                        - sine_alpha * scattered_local_y
                    )
                    scattered_global_y = (
                        sine_alpha * cosine_beta * scattered_local_x
                        + cosine_alpha * scattered_local_y
                    )
                    scattered_global_z = -sine_beta * scattered_local_x

                    diffuse_global_x = (
                        cosine_alpha * cosine_beta * diffuse_local_x - sine_alpha * diffuse_local_y
                    )
                    diffuse_global_y = (
                        sine_alpha * cosine_beta * diffuse_local_x + cosine_alpha * diffuse_local_y
                    )
                    diffuse_global_z = -sine_beta * diffuse_local_x

                    accumulated_fields.theta_component += (
                        theta_projection_u * scattered_global_x
                        + theta_projection_v * scattered_global_y
                        + theta_projection_w * scattered_global_z
                    )
                    accumulated_fields.phi_component += (
                        -sine_phi * scattered_global_x + cosine_phi * scattered_global_y
                    )
                    accumulated_fields.diffuse_theta += abs(
                        theta_projection_u * diffuse_global_x
                        + theta_projection_v * diffuse_global_y
                        + theta_projection_w * diffuse_global_z
                    )
                    accumulated_fields.diffuse_phi += abs(
                        -sine_phi * diffuse_global_x + cosine_phi * diffuse_global_y
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
    rcs_max_db, rcs_min_db = rf.plot_limits(
        simulation_result.rcs_theta_db, simulation_result.rcs_phi_db
    )
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
