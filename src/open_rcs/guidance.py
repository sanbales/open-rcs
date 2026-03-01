"""Analyst guidance helpers for roughness and correlation setup."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model_types import GeometryData


@dataclass(slots=True)
class RoughnessCorrelationGuidance:
    """Suggested roughness/correlation settings based on frequency and geometry scale."""

    wavelength_m: float
    electrical_size_l_over_lambda: float
    median_edge_length_m: float
    suggested_standard_deviation_m: float
    suggested_correlation_distance_m: float
    standard_deviation_bounds_m: tuple[float, float]
    correlation_distance_bounds_m: tuple[float, float]
    notes: tuple[str, ...]


def _triangle_edge_lengths(geometry_data: GeometryData) -> np.ndarray:
    zero_based_indices = geometry_data.vertex_indices.astype(np.int64) - 1
    vertex_coordinates = geometry_data.vertex_coordinates
    p0 = vertex_coordinates[zero_based_indices[:, 0]]
    p1 = vertex_coordinates[zero_based_indices[:, 1]]
    p2 = vertex_coordinates[zero_based_indices[:, 2]]
    edge01 = np.linalg.norm(p1 - p0, axis=1)
    edge12 = np.linalg.norm(p2 - p1, axis=1)
    edge20 = np.linalg.norm(p0 - p2, axis=1)
    return np.concatenate([edge01, edge12, edge20])


def estimate_roughness_correlation_guidance(
    frequency_hz: float,
    geometry_data: GeometryData,
) -> RoughnessCorrelationGuidance:
    """Estimate reasonable starting values for roughness and correlation inputs."""
    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be positive.")

    wavelength_m = 3e8 / frequency_hz
    bbox_min = np.min(geometry_data.vertex_coordinates, axis=0)
    bbox_max = np.max(geometry_data.vertex_coordinates, axis=0)
    characteristic_length_m = float(np.max(bbox_max - bbox_min))
    electrical_size = characteristic_length_m / wavelength_m if wavelength_m > 0 else 0.0

    edge_lengths = _triangle_edge_lengths(geometry_data)
    positive_edge_lengths = edge_lengths[edge_lengths > 0]
    median_edge_length_m = float(np.median(positive_edge_lengths)) if positive_edge_lengths.size else wavelength_m

    sigma_min = wavelength_m / 1000.0
    sigma_max = wavelength_m / 50.0
    suggested_sigma = wavelength_m / 200.0
    suggested_sigma = float(np.clip(suggested_sigma, sigma_min, sigma_max))

    corr_min = max(wavelength_m / 10.0, 0.5 * median_edge_length_m)
    corr_max = max(corr_min, 5.0 * median_edge_length_m)
    suggested_corr = float(np.clip(median_edge_length_m, corr_min, corr_max))

    notes: list[str] = [
        "Guidance assumes high-frequency PO usage and isotropic roughness statistics.",
        "Use these values as a starting point, then calibrate against known references.",
    ]
    if electrical_size < 10.0:
        notes.append("Target appears electrically small (L/lambda < 10). PO approximation may be less accurate.")
    edge_to_wavelength = median_edge_length_m / wavelength_m if wavelength_m > 0 else 0.0
    if edge_to_wavelength < 2.5:
        notes.append("Median facet edge is relatively fine (< 2.5 lambda). Runtime may increase significantly.")
    elif edge_to_wavelength > 4.5:
        notes.append("Median facet edge is relatively coarse (> 4.5 lambda). Facet noise risk may increase.")

    return RoughnessCorrelationGuidance(
        wavelength_m=wavelength_m,
        electrical_size_l_over_lambda=electrical_size,
        median_edge_length_m=median_edge_length_m,
        suggested_standard_deviation_m=suggested_sigma,
        suggested_correlation_distance_m=suggested_corr,
        standard_deviation_bounds_m=(sigma_min, sigma_max),
        correlation_distance_bounds_m=(corr_min, corr_max),
        notes=tuple(notes),
    )
