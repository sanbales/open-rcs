"""Material-library parsing and reflection-coefficient models for Open RCS."""

from __future__ import annotations

import cmath
import importlib
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import numpy as np

from .constants import (
    MATERIAL_TYPE_TO_CODE,
    SPECIFIC_MATERIAL,
    MaterialEntry,
    MaterialEntryIndex,
    MaterialLayer,
    MaterialTable,
    MaterialType,
    SphericalIndex,
)

yaml: Any
try:
    yaml = importlib.import_module("yaml")
except ImportError:  # pragma: no cover - optional material library dependency
    yaml = None


def compile_material_lookup_arrays(
    material_table: MaterialTable,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert heterogeneous material entries into dense numeric arrays for fast kernels."""
    facet_count = len(material_table)
    max_layers = max((len(entry) - MaterialEntryIndex.FIRST_LAYER for entry in material_table), default=0)
    max_layers = max(1, max_layers)

    material_type_codes = np.zeros(facet_count, dtype=np.int32)
    material_layer_count = np.zeros(facet_count, dtype=np.int32)
    epsilon_r = np.zeros((facet_count, max_layers), dtype=np.float64)
    loss_tangent = np.zeros((facet_count, max_layers), dtype=np.float64)
    mu_r_real = np.zeros((facet_count, max_layers), dtype=np.float64)
    mu_r_imag = np.zeros((facet_count, max_layers), dtype=np.float64)
    thickness_m = np.zeros((facet_count, max_layers), dtype=np.float64)

    for facet_index, entry in enumerate(material_table):
        material_type = str(entry[MaterialEntryIndex.TYPE])
        if material_type not in MATERIAL_TYPE_TO_CODE:
            raise ValueError(f"Unsupported material type: {material_type}")
        material_type_codes[facet_index] = MATERIAL_TYPE_TO_CODE[material_type]

        layers = cast(list[list[float]], entry[MaterialEntryIndex.FIRST_LAYER :])
        material_layer_count[facet_index] = len(layers)
        for layer_index, layer in enumerate(layers):
            if layer_index >= max_layers:
                break
            epsilon_r[facet_index, layer_index] = float(layer[0])
            loss_tangent[facet_index, layer_index] = float(layer[1])
            mu_r_real[facet_index, layer_index] = float(layer[2])
            mu_r_imag[facet_index, layer_index] = float(layer[3])
            thickness_m[facet_index, layer_index] = float(layer[4]) * 1e-3

    return (
        material_type_codes,
        material_layer_count,
        epsilon_r,
        loss_tangent,
        mu_r_real,
        mu_r_imag,
        thickness_m,
    )


def _canonical_material_type(material_type: str) -> str:
    normalized_type = material_type.strip().lower()
    aliases = {
        "pec": MaterialType.PEC.value,
        "composite": MaterialType.COMPOSITE.value,
        "composite layer on pec": MaterialType.COMPOSITE_ON_PEC.value,
        "multiple layers": MaterialType.MULTI_LAYER.value,
        "multiple layers on pec": MaterialType.MULTI_LAYER_ON_PEC.value,
        "composito": MaterialType.COMPOSITE.value,
        "camada de composito em pec": MaterialType.COMPOSITE_ON_PEC.value,
        "multiplas camadas": MaterialType.MULTI_LAYER.value,
        "multiplas camadas em pec": MaterialType.MULTI_LAYER_ON_PEC.value,
    }
    if normalized_type not in aliases:
        raise ValueError(f"Unsupported material type: {material_type}")
    return aliases[normalized_type]


def _parse_material_layer(layer_data: Any, context: str) -> MaterialLayer:
    if isinstance(layer_data, dict):
        raw_values = [
            layer_data.get("epsilon_r"),
            layer_data.get("loss_tangent"),
            layer_data.get("mu_r_real"),
            layer_data.get("mu_r_imag"),
            layer_data.get("thickness_mm"),
        ]
    elif isinstance(layer_data, (list, tuple)) and len(layer_data) == 5:
        raw_values = list(layer_data)
    else:
        raise ValueError(
            f"{context}: each layer must be either a mapping with "
            "'epsilon_r/loss_tangent/mu_r_real/mu_r_imag/thickness_mm' or a 5-value list."
        )

    parsed_values: MaterialLayer = []
    for value in raw_values:
        if value is None:
            raise ValueError(f"{context}: all layer fields must be present.")
        try:
            parsed_values.append(float(cast(Any, value)))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{context}: all layer values must be numeric.") from exc
    return parsed_values


def _clone_material_entry(material_entry: MaterialEntry) -> MaterialEntry:
    cloned_entry: MaterialEntry = [
        str(material_entry[MaterialEntryIndex.TYPE]),
        str(material_entry[MaterialEntryIndex.DESCRIPTION]),
    ]
    for layer in material_entry[MaterialEntryIndex.FIRST_LAYER :]:
        cloned_entry.append(list(cast(MaterialLayer, layer)))
    return cloned_entry


def _material_entry_from_yaml(material_definition: dict[str, Any], index: int) -> tuple[str, MaterialEntry]:
    material_id = str(material_definition.get("id", "")).strip()
    if not material_id:
        raise ValueError(f"materials[{index}] is missing a non-empty 'id'.")
    material_type = _canonical_material_type(str(material_definition.get("type", "")))
    description = str(material_definition.get("description", material_id))
    layers_raw = material_definition.get("layers", [])
    if layers_raw is None:
        layers_raw = []
    if not isinstance(layers_raw, list):
        raise ValueError(f"materials[{index}]: 'layers' must be a list.")

    material_entry: MaterialEntry = [material_type, description]
    if material_type != MaterialType.PEC.value and not layers_raw:
        raise ValueError(f"materials[{index}] ({material_id}) requires at least one layer.")
    for layer_index, layer_data in enumerate(layers_raw):
        context = f"materials[{index}].layers[{layer_index}]"
        material_entry.append(_parse_material_layer(layer_data, context))
    return material_id, material_entry


def _parse_facet_selector(selector: Any, ntria: int, context: str) -> set[int]:
    selected_facets: set[int] = set()
    if isinstance(selector, str):
        if selector.strip().lower() != "all":
            raise ValueError(f"{context}: unsupported selector string '{selector}'.")
        return set(range(ntria))
    if not isinstance(selector, list):
        raise ValueError(f"{context}: selector must be 'all' or a list.")

    for item in selector:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            start = int(item[0])
            stop = int(item[1])
            if stop < start:
                start, stop = stop, start
            if start < 1 or stop > ntria:
                raise ValueError(f"{context}: range [{start}, {stop}] is outside valid facet IDs 1..{ntria}.")
            selected_facets.update(range(start - 1, stop))
        else:
            facet_id = int(item)
            if facet_id < 1 or facet_id > ntria:
                raise ValueError(f"{context}: facet ID {facet_id} is outside valid range 1..{ntria}.")
            selected_facets.add(facet_id - 1)
    return selected_facets


def _parse_tags(tag_definitions: Any, ntria: int) -> dict[str, set[int]]:
    if tag_definitions is None:
        return {}
    if not isinstance(tag_definitions, dict):
        raise ValueError("'tags' must be a mapping from tag name to facet selectors.")
    parsed_tags: dict[str, set[int]] = {}
    for tag_name, selector in tag_definitions.items():
        if not isinstance(tag_name, str) or not tag_name.strip():
            raise ValueError("Tag names must be non-empty strings.")
        parsed_tags[tag_name] = _parse_facet_selector(
            selector,
            ntria,
            context=f"tags.{tag_name}",
        )
    return parsed_tags


def _resolve_assignment_selector(
    assignment: dict[str, Any],
    ntria: int,
    parsed_tags: dict[str, set[int]],
    assignment_index: int,
) -> set[int]:
    selection: set[int] = set()
    explicit_selector_used = False

    if "facets" in assignment:
        explicit_selector_used = True
        selection.update(
            _parse_facet_selector(
                assignment["facets"],
                ntria,
                context=f"assignments[{assignment_index}].facets",
            )
        )
    if "facet_indices" in assignment:
        explicit_selector_used = True
        selection.update(
            _parse_facet_selector(
                assignment["facet_indices"],
                ntria,
                context=f"assignments[{assignment_index}].facet_indices",
            )
        )
    if "facet_ranges" in assignment:
        explicit_selector_used = True
        selection.update(
            _parse_facet_selector(
                assignment["facet_ranges"],
                ntria,
                context=f"assignments[{assignment_index}].facet_ranges",
            )
        )

    for key in ("tags", "groups"):
        if key in assignment:
            explicit_selector_used = True
            raw_names = assignment[key]
            if not isinstance(raw_names, list):
                raise ValueError(f"assignments[{assignment_index}].{key} must be a list of tag names.")
            for raw_name in raw_names:
                tag_name = str(raw_name)
                if tag_name not in parsed_tags:
                    raise ValueError(f"assignments[{assignment_index}].{key} references unknown tag '{tag_name}'.")
                selection.update(parsed_tags[tag_name])

    if not explicit_selector_used:
        return set(range(ntria))
    if not selection:
        raise ValueError(f"assignments[{assignment_index}] does not select any facets.")
    return selection


def _material_table_from_yaml_document(document: dict[str, Any], ntria: int) -> MaterialTable:
    materials_raw = document.get("materials")
    if not isinstance(materials_raw, list) or not materials_raw:
        raise ValueError("YAML material file must define a non-empty 'materials' list.")

    material_catalog: dict[str, MaterialEntry] = {}
    for material_index, raw_material in enumerate(materials_raw):
        if not isinstance(raw_material, dict):
            raise ValueError(f"materials[{material_index}] must be a mapping.")
        material_id, material_entry = _material_entry_from_yaml(raw_material, material_index)
        if material_id in material_catalog:
            raise ValueError(f"Duplicate material id '{material_id}' in YAML file.")
        material_catalog[material_id] = material_entry

    default_material = document.get("default_material")
    if default_material is None and isinstance(document.get("defaults"), dict):
        default_material = cast(dict[str, Any], document["defaults"]).get("material")
    default_material_id = str(default_material).strip() if default_material is not None else ""
    if default_material_id and default_material_id not in material_catalog:
        raise ValueError(f"default_material '{default_material_id}' is not defined in materials.")

    parsed_tags = _parse_tags(document.get("tags"), ntria)
    assignments_raw = document.get("assignments", [])
    if assignments_raw is None:
        assignments_raw = []
    if not isinstance(assignments_raw, list):
        raise ValueError("'assignments' must be a list when provided.")

    facet_material_ids: list[str | None] = [None] * ntria
    if default_material_id:
        for facet_index in range(ntria):
            facet_material_ids[facet_index] = default_material_id

    for assignment_index, raw_assignment in enumerate(assignments_raw):
        if not isinstance(raw_assignment, dict):
            raise ValueError(f"assignments[{assignment_index}] must be a mapping.")
        material_id = str(raw_assignment.get("material", "")).strip()
        if not material_id:
            raise ValueError(f"assignments[{assignment_index}] is missing 'material'.")
        if material_id not in material_catalog:
            raise ValueError(f"assignments[{assignment_index}] references unknown material id '{material_id}'.")
        selected_facets = _resolve_assignment_selector(
            raw_assignment,
            ntria,
            parsed_tags,
            assignment_index,
        )
        for facet_index in selected_facets:
            facet_material_ids[facet_index] = material_id

    missing_facets = [
        index + 1 for index, assigned_material_id in enumerate(facet_material_ids) if assigned_material_id is None
    ]
    if missing_facets:
        preview = ", ".join(str(value) for value in missing_facets[:10])
        suffix = "..." if len(missing_facets) > 10 else ""
        raise ValueError(f"YAML material assignments are incomplete. Missing facet IDs: {preview}{suffix}")

    material_table: MaterialTable = []
    for assigned_material_id in facet_material_ids:
        assert assigned_material_id is not None
        material_table.append(_clone_material_entry(material_catalog[assigned_material_id]))
    return material_table


def load_material_catalog(material_path: str | Path) -> dict[str, MaterialEntry]:
    """Load and validate the material catalog section from a YAML .rcsmat file."""
    if yaml is None:  # pragma: no cover - runtime dependency guard
        raise ImportError("YAML material files require PyYAML. Install with: pip install pyyaml")

    with Path(material_path).open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise ValueError("YAML material file root must be a mapping.")

    materials_raw = document.get("materials")
    if not isinstance(materials_raw, list) or not materials_raw:
        raise ValueError("YAML material file must define a non-empty 'materials' list.")

    material_catalog: dict[str, MaterialEntry] = {}
    for material_index, raw_material in enumerate(materials_raw):
        if not isinstance(raw_material, dict):
            raise ValueError(f"materials[{material_index}] must be a mapping.")
        material_id, material_entry = _material_entry_from_yaml(raw_material, material_index)
        if material_id in material_catalog:
            raise ValueError(f"Duplicate material id '{material_id}' in YAML file.")
        material_catalog[material_id] = _clone_material_entry(material_entry)
    return material_catalog


def _load_material_table_from_yaml(material_path: str | Path, ntria: int) -> MaterialTable:
    if yaml is None:  # pragma: no cover - runtime dependency guard
        raise ImportError("YAML material files require PyYAML. Install with: pip install pyyaml")

    with Path(material_path).open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise ValueError("YAML material file root must be a mapping.")
    return _material_table_from_yaml_document(document, ntria)


def save_list_in_file(material_rows: list, output_file: str) -> None:
    """Write parsed material entries to the legacy CSV-like text format."""
    serialized_rows = []
    for row in material_rows:
        entry_str = str(row[MaterialEntryIndex.TYPE]) + "," + str(row[MaterialEntryIndex.DESCRIPTION])
        for layer in row[MaterialEntryIndex.FIRST_LAYER :]:
            for i in range(len(layer)):
                entry_str = entry_str + "," + str(layer[i])
        serialized_rows.append(entry_str)

    with open(output_file, "w", encoding="utf-8") as file:
        for row in serialized_rows:
            file.write(row + "\n")


def get_entries_from_material_file(ntria: int, matrlpath: str) -> MaterialTable:
    """Load material entries from text or YAML and validate facet count alignment."""
    material_path = Path(matrlpath)
    try:
        if material_path.suffix.lower() in {".rcsmat", ".yaml", ".yml"}:
            matrl = _load_material_table_from_yaml(material_path, ntria)
        else:
            with material_path.open("r", encoding="utf-8") as file:
                matrl = get_material_properties_from_file(file)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Material file not found: {material_path}") from exc
    except ValueError as exc:
        raise ValueError(f"Invalid material file: {material_path}") from exc

    if ntria != len(matrl):
        raise ValueError("Number of material entries does not match the number of facets.")

    return matrl


def get_material_properties_from_file(rows: Iterable[str]) -> MaterialTable:
    """Parse material entries from an iterable of text rows."""
    material_text_rows = list(rows)
    return convert_material_textlist_to_list(material_text_rows)


def convert_material_textlist_to_list(text_rows: Iterable[str]) -> MaterialTable:
    """Convert legacy comma-separated material rows into structured entries."""
    material_table: MaterialTable = []
    for raw_row in text_rows:
        raw_row = raw_row.strip()
        if not raw_row:
            continue
        entries = [value.strip() for value in raw_row.split(",")]
        formatted_entries: MaterialEntry = [
            _canonical_material_type(entries[MaterialEntryIndex.TYPE]),
            entries[MaterialEntryIndex.DESCRIPTION],
        ]

        layer_values = entries[MaterialEntryIndex.FIRST_LAYER :]
        for index, entry in enumerate(layer_values):
            if index % 5 == 0:
                formatted_entries.append([])
            cast(list[float], formatted_entries[-1]).append(float(entry))

        material_table.append(formatted_entries)
    return material_table


def rotation_transform_matrix(alpha, beta):
    """Build the global-to-local rotation matrix from facet alpha/beta angles."""
    T1 = np.array([[np.cos(alpha), np.sin(alpha), 0], [-np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]])
    T2 = np.array([[np.cos(beta), 0, -np.sin(beta)], [0, 1, 0], [np.sin(beta), 0, np.cos(beta)]])
    return np.dot(T2, T1)


def refl_coeff(er1, mr1, er2, mr2, thetai):
    """Compute Fresnel reflection coefficients at an interface between two media."""
    m0 = 4 * np.pi * 1e-7
    e0 = 8.854e-12
    TIR = 0
    sinthetat = np.sin(thetai) * np.sqrt(np.real(er1) * np.real(mr1) / (np.real(er2) * np.real(mr2)))
    if sinthetat > 1:
        TIR = 1
        thetat = np.pi / 2
    else:
        thetat = np.arcsin(sinthetat)

    n1 = np.sqrt(mr1 * m0 / (er1 * e0))
    n2 = np.sqrt(mr2 * m0 / (er2 * e0))
    gammaperp = (n2 * np.cos(thetai) - n1 * np.cos(thetat)) / (n2 * np.cos(thetai) + n1 * np.cos(thetat))
    gammapar = (n2 * np.cos(thetat) - n1 * np.cos(thetai)) / (n2 * np.cos(thetat) + n1 * np.cos(thetai))
    return gammapar, gammaperp, thetat, TIR


def spher2cart(spherical_vector: np.ndarray) -> np.ndarray:
    """Convert a spherical vector ``[r, theta, phi]`` to Cartesian coordinates."""
    radius = float(spherical_vector[0])
    theta = float(spherical_vector[1])
    phi = float(spherical_vector[2])
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    return np.array([x, y, z])


def cart2spher(cart_vector: np.ndarray) -> np.ndarray:
    """Convert a Cartesian vector ``[x, y, z]`` to spherical coordinates."""
    x = float(cart_vector[0])
    y = float(cart_vector[1])
    z = float(cart_vector[2])
    radius = np.sqrt(x**2 + y**2 + z**2)
    theta = math.atan2(np.sqrt(x**2 + y**2), z)
    phi = math.atan2(y, x)
    return np.array([radius, theta, phi], dtype=float)


def spherical_global_to_local(
    spherical_vector: np.ndarray,
    transform_matrix_global_to_local: np.ndarray,
) -> np.ndarray:
    """Apply a rotation to a spherical vector via Cartesian conversion."""
    cartesian_vector = spher2cart(spherical_vector)
    cartesian_vector = np.dot(transform_matrix_global_to_local, cartesian_vector)
    return cart2spher(cartesian_vector)


def _resolve_local_incidence_theta(
    thri: float,
    phrii: float,
    alpha: float,
    beta: float,
    local_theta: float | None,
) -> float:
    if local_theta is not None:
        return float(local_theta)
    transform_global_to_local = rotation_transform_matrix(alpha, beta)
    spherical_vector = spherical_global_to_local(
        np.array([1.0, thri, phrii], dtype=float),
        transform_global_to_local,
    )
    return float(spherical_vector[SphericalIndex.THETA])


def _mat2_mul(
    a00: complex,
    a01: complex,
    a10: complex,
    a11: complex,
    b00: complex,
    b01: complex,
    b10: complex,
    b11: complex,
) -> tuple[complex, complex, complex, complex]:
    return (
        a00 * b00 + a01 * b10,
        a00 * b01 + a01 * b11,
        a10 * b00 + a11 * b10,
        a10 * b01 + a11 * b11,
    )


def _layer_properties(layer: list[float]) -> tuple[complex, complex, float]:
    epsilon_r = layer[0]
    loss_tangent = layer[1]
    mu_r_real = layer[2]
    mu_r_imag = layer[3]
    thickness_m = layer[4] * 1e-3
    epsilon_complex = epsilon_r - 1j * loss_tangent * epsilon_r
    mu_complex = mu_r_real - 1j * mu_r_imag
    return epsilon_complex, mu_complex, thickness_m


def _apply_transfer_matrix(
    m00: complex,
    m01: complex,
    m10: complex,
    m11: complex,
    gamma: complex,
    phase: complex,
) -> tuple[complex, complex, complex, complex]:
    exp_phase = cmath.exp(1j * phase)
    exp_neg_phase = cmath.exp(-1j * phase)
    t00 = exp_phase
    t01 = gamma * exp_neg_phase
    t10 = gamma * exp_phase
    t11 = exp_neg_phase
    return _mat2_mul(m00, m01, m10, m11, t00, t01, t10, t11)


def refl_coeff_composite(
    thri: float,
    phrii: float,
    alpha: float,
    beta: float,
    freq: float,
    matrlLine: list,
    local_theta: float | None = None,
) -> tuple[complex, complex]:
    """Compute reflection coefficients for a single composite layer in free space."""
    layer = cast(list[float], matrlLine[MaterialEntryIndex.FIRST_LAYER])
    local_incidence_theta = _resolve_local_incidence_theta(thri, phrii, alpha, beta, local_theta)

    epsilon_complex, mu_complex, thickness_m = _layer_properties(layer)
    gamma_parallel_1, gamma_perpendicular_1, _theta_t, _tir = refl_coeff(
        1, 1, epsilon_complex, mu_complex, local_incidence_theta
    )
    gamma_parallel_2 = -gamma_parallel_1
    gamma_perpendicular_2 = -gamma_perpendicular_1
    wave_speed = 3e8 / math.sqrt(np.real(epsilon_complex) * np.real(mu_complex))
    phase = 2.0 * math.pi * thickness_m / (wave_speed / freq)

    m00p, m01p, m10p, m11p = _apply_transfer_matrix(
        1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 1.0 + 0j, complex(gamma_parallel_1), phase
    )
    m00s, m01s, m10s, m11s = _apply_transfer_matrix(
        1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 1.0 + 0j, complex(gamma_perpendicular_1), phase
    )
    m00p, m01p, m10p, m11p = _mat2_mul(
        m00p,
        m01p,
        m10p,
        m11p,
        1.0 + 0j,
        complex(gamma_parallel_2),
        complex(gamma_parallel_2),
        1.0 + 0j,
    )
    m00s, m01s, m10s, m11s = _mat2_mul(
        m00s,
        m01s,
        m10s,
        m11s,
        1.0 + 0j,
        complex(gamma_perpendicular_2),
        complex(gamma_perpendicular_2),
        1.0 + 0j,
    )
    reflection_parallel = m10p / m00p
    reflection_perpendicular = m10s / m00s
    return reflection_perpendicular, reflection_parallel


def refl_coeff_composite_layer_on_pec(
    thri: float,
    phrii: float,
    alpha: float,
    beta: float,
    freq: float,
    matrlLine: list,
    local_theta: float | None = None,
) -> tuple[complex, complex]:
    """Compute reflection coefficients for layered composite backed by PEC."""
    layers = cast(list[list[float]], matrlLine[MaterialEntryIndex.FIRST_LAYER :])
    local_incidence_theta = _resolve_local_incidence_theta(thri, phrii, alpha, beta, local_theta)
    sine_incidence = math.sin(local_incidence_theta)
    cosine_incidence = math.cos(local_incidence_theta)
    wave = 3e8 / freq
    beta_0 = 2.0 * math.pi / wave
    z_0 = 1.0 + 0j

    w00p, w01p, w10p, w11p = 1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 1.0 + 0j
    w00s, w01s, w10s, w11s = 1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 1.0 + 0j
    previous_z_parallel: complex | None = None
    previous_z_perpendicular: complex | None = None

    for layer_index, layer in enumerate(layers):
        epsilon_complex, mu_complex, thickness_m = _layer_properties(layer)
        impedance_ratio = epsilon_complex / mu_complex
        root_term = cmath.sqrt(impedance_ratio - sine_incidence**2)
        z_parallel = root_term / (impedance_ratio * cosine_incidence)
        z_perpendicular = cosine_incidence / root_term

        if layer_index == 0:
            gamma_parallel = (z_parallel - z_0) / (z_parallel + z_0)
            gamma_perpendicular = (z_perpendicular - z_0) / (z_perpendicular + z_0)
        else:
            assert previous_z_parallel is not None
            assert previous_z_perpendicular is not None
            gamma_parallel = (z_parallel - previous_z_parallel) / (z_parallel + previous_z_parallel)
            gamma_perpendicular = (z_perpendicular - previous_z_perpendicular) / (
                z_perpendicular + previous_z_perpendicular
            )

        previous_z_parallel = z_parallel
        previous_z_perpendicular = z_perpendicular
        tau_parallel = 1.0 + gamma_parallel
        tau_perpendicular = 1.0 + gamma_perpendicular
        phase = beta_0 * thickness_m * cmath.sqrt(epsilon_complex * mu_complex - sine_incidence**2)

        w00p, w01p, w10p, w11p = _apply_transfer_matrix(w00p, w01p, w10p, w11p, gamma_parallel, phase)
        w00s, w01s, w10s, w11s = _apply_transfer_matrix(w00s, w01s, w10s, w11s, gamma_perpendicular, phase)
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


def refl_coeff_multi_layers(
    thri: float,
    phrii: float,
    alpha: float,
    beta: float,
    freq: float,
    matrlLine: list,
    local_theta: float | None = None,
) -> tuple[complex, complex]:
    """Compute reflection coefficients for multiple dielectric/magnetic layers."""
    layers = cast(list[list[float]], matrlLine[MaterialEntryIndex.FIRST_LAYER :])
    local_incidence_theta = _resolve_local_incidence_theta(thri, phrii, alpha, beta, local_theta)

    m00p, m01p, m10p, m11p = 1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 1.0 + 0j
    m00s, m01s, m10s, m11s = 1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 1.0 + 0j
    previous_epsilon = 1.0 + 0j
    previous_mu = 1.0 + 0j
    previous_theta = local_incidence_theta
    last_phase = 0.0 + 0j

    for layer_index, layer in enumerate(layers):
        epsilon_complex, mu_complex, thickness_m = _layer_properties(layer)
        if layer_index == 0:
            gamma_parallel, gamma_perpendicular, theta_transmitted, _tir = refl_coeff(
                1.0 + 0j, 1.0 + 0j, epsilon_complex, mu_complex, local_incidence_theta
            )
        else:
            gamma_parallel, gamma_perpendicular, theta_transmitted, _tir = refl_coeff(
                previous_epsilon, previous_mu, epsilon_complex, mu_complex, previous_theta
            )

        wave_speed = 3e8 / math.sqrt(np.real(epsilon_complex) * np.real(mu_complex))
        last_phase = 2.0 * math.pi * thickness_m / (wave_speed / freq)
        m00p, m01p, m10p, m11p = _apply_transfer_matrix(m00p, m01p, m10p, m11p, complex(gamma_parallel), last_phase)
        m00s, m01s, m10s, m11s = _apply_transfer_matrix(
            m00s, m01s, m10s, m11s, complex(gamma_perpendicular), last_phase
        )
        previous_epsilon = epsilon_complex
        previous_mu = mu_complex
        previous_theta = float(theta_transmitted)

    gamma_parallel_exit, gamma_perpendicular_exit, _theta_exit, _tir = refl_coeff(
        previous_epsilon, previous_mu, 1.0 + 0j, 1.0 + 0j, previous_theta
    )
    m00p, m01p, m10p, m11p = _apply_transfer_matrix(m00p, m01p, m10p, m11p, complex(gamma_parallel_exit), last_phase)
    m00s, m01s, m10s, m11s = _apply_transfer_matrix(
        m00s, m01s, m10s, m11s, complex(gamma_perpendicular_exit), last_phase
    )
    reflection_parallel = m10p / m00p
    reflection_perpendicular = m10s / m00s
    return reflection_perpendicular, reflection_parallel


def refl_coeff_multi_layers_on_pec(
    thri: float,
    phrii: float,
    alpha: float,
    beta: float,
    freq: float,
    matrlLine: list,
    local_theta: float | None = None,
) -> tuple[complex, complex]:
    """Compute reflection coefficients for multi-layer stacks backed by PEC."""
    return refl_coeff_composite_layer_on_pec(
        thri=thri,
        phrii=phrii,
        alpha=alpha,
        beta=beta,
        freq=freq,
        matrlLine=matrlLine,
        local_theta=local_theta,
    )


def get_reflection_coeff_from_material(
    thri: float,
    phrii: float,
    alpha: float,
    beta: float,
    freq: float,
    matrlLine: list,
    local_theta: float | None = None,
) -> tuple[complex, complex]:
    """Dispatch reflection calculation based on a single facet material entry."""
    reflection_perpendicular: complex = 0.0 + 0.0j
    reflection_parallel: complex = 0.0 + 0.0j

    if matrlLine[MaterialEntryIndex.TYPE] == MaterialType.PEC.value:
        reflection_perpendicular = -1.0 + 0.0j
        reflection_parallel = -1.0 + 0.0j
    elif matrlLine[MaterialEntryIndex.TYPE] == MaterialType.COMPOSITE.value:
        reflection_perpendicular, reflection_parallel = refl_coeff_composite(
            thri, phrii, alpha, beta, freq, matrlLine, local_theta=local_theta
        )
    elif matrlLine[MaterialEntryIndex.TYPE] == MaterialType.COMPOSITE_ON_PEC.value:
        reflection_perpendicular, reflection_parallel = refl_coeff_composite_layer_on_pec(
            thri, phrii, alpha, beta, freq, matrlLine, local_theta=local_theta
        )
    elif matrlLine[MaterialEntryIndex.TYPE] == MaterialType.MULTI_LAYER.value:
        reflection_perpendicular, reflection_parallel = refl_coeff_multi_layers(
            thri, phrii, alpha, beta, freq, matrlLine, local_theta=local_theta
        )
    elif matrlLine[MaterialEntryIndex.TYPE] == MaterialType.MULTI_LAYER_ON_PEC.value:
        reflection_perpendicular, reflection_parallel = refl_coeff_multi_layers_on_pec(
            thri, phrii, alpha, beta, freq, matrlLine, local_theta=local_theta
        )
    else:
        material_type = str(matrlLine[MaterialEntryIndex.TYPE])
        raise ValueError(f"Unsupported material type: {material_type}")
    return reflection_perpendicular, reflection_parallel


def reflection_coefficients(
    rs: float,
    index: int,
    th2: float,
    thri: float,
    phrii: float,
    alpha: float,
    beta: float,
    freq: float,
    matrl: list,
    local_cos_theta: float | None = None,
) -> tuple[complex, complex]:
    """Return facet reflection coefficients for resistive or material-specific modes."""
    reflection_perpendicular: complex = 0.0 + 0.0j
    reflection_parallel: complex = 0.0 + 0.0j
    if rs == SPECIFIC_MATERIAL:
        reflection_perpendicular, reflection_parallel = get_reflection_coeff_from_material(
            thri, phrii, alpha, beta, freq, matrl[index], local_theta=th2
        )
    else:
        cosine_local_theta = math.cos(th2) if local_cos_theta is None else local_cos_theta
        reflection_perpendicular = complex(-1 / (2 * rs * cosine_local_theta + 1))
        reflection_parallel = 0.0 + 0.0j
        if (2 * rs + cosine_local_theta) != 0:
            reflection_parallel = complex(-cosine_local_theta / (2 * rs + cosine_local_theta))
    return reflection_perpendicular, reflection_parallel
