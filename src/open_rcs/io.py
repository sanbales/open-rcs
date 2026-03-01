"""Legacy input-file parsing helpers for building typed simulation configs."""

from enum import IntEnum
from pathlib import Path

from .model_types import (
    AngleSweep,
    BistaticSimulationConfig,
    MaterialConfig,
    MonostaticSimulationConfig,
)
from .rcs_functions import SPECIFIC_MATERIAL
from .stl_module import convert_stl


class LegacyInputIndex(IntEnum):
    """Column indices for legacy ``input_data_file_*.dat`` parameter lists."""

    INPUT_MODEL = 0
    FREQUENCY_GHZ = 1
    RESISTIVITY_MODE = 5
    PHI_START = 6
    PHI_STOP = 7
    PHI_STEP = 8
    THETA_START = 9
    THETA_STOP = 10
    THETA_STEP = 11
    INCIDENT_THETA = 12
    INCIDENT_PHI = 13


def _parse_param(value: str) -> float | str:  # pragma: no cover
    value = value.strip()
    try:
        return float(value)
    except ValueError:
        return value


def get_params_from_file(  # pragma: no cover
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

    param_list[LegacyInputIndex.FREQUENCY_GHZ] = float(param_list[LegacyInputIndex.FREQUENCY_GHZ]) * 1e9
    convert_stl(Path("./stl_models") / str(param_list[LegacyInputIndex.INPUT_MODEL]))

    if int(param_list[LegacyInputIndex.RESISTIVITY_MODE]) != SPECIFIC_MATERIAL:
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
        resistivity_mode=int(param_list[LegacyInputIndex.RESISTIVITY_MODE]),
        material_path=str(param_list[-1]),
    )

    if method == "monostatic":
        return MonostaticSimulationConfig(
            input_model=str(param_list[LegacyInputIndex.INPUT_MODEL]),
            frequency_hz=float(param_list[LegacyInputIndex.FREQUENCY_GHZ]),
            correlation_distance_m=float(param_list[2]),
            standard_deviation_m=float(param_list[3]),
            incident_polarization=int(param_list[4]),
            angle_sweep=angle_sweep,
            material=material,
        )
    if method == "bistatic":
        return BistaticSimulationConfig(
            input_model=str(param_list[LegacyInputIndex.INPUT_MODEL]),
            frequency_hz=float(param_list[LegacyInputIndex.FREQUENCY_GHZ]),
            correlation_distance_m=float(param_list[2]),
            standard_deviation_m=float(param_list[3]),
            incident_polarization=int(param_list[4]),
            angle_sweep=angle_sweep,
            incident_theta_deg=float(param_list[12]),
            incident_phi_deg=float(param_list[13]),
            material=material,
        )
    raise ValueError("method must be 'monostatic' or 'bistatic'.")
