"""Open RCS computational package."""

from .model_types import (
    AngleSweep,
    BistaticSimulationConfig,
    FieldAccumulator,
    GeometryData,
    MaterialConfig,
    MonostaticSimulationConfig,
    RcsComputationResult,
    SolverResult,
)
from .notebook_ui import build_plotly_figures, launch_rcs_widget
from .rcs_bistatic import run_bistatic, simulate_bistatic
from .rcs_functions import build_geometry_from_stl, extract_coordinates_data, get_params_from_file
from .rcs_monostatic import run_monostatic, simulate_monostatic
from .stl_module import convert_stl

__all__ = [
    "AngleSweep",
    "MaterialConfig",
    "MonostaticSimulationConfig",
    "BistaticSimulationConfig",
    "GeometryData",
    "FieldAccumulator",
    "RcsComputationResult",
    "SolverResult",
    "convert_stl",
    "extract_coordinates_data",
    "build_geometry_from_stl",
    "get_params_from_file",
    "simulate_monostatic",
    "simulate_bistatic",
    "run_monostatic",
    "run_bistatic",
    "build_plotly_figures",
    "launch_rcs_widget",
]
