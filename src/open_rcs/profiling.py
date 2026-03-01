"""Profiling utilities for Open RCS simulations."""

from __future__ import annotations

import cProfile
import io
import pstats
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .model_types import BistaticSimulationConfig, GeometryData, MonostaticSimulationConfig
from .rcs_bistatic import simulate_bistatic
from .rcs_monostatic import simulate_monostatic

SortKey = Literal["cumtime", "tottime", "calls"]


@dataclass(slots=True)
class ProfileReport:
    """Structured profiling result for a simulation run."""

    mode: Literal["monostatic", "bistatic"]
    sort_by: SortKey
    top_n: int
    text: str
    output_path: str | None = None


def _format_profile(
    profiler: cProfile.Profile,
    sort_by: SortKey,
    top_n: int,
) -> str:
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs().sort_stats(sort_by).print_stats(top_n)
    return stream.getvalue()


def _write_report(report_text: str, output_path: str | Path) -> str:
    """Write profile text to disk, creating parent directories when needed."""
    output_target = Path(output_path)
    output_target.parent.mkdir(parents=True, exist_ok=True)
    output_target.write_text(report_text, encoding="utf-8")
    return str(output_target)


def profile_monostatic(
    simulation_config: MonostaticSimulationConfig,
    geometry_data: GeometryData,
    *,
    sort_by: SortKey = "cumtime",
    top_n: int = 40,
    output_path: str | Path | None = None,
) -> ProfileReport:
    """Profile monostatic simulation and return a formatted report."""
    profiler = cProfile.Profile()
    profiler.enable()
    simulate_monostatic(simulation_config, geometry_data)
    profiler.disable()

    report_text = _format_profile(profiler, sort_by, top_n)
    final_output_path: str | None = None
    if output_path is not None:
        final_output_path = _write_report(report_text, output_path)

    return ProfileReport(
        mode="monostatic",
        sort_by=sort_by,
        top_n=top_n,
        text=report_text,
        output_path=final_output_path,
    )


def profile_bistatic(
    simulation_config: BistaticSimulationConfig,
    geometry_data: GeometryData,
    *,
    sort_by: SortKey = "cumtime",
    top_n: int = 40,
    output_path: str | Path | None = None,
) -> ProfileReport:
    """Profile bistatic simulation and return a formatted report."""
    profiler = cProfile.Profile()
    profiler.enable()
    simulate_bistatic(simulation_config, geometry_data)
    profiler.disable()

    report_text = _format_profile(profiler, sort_by, top_n)
    final_output_path: str | None = None
    if output_path is not None:
        final_output_path = _write_report(report_text, output_path)

    return ProfileReport(
        mode="bistatic",
        sort_by=sort_by,
        top_n=top_n,
        text=report_text,
        output_path=final_output_path,
    )
