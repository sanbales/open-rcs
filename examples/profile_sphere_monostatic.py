"""Profile monostatic sphere simulation and print top hotspots.

Usage in Jupyter:
    %run examples/profile_sphere_monostatic.py
"""

from __future__ import annotations

from pathlib import Path

from open_rcs import (
    AngleSweep,
    MaterialConfig,
    MonostaticSimulationConfig,
    build_geometry_from_stl,
    profile_monostatic,
)


def run_profile(project_root: str | Path = ".") -> str:
    """Run a monostatic sphere profile and return the raw profiler report text."""
    root = Path(project_root).resolve()
    model_path = root / "stl_models" / "sphere.stl"

    simulation_config = MonostaticSimulationConfig(
        input_model="sphere.stl",
        frequency_hz=130e9,
        correlation_distance_m=0.0,
        standard_deviation_m=0.0,
        incident_polarization=0,
        angle_sweep=AngleSweep(
            phi_start_deg=0.0,
            phi_stop_deg=180.0,
            phi_step_deg=5.0,
            theta_start_deg=0.0,
            theta_stop_deg=180.0,
            theta_step_deg=5.0,
        ),
        material=MaterialConfig(resistivity_mode=0.0, material_path=str(root / "matrl.txt")),
    )
    geometry_data = build_geometry_from_stl(model_path, simulation_config.material.resistivity_mode)

    report = profile_monostatic(
        simulation_config,
        geometry_data,
        sort_by="cumtime",
        top_n=60,
        output_path=root / "results" / "profile_monostatic_sphere.txt",
    )
    print(report.text)
    if report.output_path:
        print(f"Profile saved to: {report.output_path}")
    return report.text


if __name__ == "__main__":
    run_profile()
