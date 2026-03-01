"""Notebook-friendly in-memory sphere demo (no image files written).

Usage in Jupyter:
    %run examples/sphere_inmemory_notebook_demo.py
"""

from __future__ import annotations

from pathlib import Path

from open_rcs import (
    AngleSweep,
    MaterialConfig,
    MonostaticSimulationConfig,
    build_geometry_from_stl,
    build_plotly_figures,
    simulate_monostatic,
)


def run_sphere_inmemory_demo(project_root: str | Path = "."):
    """Run the in-memory sphere demo and return simulation data plus Plotly figures."""
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
            phi_stop_deg=360.0,
            phi_step_deg=5.0,
            theta_start_deg=0.0,
            theta_stop_deg=180.0,
            theta_step_deg=5.0,
        ),
        material=MaterialConfig(resistivity_mode=0.0, material_path=str(root / "matrl.txt")),
    )

    geometry_data = build_geometry_from_stl(model_path, simulation_config.material.resistivity_mode)
    simulation_result = simulate_monostatic(simulation_config, geometry_data)

    fig_2d, fig_3d = build_plotly_figures(
        simulation_result,
        model_path,
        chart_mode="xy",
        component="max",
        radial_scale=1.0,
        rcs_opacity=0.55,
    )
    fig_2d.show()
    fig_3d.show()
    return simulation_result, fig_2d, fig_3d


if __name__ == "__main__":
    run_sphere_inmemory_demo()
