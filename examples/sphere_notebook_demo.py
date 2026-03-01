"""Notebook-friendly monostatic sphere example for Open RCS.

Usage in Jupyter:
    %run examples/sphere_notebook_demo.py
"""

from __future__ import annotations

from pathlib import Path

from open_rcs import (
    AngleSweep,
    MaterialConfig,
    MonostaticSimulationConfig,
    build_geometry_from_stl,
    run_monostatic,
)


def run_sphere_example(project_root: str | Path = "."):
    """Run a monostatic sphere example and return the generated result paths."""
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
            phi_stop_deg=0.0,
            phi_step_deg=0.0,
            theta_start_deg=0.0,
            theta_stop_deg=360.0,
            theta_step_deg=2.0,
        ),
        material=MaterialConfig(
            resistivity_mode=0,
            material_path=str(root / "matrl.txt"),
        ),
    )

    geometry_data = build_geometry_from_stl(model_path, simulation_config.material.resistivity_mode)
    result = run_monostatic(simulation_config, geometry_data)

    print("Plot:", result.plot_path)
    print("Mesh Figure:", result.figure_path)
    print("Data File:", result.data_path)

    try:
        from IPython.display import Image, display

        display(Image(filename=result.plot_path))
        display(Image(filename=result.figure_path))
    except Exception:
        pass

    return result


if __name__ == "__main__":
    run_sphere_example()
