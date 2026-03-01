from __future__ import annotations

from pathlib import Path

from open_rcs import (
    AngleSweep,
    BistaticSimulationConfig,
    MaterialConfig,
    MonostaticSimulationConfig,
    build_geometry_from_stl,
    profile_bistatic,
    profile_monostatic,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PLATE_STL = REPO_ROOT / "stl_models" / "plate.stl"
DUMMY_MATERIAL_PATH = str(REPO_ROOT / "matrl.txt")


def test_profile_monostatic_writes_report_file(tmp_path: Path) -> None:
    config = MonostaticSimulationConfig(
        input_model="plate.stl",
        frequency_hz=10.0e9,
        correlation_distance_m=0.0,
        standard_deviation_m=0.0,
        incident_polarization=0,
        angle_sweep=AngleSweep(
            phi_start_deg=0.0,
            phi_stop_deg=0.0,
            phi_step_deg=1.0,
            theta_start_deg=0.0,
            theta_stop_deg=0.0,
            theta_step_deg=1.0,
        ),
        material=MaterialConfig(
            resistivity_mode=0.2,
            material_path=DUMMY_MATERIAL_PATH,
        ),
        use_numba=False,
    )
    geometry_data = build_geometry_from_stl(PLATE_STL, config.material.resistivity_mode)
    output_path = tmp_path / "nested" / "profile_report.txt"

    report = profile_monostatic(
        config,
        geometry_data,
        sort_by="cumtime",
        top_n=20,
        output_path=output_path,
    )

    assert report.mode == "monostatic"
    assert report.output_path is not None
    assert output_path.exists()
    assert "simulate_monostatic" in report.text
    assert output_path.read_text(encoding="utf-8") == report.text


def test_profile_bistatic_without_output_path() -> None:
    config = BistaticSimulationConfig(
        input_model="plate.stl",
        frequency_hz=10.0e9,
        correlation_distance_m=0.0,
        standard_deviation_m=0.0,
        incident_polarization=0,
        angle_sweep=AngleSweep(
            phi_start_deg=0.0,
            phi_stop_deg=0.0,
            phi_step_deg=1.0,
            theta_start_deg=0.0,
            theta_stop_deg=0.0,
            theta_step_deg=1.0,
        ),
        incident_theta_deg=35.0,
        incident_phi_deg=40.0,
        material=MaterialConfig(
            resistivity_mode=0.2,
            material_path=DUMMY_MATERIAL_PATH,
        ),
        use_numba=False,
    )
    geometry_data = build_geometry_from_stl(PLATE_STL, config.material.resistivity_mode)
    report = profile_bistatic(config, geometry_data, sort_by="cumtime", top_n=20, output_path=None)

    assert report.mode == "bistatic"
    assert report.output_path is None
    assert "simulate_bistatic" in report.text


def test_profile_bistatic_writes_report_file(tmp_path: Path) -> None:
    config = BistaticSimulationConfig(
        input_model="plate.stl",
        frequency_hz=10.0e9,
        correlation_distance_m=0.0,
        standard_deviation_m=0.0,
        incident_polarization=0,
        angle_sweep=AngleSweep(
            phi_start_deg=0.0,
            phi_stop_deg=0.0,
            phi_step_deg=1.0,
            theta_start_deg=0.0,
            theta_stop_deg=0.0,
            theta_step_deg=1.0,
        ),
        incident_theta_deg=35.0,
        incident_phi_deg=40.0,
        material=MaterialConfig(
            resistivity_mode=0.2,
            material_path=DUMMY_MATERIAL_PATH,
        ),
        use_numba=False,
    )
    geometry_data = build_geometry_from_stl(PLATE_STL, config.material.resistivity_mode)
    output_path = tmp_path / "nested" / "profile_bistatic_report.txt"
    report = profile_bistatic(
        config,
        geometry_data,
        sort_by="cumtime",
        top_n=20,
        output_path=output_path,
    )

    assert report.mode == "bistatic"
    assert report.output_path is not None
    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8") == report.text
