"""Tests for STL mesh conversion and geometry loading helpers."""

from __future__ import annotations

from pathlib import Path

from open_rcs.stl_module import convert_stl

REPO_ROOT = Path(__file__).resolve().parents[1]
PLATE_STL = REPO_ROOT / "stl_models" / "plate.stl"


def test_convert_stl_writes_coordinates_and_facets_files(tmp_path: Path) -> None:
    """Verify STL conversion writes output files and returns expected array shapes."""
    coordinates_path = tmp_path / "coordinates.txt"
    facets_path = tmp_path / "facets.txt"

    coordinates, facets = convert_stl(
        PLATE_STL,
        coordinates_output=coordinates_path,
        facets_output=facets_path,
    )

    assert coordinates_path.exists()
    assert facets_path.exists()
    assert coordinates.shape[1] == 3
    assert facets.shape[1] == 6
