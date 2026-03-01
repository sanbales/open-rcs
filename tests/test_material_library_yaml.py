from __future__ import annotations

from pathlib import Path

import pytest

from open_rcs import rcs_functions as rf

pytest.importorskip("yaml")


def test_rcsmat_yaml_assignments_with_tags_and_ranges(tmp_path: Path) -> None:
    material_path = tmp_path / "library.rcsmat"
    material_path.write_text(
        "\n".join(
            [
                "version: 1",
                "materials:",
                "  - id: pec",
                "    type: PEC",
                "    description: perfect conductor",
                "  - id: coat",
                "    type: Composite",
                "    description: coated skin",
                "    layers:",
                "      - epsilon_r: 2.5",
                "        loss_tangent: 0.02",
                "        mu_r_real: 1.0",
                "        mu_r_imag: 0.01",
                "        thickness_mm: 1.2",
                "tags:",
                "  nose: [1, 2]",
                "assignments:",
                "  - material: pec",
                "    facets: all",
                "  - material: coat",
                "    tags: [nose]",
                "  - material: coat",
                "    facet_ranges: [[5, 6]]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    material_table = rf.get_entries_from_material_file(6, str(material_path))
    material_catalog = rf.load_material_catalog(material_path)

    assert set(material_catalog.keys()) == {"pec", "coat"}
    assert len(material_table) == 6
    assert material_table[0][rf.TYPE] == rf.MATERIAL_TYPE_COMPOSITE
    assert material_table[1][rf.TYPE] == rf.MATERIAL_TYPE_COMPOSITE
    assert material_table[2][rf.TYPE] == rf.MATERIAL_TYPE_PEC
    assert material_table[3][rf.TYPE] == rf.MATERIAL_TYPE_PEC
    assert material_table[4][rf.TYPE] == rf.MATERIAL_TYPE_COMPOSITE
    assert material_table[5][rf.TYPE] == rf.MATERIAL_TYPE_COMPOSITE


def test_rcsmat_yaml_reports_invalid_tag_reference(tmp_path: Path) -> None:
    material_path = tmp_path / "invalid.rcsmat"
    material_path.write_text(
        "\n".join(
            [
                "version: 1",
                "materials:",
                "  - id: pec",
                "    type: PEC",
                "    description: perfect conductor",
                "assignments:",
                "  - material: pec",
                "    tags: [missing_tag]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        rf.get_entries_from_material_file(4, str(material_path))
