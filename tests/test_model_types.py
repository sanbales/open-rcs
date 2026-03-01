"""Tests for typed model configuration and result containers."""

from __future__ import annotations

from open_rcs.constants import RadarBand


def test_radar_band_properties_and_string_list() -> None:
    """Verify radar-band frequency properties and formatted labels."""
    band = RadarBand.X
    assert band.min_freq == 8.0
    assert band.center_freq == 10.0
    assert band.max_freq == 12.0

    labels = RadarBand.to_string_list()
    assert any(label.startswith("X: 8.0-12.0 GHz") for label in labels)
    assert len(labels) == len(RadarBand)
