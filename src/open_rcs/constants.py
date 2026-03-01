from __future__ import annotations

from enum import Enum, IntEnum
from pathlib import Path
from typing import Final, TypeAlias


class RadarBand(Enum):
    """IEEE Standard Letter Designations for Radar-Frequency Bands (GHz)."""

    HF = (0.003, 0.03)
    VHF = (0.03, 0.3)
    UHF = (0.3, 1.0)
    L = (1.0, 2.0)
    S = (2.0, 4.0)
    C = (4.0, 8.0)
    X = (8.0, 12.0)
    KU = (12.0, 18.0)
    K = (18.0, 27.0)
    KA = (27.0, 40.0)
    V = (40.0, 75.0)
    W = (75.0, 110.0)

    @property
    def min_freq(self) -> float:
        return self.value[0]

    @property
    def center_freq(self) -> float:
        return 0.5 * sum(self.value)

    @property
    def max_freq(self) -> float:
        return self.value[1]

    @classmethod
    def to_string_list(cls) -> list[str]:
        """Returns a list of strings formatted as 'NAME: MIN-MAX GHz'."""
        return [f"{band.name}: {band.value[0]}-{band.value[1]} GHz" for band in cls]


class FontSize(IntEnum):
    SMALL = 8
    MEDIUM = 10
    LARGE = 12


class SphericalIndex(IntEnum):
    THETA = 1


class MaterialEntryIndex(IntEnum):
    TYPE = 0
    DESCRIPTION = 1
    FIRST_LAYER = 2


class MaterialType(str, Enum):
    PEC = "PEC"
    COMPOSITE = "Composite"
    COMPOSITE_ON_PEC = "Composite Layer on PEC"
    MULTI_LAYER = "Multiple Layers"
    MULTI_LAYER_ON_PEC = "Multiple Layers on PEC"


class MaterialCode(IntEnum):
    PEC = 0
    COMPOSITE = 1
    COMPOSITE_ON_PEC = 2
    MULTI_LAYER = 3
    MULTI_LAYER_ON_PEC = 4


SPECIFIC_MATERIAL: Final[float] = 1.0
MATERIAL_TYPE_TO_CODE: Final[dict[str, int]] = {
    MaterialType.PEC.value: int(MaterialCode.PEC),
    MaterialType.COMPOSITE.value: int(MaterialCode.COMPOSITE),
    MaterialType.COMPOSITE_ON_PEC.value: int(MaterialCode.COMPOSITE_ON_PEC),
    MaterialType.MULTI_LAYER.value: int(MaterialCode.MULTI_LAYER),
    MaterialType.MULTI_LAYER_ON_PEC.value: int(MaterialCode.MULTI_LAYER_ON_PEC),
}
RESULTS_DIR: Final[Path] = Path("./results")
MaterialLayer: TypeAlias = list[float]
MaterialEntry: TypeAlias = list[str | MaterialLayer]
MaterialTable: TypeAlias = list[MaterialEntry]
MaterialCatalog: TypeAlias = dict[str, MaterialEntry]
