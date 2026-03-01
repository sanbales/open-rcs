"""STL utilities for Open RCS.

This module converts an STL mesh into the text-based coordinates/facets format
consumed by the RCS solvers.
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path

import numpy as np
from stl import mesh

PathStr = str | Path | PathLike[str]


def convert_stl(
    file_path: PathStr,
    *,
    coordinates_output: PathStr | None = "coordinates.txt",
    facets_output: PathStr | None = "facets.txt",
) -> tuple[np.ndarray, np.ndarray]:
    """Convert an STL file into coordinate and facet arrays.

    Parameters
    ----------
    file_path:
        STL input path.
    coordinates_output:
        Optional output file for coordinates. Pass ``None`` to skip writing.
    facets_output:
        Optional output file for facets. Pass ``None`` to skip writing.

    Returns:
    -------
    tuple[np.ndarray, np.ndarray]
        ``(coordinates, facets)`` where coordinates has shape ``(N, 3)`` and
        facets has shape ``(M, 6)`` as
        ``[facet_id, v1, v2, v3, ilum_flag, rs]``.

    """
    stl_mesh = mesh.Mesh.from_file(str(file_path))
    faces = np.asarray(stl_mesh.vectors, dtype=np.float64)
    flat_vertices = faces.reshape(-1, 3)

    coordinates, inverse = np.unique(flat_vertices, axis=0, return_inverse=True)
    vertex_indices = inverse.reshape(-1, 3) + 1  # one-based indexing for solver compatibility

    facet_ids = np.arange(1, len(faces) + 1, dtype=np.int64).reshape(-1, 1)
    ilum_flags = np.ones((len(faces), 1), dtype=np.int64)
    rs_values = np.zeros((len(faces), 1), dtype=np.int64)
    facets = np.hstack((facet_ids, vertex_indices.astype(np.int64), ilum_flags, rs_values))

    if coordinates_output is not None:
        np.savetxt(Path(coordinates_output), coordinates, fmt="%f", delimiter=" ")
    if facets_output is not None:
        np.savetxt(Path(facets_output), facets, fmt="%d", delimiter=" ")

    return coordinates, facets
