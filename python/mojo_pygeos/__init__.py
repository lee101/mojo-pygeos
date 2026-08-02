"""Mojo-backed vectorised planar geometry, compatible with a pygeos subset."""

from .geometry import (
    Geometry, area, bounds, box, centroid, contains, covers, distance, from_wkt,
    get_coordinates, get_num_coordinates, get_type_id, get_x, get_y, intersects,
    linearrings, linestrings, length, points, polygons, to_wkt, within,
)

__all__ = [name for name in globals() if not name.startswith("_")]
