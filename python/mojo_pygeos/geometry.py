"""A compact, vectorised subset of the :mod:`pygeos` public API.

Geometry objects hold planar float64 coordinates.  Arrays are ordinary NumPy
object arrays so NumPy broadcasting has the same useful shape rules as pygeos.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable
import weakref

import numpy as np

from ._lib import addr, f64, i64, lib

POINT, LINESTRING, LINEARRING, POLYGON = 0, 1, 2, 3


_PACK_CACHE: dict[int, tuple[weakref.ReferenceType, tuple[np.ndarray, ...]]] = {}


class _GeometryArray(np.ndarray):
    def __array_finalize__(self, source):
        self._point_coords = getattr(source, "_point_coords", None)
        self._point_x = getattr(source, "_point_x", None)
        self._point_y = getattr(source, "_point_y", None)


@dataclass(frozen=True)
class Geometry:
    """One two-dimensional point, line string, linear ring, or polygon."""

    type_id: int
    coords: np.ndarray
    rings: tuple[int, ...] = ()

    def __post_init__(self):
        coords = f64(self.coords)
        if coords.ndim != 2 or coords.shape[1] != 2 or not len(coords):
            raise ValueError("coordinates must be a non-empty (n, 2) array")
        if self.type_id not in (POINT, LINESTRING, LINEARRING, POLYGON):
            raise ValueError("unknown geometry type")
        if self.type_id == POINT and len(coords) != 1:
            raise ValueError("a point needs exactly one coordinate")
        if self.type_id in (LINESTRING, LINEARRING) and len(coords) < 2:
            raise ValueError("a line needs at least two coordinates")
        if self.type_id == LINEARRING and (len(coords) < 4 or not np.array_equal(coords[0], coords[-1])):
            raise ValueError("a linear ring must be closed and have at least three vertices")
        if self.type_id == POLYGON:
            if len(self.rings) < 2 or self.rings[0] != 0 or self.rings[-1] != len(coords):
                raise ValueError("polygon ring offsets must span its coordinates")
            if any(start >= end for start, end in zip(self.rings, self.rings[1:])):
                raise ValueError("polygon rings must be non-empty")
            for start, end in zip(self.rings, self.rings[1:]):
                ring = coords[start:end]
                if len(ring) < 4 or not np.array_equal(ring[0], ring[-1]):
                    raise ValueError("polygon rings must be closed and have at least three vertices")
        object.__setattr__(self, "coords", coords)

    @property
    def geom_type(self) -> str:
        return ("POINT", "LINESTRING", "LINEARRING", "POLYGON")[self.type_id]

    def __repr__(self) -> str:
        return to_wkt(self)

    @classmethod
    def _point_view(cls, xy: np.ndarray):
        geometry = object.__new__(cls)
        object.__setattr__(geometry, "type_id", POINT)
        object.__setattr__(geometry, "coords", xy.reshape(1, 2))
        object.__setattr__(geometry, "rings", ())
        return geometry


def _ring(coords) -> np.ndarray:
    a = f64(coords).reshape(-1, 2)
    if len(a) < 3:
        raise ValueError("a ring needs at least three coordinates")
    return a if np.array_equal(a[0], a[-1]) else np.vstack((a, a[0]))


def _object(values, shape, point_coords=None, point_x=None, point_y=None):
    out = np.empty(len(values), dtype=object).view(_GeometryArray)
    out[:] = values
    out = out.reshape(shape)
    out._point_coords = point_coords
    out._point_x = point_x
    out._point_y = point_y
    return out


def points(coords, y=None, z=None, indices=None, out=None, **kwargs):
    """Create point geometries, matching ``pygeos.points`` for 2-D input."""
    if out is not None or kwargs:
        raise NotImplementedError("out and additional constructor options are not supported")
    if z is not None:
        raise NotImplementedError("this planar port does not accept z coordinates")
    if y is not None:
        x, yv = np.broadcast_arrays(f64(coords), f64(y))
        point_coords = np.ascontiguousarray(np.stack((x, yv), axis=-1))
        return _object([Geometry._point_view(xy) for xy in point_coords.reshape(-1, 2)], x.shape,
                       point_coords, np.ascontiguousarray(point_coords[..., 0]), np.ascontiguousarray(point_coords[..., 1]))
    a = f64(coords)
    if a.shape[-1] != 2:
        raise ValueError("point coordinates must end in (x, y)")
    if indices is not None:
        indices = i64(indices).reshape(-1)
        if len(indices) != len(a):
            raise ValueError("indices must have one entry per coordinate")
        result = np.empty(int(indices.max()) + 1, dtype=object)
        result[:] = None
        for index, xy in zip(indices, a.reshape(-1, 2)):
            result[index] = Geometry(POINT, xy.reshape(1, 2))
        return result
    if a.ndim == 1:
        return Geometry(POINT, a.reshape(1, 2))
    return _object([Geometry._point_view(xy) for xy in a.reshape(-1, 2)], a.shape[:-1], a,
                   np.ascontiguousarray(a[..., 0]), np.ascontiguousarray(a[..., 1]))


def linestrings(coords, y=None, z=None, indices=None, out=None, **kwargs):
    """Create line strings from ``(n, 2)`` or ``(..., n, 2)`` coordinates."""
    if out is not None or kwargs:
        raise NotImplementedError("out and additional constructor options are not supported")
    if y is not None or z is not None or indices is not None:
        raise NotImplementedError("use packed (..., n, 2) coordinates for line strings")
    a = f64(coords)
    if a.ndim == 2:
        return Geometry(LINESTRING, a)
    if a.ndim < 3 or a.shape[-1] != 2:
        raise ValueError("line coordinates must have shape (..., n, 2)")
    return _object([Geometry(LINESTRING, part) for part in a.reshape(-1, a.shape[-2], 2)], a.shape[:-2])


def linearrings(coords, y=None, z=None, indices=None, out=None, **kwargs):
    if out is not None or kwargs:
        raise NotImplementedError("out and additional constructor options are not supported")
    if y is not None or z is not None or indices is not None:
        raise NotImplementedError("use packed (..., n, 2) coordinates for linear rings")
    a = f64(coords)
    if a.ndim == 2:
        return Geometry(LINEARRING, _ring(a))
    if a.ndim < 3 or a.shape[-1] != 2:
        raise ValueError("ring coordinates must have shape (..., n, 2)")
    return _object([Geometry(LINEARRING, _ring(part)) for part in a.reshape(-1, a.shape[-2], 2)], a.shape[:-2])


def polygons(geometries, holes=None, indices=None, out=None, **kwargs):
    """Create polygons from an exterior ring and optional interior rings."""
    if out is not None or kwargs:
        raise NotImplementedError("out and additional constructor options are not supported")
    if indices is not None:
        raise NotImplementedError("indexed polygon construction is not covered")
    if isinstance(geometries, Geometry):
        exterior = _ring(geometries.coords)
    else:
        a = f64(geometries)
        if a.ndim != 2 or a.shape[1] != 2:
            raise ValueError("polygon exterior must be a linear ring or (n, 2) coordinates")
        exterior = _ring(a)
    parts = [exterior]
    if holes is not None:
        if isinstance(holes, Geometry):
            hole_values = [holes]
        else:
            h = np.asarray(holes)
            hole_values = list(h) if h.ndim == 3 else list(np.asarray(holes, dtype=object).flat)
        for hole in hole_values:
            parts.append(_ring(hole.coords if isinstance(hole, Geometry) else hole))
    coords = np.ascontiguousarray(np.vstack(parts), dtype=np.float64)
    return Geometry(POLYGON, coords, tuple(np.cumsum([0] + [len(p) for p in parts]).tolist()))


def box(xmin, ymin, xmax, ymax, ccw=True, **kwargs):
    if kwargs:
        raise NotImplementedError("additional box options are not supported")
    # Match pygeos' canonical start vertex as well as its orientation.
    pts = np.array([[xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]])
    if not ccw:
        pts = pts[::-1]
    return polygons(pts)


def _geometries(value):
    if isinstance(value, Geometry):
        return np.array([value], dtype=object), ()
    a = value if isinstance(value, _GeometryArray) else np.asarray(value, dtype=object)
    if not a.size:
        raise ValueError("geometry arrays must be non-empty")
    if not all(isinstance(g, Geometry) for g in a.flat):
        raise TypeError("expected a Geometry or an array of Geometry objects")
    return a.reshape(-1), a.shape


def _pack(flat: Iterable[Geometry]):
    coords, coord_offsets, ring_offsets, geom_rings, kinds = [], [0], [0], [0], []
    for g in flat:
        coords.append(g.coords)
        coord_offsets.append(coord_offsets[-1] + len(g.coords))
        kinds.append(POINT if g.type_id == POINT else LINESTRING if g.type_id in (LINESTRING, LINEARRING) else POLYGON)
        if g.type_id == POLYGON:
            ends = g.rings[1:]
        else:
            ends = (len(g.coords),)
        base = coord_offsets[-2]
        for end in ends:
            ring_offsets.append(base + end)
        geom_rings.append(len(ring_offsets) - 1)
    return (np.ascontiguousarray(np.vstack(coords), dtype=np.float64), i64(coord_offsets),
            i64(ring_offsets), i64(geom_rings), i64(kinds))


def _cached_pack(value, flat):
    if isinstance(value, Geometry):
        return _pack(flat)
    key = id(value)
    cached = _PACK_CACHE.get(key)
    if cached is not None and cached[0]() is value:
        return cached[1]
    packed = _pack(flat)

    def discard(_):
        _PACK_CACHE.pop(key, None)

    _PACK_CACHE[key] = (weakref.ref(value, discard), packed)
    return packed


def _metrics(value):
    flat, shape = _geometries(value)
    packed = _cached_pack(value, flat)
    n = len(flat)
    areas, lengths = np.empty(n), np.empty(n)
    b = np.empty((n, 4))
    c = np.empty((n, 2))
    lib().mpg_metrics(*(addr(x) for x in packed), n, addr(areas), addr(lengths), addr(b), addr(c))
    return shape, areas, lengths, b, c


def _reshape(values, shape):
    return values[0] if not shape else values.reshape(shape)


def area(geometry, **kwargs):
    shape, values, _, _, _ = _metrics(geometry)
    return _reshape(values, shape)


def length(geometry, **kwargs):
    shape, _, values, _, _ = _metrics(geometry)
    return _reshape(values, shape)


def bounds(geometry, **kwargs):
    shape, _, _, values, _ = _metrics(geometry)
    return values[0] if not shape else values.reshape(shape + (4,))


def centroid(geometry, **kwargs):
    shape, _, _, _, values = _metrics(geometry)
    result = [Geometry._point_view(xy) for xy in values]
    return result[0] if not shape else _object(result, shape)


def get_type_id(geometry, **kwargs):
    flat, shape = _geometries(geometry)
    return _reshape(np.array([g.type_id for g in flat], dtype=np.int64), shape)


def get_num_coordinates(geometry, **kwargs):
    flat, shape = _geometries(geometry)
    return _reshape(np.array([len(g.coords) for g in flat], dtype=np.int64), shape)


def get_coordinates(geometry, include_z=False, return_index=False):
    if include_z:
        raise NotImplementedError("this planar port has no z dimension")
    flat, _ = _geometries(geometry)
    coordinates = np.vstack([g.coords for g in flat])
    if return_index:
        return coordinates, np.repeat(np.arange(len(flat)), [len(g.coords) for g in flat])
    return coordinates


def get_x(point, **kwargs):
    flat, shape = _geometries(point)
    if any(g.type_id != POINT for g in flat):
        raise TypeError("get_x requires point geometries")
    return _reshape(np.array([g.coords[0, 0] for g in flat]), shape)


def get_y(point, **kwargs):
    flat, shape = _geometries(point)
    if any(g.type_id != POINT for g in flat):
        raise TypeError("get_y requires point geometries")
    return _reshape(np.array([g.coords[0, 1] for g in flat]), shape)


def _broadcast(a, b):
    aa, ashape = _geometries(a)
    bb, bshape = _geometries(b)
    aa = np.asarray(aa, dtype=object).reshape(ashape or ())
    bb = np.asarray(bb, dtype=object).reshape(bshape or ())
    x, y = np.broadcast_arrays(aa, bb)
    return x.reshape(-1), y.reshape(-1), x.shape


def _point_relation(polygons_, points_):
    packed = _pack(polygons_)
    xy = np.ascontiguousarray(np.vstack([p.coords[0] for p in points_]))
    xs = np.ascontiguousarray(xy[:, 0])
    ys = np.ascontiguousarray(xy[:, 1])
    result = np.empty(len(points_), dtype=np.int64)
    lib().mpg_point_relation(addr(packed[0]), addr(packed[2]), addr(packed[3]), addr(packed[4]),
                             len(points_), addr(xs), addr(ys), addr(result))
    return result


def _relations(a, b, mode):
    aa, bb, shape = _broadcast(a, b)
    result = np.zeros(len(aa), dtype=bool)
    poly_point = np.array([x.type_id == POLYGON and y.type_id == POINT for x, y in zip(aa, bb)])
    if np.any(poly_point):
        relation = _point_relation(aa[poly_point], bb[poly_point])
        result[poly_point] = relation == (2 if mode == "contains" else 1) if mode == "contains" else relation != 0
    point_poly = np.array([x.type_id == POINT and y.type_id == POLYGON for x, y in zip(aa, bb)])
    if np.any(point_poly):
        relation = _point_relation(bb[point_poly], aa[point_poly])
        result[point_poly] = relation == 2 if mode == "within" else relation != 0
    point_point = np.array([x.type_id == POINT and y.type_id == POINT for x, y in zip(aa, bb)])
    if np.any(point_point):
        result[point_point] = [np.array_equal(x.coords, y.coords) for x, y in zip(aa[point_point], bb[point_point])]
    if np.any(~(poly_point | point_poly | point_point)):
        raise NotImplementedError("relations currently cover point/polygon and point/point pairs")
    return result.item() if not shape else result.reshape(shape)


def contains(a, b, **kwargs):
    return _relations(a, b, "contains")


def covers(a, b, **kwargs):
    return _relations(a, b, "covers")


def within(a, b, **kwargs):
    return _relations(a, b, "within")


def intersects(a, b, **kwargs):
    return _relations(a, b, "intersects")


def distance(a, b, **kwargs):
    if isinstance(a, _GeometryArray) and a._point_x is not None and isinstance(b, Geometry) and b.type_id == POINT:
        values = np.empty(a.size, dtype=np.float64)
        lib().mpg_point_distance_scalar(addr(a._point_x.reshape(-1)), addr(a._point_y.reshape(-1)), values.size,
                                        b.coords[0, 0], b.coords[0, 1], addr(values))
        return values.reshape(a.shape)
    if isinstance(b, _GeometryArray) and b._point_x is not None and isinstance(a, Geometry) and a.type_id == POINT:
        values = np.empty(b.size, dtype=np.float64)
        lib().mpg_point_distance_scalar(addr(b._point_x.reshape(-1)), addr(b._point_y.reshape(-1)), values.size,
                                        a.coords[0, 0], a.coords[0, 1], addr(values))
        return values.reshape(b.shape)
    a_coords = a.coords.reshape(2) if isinstance(a, Geometry) and a.type_id == POINT else getattr(a, "_point_coords", None)
    b_coords = b.coords.reshape(2) if isinstance(b, Geometry) and b.type_id == POINT else getattr(b, "_point_coords", None)
    if a_coords is not None and b_coords is not None:
        try:
            left, right = np.broadcast_arrays(a_coords, b_coords)
        except ValueError:
            pass
        else:
            if left.shape[-1] == 2:
                values = np.hypot(left[..., 0] - right[..., 0], left[..., 1] - right[..., 1])
                return values.item() if values.ndim == 0 else values
    aa, bb, shape = _broadcast(a, b)
    if any(x.type_id != POINT or y.type_id != POINT for x, y in zip(aa, bb)):
        raise NotImplementedError("distance currently covers point/point pairs")
    values = np.hypot(*np.transpose(np.vstack([x.coords[0] - y.coords[0] for x, y in zip(aa, bb)])))
    return values.item() if not shape else values.reshape(shape)


def to_wkt(geometry, rounding_precision=6, trim=True, output_dimension=3, old_3d=False, **kwargs):
    flat, shape = _geometries(geometry)
    def fmt(v):
        text = f"{v:.{rounding_precision}f}"
        return text.rstrip("0").rstrip(".") if trim and "." in text else text
    def one(g):
        coords = lambda a: ", ".join(f"{fmt(x)} {fmt(y)}" for x, y in a)
        if g.type_id == POINT:
            return f"POINT ({coords(g.coords)})"
        if g.type_id in (LINESTRING, LINEARRING):
            name = "LINEARRING" if g.type_id == LINEARRING else "LINESTRING"
            return f"{name} ({coords(g.coords)})"
        return "POLYGON (" + ", ".join("(" + coords(g.coords[s:e]) + ")" for s, e in zip(g.rings[:-1], g.rings[1:])) + ")"
    values = np.array([one(g) for g in flat], dtype=object)
    return values[0] if not shape else values.reshape(shape)


def from_wkt(geometry, on_invalid="raise", **kwargs):
    def parse(text):
        text = str(text).strip()
        pairs = lambda s: np.array([[float(v) for v in p.split()[:2]] for p in s.split(",")])
        if text.upper().startswith("POINT"):
            return points(pairs(re.search(r"\((.*)\)", text).group(1))[0])
        if text.upper().startswith("LINESTRING"):
            return linestrings(pairs(re.search(r"\((.*)\)", text).group(1)))
        if text.upper().startswith("POLYGON"):
            rings = re.findall(r"\(([^()]+)\)", text)
            return polygons(pairs(rings[0]), holes=[pairs(r) for r in rings[1:]])
        raise ValueError("only POINT, LINESTRING, and POLYGON WKT are covered")
    a = np.asarray(geometry, dtype=object)
    try:
        values = [parse(v) for v in a.flat]
    except ValueError:
        if on_invalid == "ignore":
            return None
        raise
    return values[0] if a.ndim == 0 else _object(values, a.shape)
