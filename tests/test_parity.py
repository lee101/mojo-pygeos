"""Numerical and behavioural parity with the upstream pygeos 0.14 package."""

import numpy as np
import pytest

import mojo_pygeos as mpg
import pygeos
from mojo_pygeos.geometry import _PACK_CACHE


def both(kind, coords):
    return getattr(mpg, kind)(coords), getattr(pygeos, kind)(coords)


@pytest.mark.parametrize("kind,coords", [
    ("points", [2.0, -1.0]),
    ("linestrings", [[0, 0], [3, 4], [6, 4]]),
    ("linearrings", [[0, 0], [4, 0], [4, 3], [0, 3]]),
    ("polygons", [[0, 0], [4, 0], [4, 3], [0, 3]]),
])
def test_scalar_measurements_match_pygeos(kind, coords):
    ours, theirs = both(kind, coords)
    assert mpg.area(ours) == pytest.approx(pygeos.area(theirs))
    assert mpg.length(ours) == pytest.approx(pygeos.length(theirs))
    assert np.allclose(mpg.bounds(ours), pygeos.bounds(theirs))
    assert np.allclose(mpg.get_coordinates(mpg.centroid(ours)), pygeos.get_coordinates(pygeos.centroid(theirs)))
    assert mpg.get_type_id(ours) == pygeos.get_type_id(theirs)
    assert mpg.get_num_coordinates(ours) == pygeos.get_num_coordinates(theirs)


def test_polygon_with_hole_matches_pygeos():
    shell = [[0, 0], [10, 0], [10, 10], [0, 10]]
    hole = [[1, 1], [3, 1], [3, 3], [1, 3]]
    ours = mpg.polygons(shell, holes=[hole])
    theirs = pygeos.polygons(shell, holes=[hole])
    assert mpg.area(ours) == pytest.approx(pygeos.area(theirs))
    assert mpg.length(ours) == pytest.approx(pygeos.length(theirs))
    assert np.allclose(mpg.get_coordinates(mpg.centroid(ours)), pygeos.get_coordinates(pygeos.centroid(theirs)))
    assert np.allclose(mpg.bounds(ours), pygeos.bounds(theirs))


def test_vectorized_metrics_match_pygeos():
    coords = np.array([
        [[0, 0], [2, 0], [2, 1], [0, 1]],
        [[-1, -2], [3, -2], [3, 2], [-1, 2]],
        [[4, 1], [5, 1], [5, 5], [4, 5]],
    ], dtype=float)
    ours, theirs = mpg.polygons(coords[0]), pygeos.polygons(coords[0])
    # Upstream does not accept a stacked coordinate array for polygons; make
    # the equivalent vector explicitly and test the vectorized operations.
    ourv = np.array([mpg.polygons(c) for c in coords], dtype=object)
    theirv = np.array([pygeos.polygons(c) for c in coords], dtype=object)
    assert np.allclose(mpg.area(ourv), pygeos.area(theirv))
    assert np.allclose(mpg.length(ourv), pygeos.length(theirv))
    assert np.allclose(mpg.bounds(ourv), pygeos.bounds(theirv))
    assert np.allclose(mpg.get_coordinates(mpg.centroid(ourv)), pygeos.get_coordinates(pygeos.centroid(theirv)))
    assert mpg.area(ours) == pytest.approx(pygeos.area(theirs))


def test_coordinate_accessors_and_broadcasted_point_distance_match_pygeos():
    ours = mpg.points([[0, 0], [3, 4], [-2, 1]])
    theirs = pygeos.points([[0, 0], [3, 4], [-2, 1]])
    assert np.allclose(mpg.get_x(ours), pygeos.get_x(theirs))
    assert np.allclose(mpg.get_y(ours), pygeos.get_y(theirs))
    assert np.allclose(mpg.distance(ours, mpg.points([0, 0])), pygeos.distance(theirs, pygeos.points([0, 0])))
    coords, index = mpg.get_coordinates(ours, return_index=True)
    ref_coords, ref_index = pygeos.get_coordinates(theirs, return_index=True)
    assert np.allclose(coords, ref_coords)
    assert np.array_equal(index, ref_index)


def test_simd_point_distance_handles_scalar_tail():
    ours = mpg.points([[0, 0], [3, 4], [-2, 1], [8, -6], [1, 1]])
    reference = pygeos.points([[0, 0], [3, 4], [-2, 1], [8, -6], [1, 1]])
    query = mpg.points([1, -1])
    ref_query = pygeos.points([1, -1])
    assert np.allclose(mpg.distance(ours, query), pygeos.distance(reference, ref_query))


def test_metrics_reuse_packed_array_buffers():
    geometries = np.array([mpg.polygons([[0, 0], [2, 0], [2, 1], [0, 1]])], dtype=object)
    mpg.area(geometries)
    packed = _PACK_CACHE[id(geometries)][1]
    mpg.length(geometries)
    assert _PACK_CACHE[id(geometries)][1] is packed


def test_point_polygon_predicates_match_pygeos_interior_boundary_hole_and_outside():
    shell = [[0, 0], [10, 0], [10, 10], [0, 10]]
    hole = [[2, 2], [4, 2], [4, 4], [2, 4]]
    ours = mpg.polygons(shell, holes=[hole])
    theirs = pygeos.polygons(shell, holes=[hole])
    op = mpg.points([[1, 1], [0, 5], [3, 3], [12, 1]])
    tp = pygeos.points([[1, 1], [0, 5], [3, 3], [12, 1]])
    assert np.array_equal(mpg.contains(ours, op), pygeos.contains(theirs, tp))
    assert np.array_equal(mpg.covers(ours, op), pygeos.covers(theirs, tp))
    assert np.array_equal(mpg.intersects(ours, op), pygeos.intersects(theirs, tp))
    assert np.array_equal(mpg.within(op, ours), pygeos.within(tp, theirs))


def test_wkt_roundtrip_matches_pygeos_canonical_output():
    text = "POLYGON ((0 0, 3 0, 3 2, 0 2, 0 0), (1 1, 2 1, 2 1.5, 1 1.5, 1 1))"
    ours = mpg.from_wkt(text)
    theirs = pygeos.from_wkt(text)
    assert mpg.to_wkt(ours) == pygeos.to_wkt(theirs)
    assert mpg.to_wkt(mpg.from_wkt(mpg.to_wkt(ours))) == mpg.to_wkt(ours)


@pytest.mark.parametrize("text", ["POINT (2 -1)", "LINESTRING (0 0, 3 4, 6 4)"])
def test_point_and_linestring_wkt_match_pygeos(text):
    assert mpg.to_wkt(mpg.from_wkt(text)) == pygeos.to_wkt(pygeos.from_wkt(text))


def test_box_and_point_point_predicates_match_pygeos():
    ours, theirs = mpg.box(0, 0, 10, 5), pygeos.box(0, 0, 10, 5)
    assert mpg.to_wkt(ours) == pygeos.to_wkt(theirs)
    point, ref_point = mpg.points([2, 3]), pygeos.points([2, 3])
    for operation in (mpg.contains, mpg.covers, mpg.within, mpg.intersects):
        assert operation(point, point) == getattr(pygeos, operation.__name__)(ref_point, ref_point)


def test_precision_loss_and_invalid_geometry_are_rejected_before_ffi():
    with pytest.raises(ValueError, match="exactly representable"):
        mpg.points([2**53 + 1, 0])
    with pytest.raises(ValueError, match="linear ring"):
        mpg.Geometry(2, [[0, 0], [1, 0]])
    with pytest.raises(NotImplementedError, match="out"):
        mpg.points([0, 0], out=np.empty(1, dtype=object))
