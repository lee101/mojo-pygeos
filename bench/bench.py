"""Measure the public vectorised API against upstream pygeos on identical data."""

from __future__ import annotations

import os
import platform
import sys
import time

import numpy as np
import pygeos

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "python"))
import mojo_pygeos as mpg  # noqa: E402


def best(fn, repeat=3):
    value = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        value = min(value, time.perf_counter() - start)
    return value


def row(name, ours, theirs):
    ratio = theirs / ours
    verdict = "faster" if ratio > 1 else "slower"
    print(f"| {name} | {ours * 1e3:.1f} ms | {theirs * 1e3:.1f} ms | {ratio:.2f}x {verdict} |")


def main():
    n = 100_000
    rng = np.random.default_rng(0)
    xy = rng.normal(size=(n, 4, 2))
    xy[:, 2] += [2.0, 2.0]
    xy[:, 3] += [-1.0, 2.0]
    closed = np.concatenate((xy, xy[:, :1]), axis=1)
    ours = np.array([mpg.polygons(coords) for coords in closed], dtype=object)
    theirs = np.array([pygeos.polygons(coords) for coords in closed], dtype=object)
    points_ours = mpg.points(rng.normal(size=(n, 2)))
    points_theirs = pygeos.points(mpg.get_coordinates(points_ours))
    query_ours = mpg.points([0.25, 0.25])
    query_theirs = pygeos.points([0.25, 0.25])

    mpg.area(ours)
    print("# mojo-pygeos benchmark")
    print(f"Machine: {platform.node()} | {platform.platform()} | Python {platform.python_version()}")
    print("| kernel | mojo-pygeos | pygeos 0.14 | ratio |")
    print("| --- | ---: | ---: | --- |")
    row("area, 100k quadrilaterals", best(lambda: mpg.area(ours)), best(lambda: pygeos.area(theirs)))
    row("length, 100k quadrilaterals", best(lambda: mpg.length(ours)), best(lambda: pygeos.length(theirs)))
    row("bounds, 100k quadrilaterals", best(lambda: mpg.bounds(ours)), best(lambda: pygeos.bounds(theirs)))
    row("centroid, 100k quadrilaterals", best(lambda: mpg.centroid(ours)), best(lambda: pygeos.centroid(theirs)))
    row("point distance, 100k pairs", best(lambda: mpg.distance(points_ours, query_ours)),
        best(lambda: pygeos.distance(points_theirs, query_theirs)))


if __name__ == "__main__":
    main()
