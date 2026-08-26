"""Packed-coordinate planar-geometry kernels exposed to ctypes."""

from std.math import sqrt
from std.sys.info import simd_width_of

comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]


def distance(x0: Float64, y0: Float64, x1: Float64, y1: Float64) -> Float64:
    var dx = x1 - x0
    var dy = y1 - y0
    return sqrt(dx * dx + dy * dy)


def area_one(
    xs: Ptr,
    ys: Ptr,
    ring_offsets_addr: Int,
    geometry_ring_offsets_addr: Int,
    kinds_addr: Int,
    g: Int,
    areas: Ptr,
):
    var ring_offsets = IPtr(unsafe_from_address=ring_offsets_addr)
    var geometry_ring_offsets = IPtr(unsafe_from_address=geometry_ring_offsets_addr)
    var kinds = IPtr(unsafe_from_address=kinds_addr)
    if kinds[g] != 3:
        areas[g] = 0.0
        return
    comptime W = simd_width_of[DType.float64]()
    var first_ring = Int(geometry_ring_offsets[g])
    var last_ring = Int(geometry_ring_offsets[g + 1])
    var total_area = 0.0
    for r in range(first_ring, last_ring):
        var i = Int(ring_offsets[r])
        var end = Int(ring_offsets[r + 1]) - 1
        var sums = SIMD[DType.float64, W](0.0)
        while i + W <= end:
            var x0 = xs.load[width=W](i)
            var y0 = ys.load[width=W](i)
            var x1 = xs.load[width=W](i + 1)
            var y1 = ys.load[width=W](i + 1)
            sums += x0 * y1 - x1 * y0
            i += W
        var area2 = sums.reduce_add()
        while i < end:
            area2 += xs[i] * ys[i + 1] - xs[i + 1] * ys[i]
            i += 1
        var sign = 1.0 if r == first_ring else -1.0
        total_area += sign * abs(area2) * 0.5
    areas[g] = total_area


@export("mpg_area")
def mpg_area(
    xs_addr: Int,
    ys_addr: Int,
    ring_offsets_addr: Int,
    geometry_ring_offsets_addr: Int,
    kinds_addr: Int,
    n: Int,
    area_addr: Int,
) abi("C"):
    var xs = Ptr(unsafe_from_address=xs_addr)
    var ys = Ptr(unsafe_from_address=ys_addr)
    var areas = Ptr(unsafe_from_address=area_addr)
    for g in range(n):
        area_one(xs, ys, ring_offsets_addr, geometry_ring_offsets_addr, kinds_addr, g, areas)


def length_one(xs: Ptr, ys: Ptr, ring_offsets: IPtr, geometry_ring_offsets: IPtr, kinds: IPtr, g: Int, lengths: Ptr):
    if kinds[g] == 0:
        lengths[g] = 0.0
        return
    comptime W = simd_width_of[DType.float64]()
    var total = 0.0
    for r in range(Int(geometry_ring_offsets[g]), Int(geometry_ring_offsets[g + 1])):
        var i = Int(ring_offsets[r])
        var end = Int(ring_offsets[r + 1]) - 1
        var sums = SIMD[DType.float64, W](0.0)
        while i + W <= end:
            var dx = xs.load[width=W](i + 1) - xs.load[width=W](i)
            var dy = ys.load[width=W](i + 1) - ys.load[width=W](i)
            sums += sqrt(dx * dx + dy * dy)
            i += W
        total += sums.reduce_add()
        while i < end:
            total += distance(xs[i], ys[i], xs[i + 1], ys[i + 1])
            i += 1
    lengths[g] = total


@export("mpg_length")
def mpg_length(xs_addr: Int, ys_addr: Int, ring_offsets_addr: Int, geometry_ring_offsets_addr: Int, kinds_addr: Int, n: Int, length_addr: Int) abi("C"):
    var xs = Ptr(unsafe_from_address=xs_addr)
    var ys = Ptr(unsafe_from_address=ys_addr)
    var ring_offsets = IPtr(unsafe_from_address=ring_offsets_addr)
    var geometry_ring_offsets = IPtr(unsafe_from_address=geometry_ring_offsets_addr)
    var kinds = IPtr(unsafe_from_address=kinds_addr)
    var lengths = Ptr(unsafe_from_address=length_addr)
    for g in range(n):
        length_one(xs, ys, ring_offsets, geometry_ring_offsets, kinds, g, lengths)


def bounds_one(xs: Ptr, ys: Ptr, coord_offsets: IPtr, g: Int, bounds_ptr: Ptr):
    var begin = Int(coord_offsets[g])
    var end = Int(coord_offsets[g + 1])
    var lo_x = xs[begin]
    var hi_x = lo_x
    var lo_y = ys[begin]
    var hi_y = lo_y
    for i in range(begin + 1, end):
        lo_x = min(lo_x, xs[i])
        hi_x = max(hi_x, xs[i])
        lo_y = min(lo_y, ys[i])
        hi_y = max(hi_y, ys[i])
    bounds_ptr[4 * g] = lo_x
    bounds_ptr[4 * g + 1] = lo_y
    bounds_ptr[4 * g + 2] = hi_x
    bounds_ptr[4 * g + 3] = hi_y


@export("mpg_bounds")
def mpg_bounds(xs_addr: Int, ys_addr: Int, coord_offsets_addr: Int, n: Int, bounds_addr: Int) abi("C"):
    var xs = Ptr(unsafe_from_address=xs_addr)
    var ys = Ptr(unsafe_from_address=ys_addr)
    var coord_offsets = IPtr(unsafe_from_address=coord_offsets_addr)
    var bounds_ptr = Ptr(unsafe_from_address=bounds_addr)
    for g in range(n):
        bounds_one(xs, ys, coord_offsets, g, bounds_ptr)


def centroid_one(xs: Ptr, ys: Ptr, coord_offsets: IPtr, ring_offsets: IPtr, geometry_ring_offsets: IPtr, kinds: IPtr, g: Int, centroids: Ptr):
    comptime W = simd_width_of[DType.float64]()
    var begin = Int(coord_offsets[g])
    var kind = Int(kinds[g])
    var center_x = xs[begin]
    var center_y = ys[begin]
    if kind == 1:
        var i = begin
        var end = Int(coord_offsets[g + 1]) - 1
        var length_v = SIMD[DType.float64, W](0.0)
        var x_v = SIMD[DType.float64, W](0.0)
        var y_v = SIMD[DType.float64, W](0.0)
        while i + W <= end:
            var x0 = xs.load[width=W](i)
            var y0 = ys.load[width=W](i)
            var x1 = xs.load[width=W](i + 1)
            var y1 = ys.load[width=W](i + 1)
            var d = sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0))
            length_v += d
            x_v += (x0 + x1) * 0.5 * d
            y_v += (y0 + y1) * 0.5 * d
            i += W
        var total_length = length_v.reduce_add()
        var weighted_x = x_v.reduce_add()
        var weighted_y = y_v.reduce_add()
        while i < end:
            var d = distance(xs[i], ys[i], xs[i + 1], ys[i + 1])
            total_length += d
            weighted_x += (xs[i] + xs[i + 1]) * 0.5 * d
            weighted_y += (ys[i] + ys[i + 1]) * 0.5 * d
            i += 1
        if total_length != 0.0:
            center_x = weighted_x / total_length
            center_y = weighted_y / total_length
    elif kind == 3:
        var first_ring = Int(geometry_ring_offsets[g])
        var total_area = 0.0
        center_x = 0.0
        center_y = 0.0
        for r in range(first_ring, Int(geometry_ring_offsets[g + 1])):
            var i = Int(ring_offsets[r])
            var end = Int(ring_offsets[r + 1]) - 1
            var area_v = SIMD[DType.float64, W](0.0)
            var x_v = SIMD[DType.float64, W](0.0)
            var y_v = SIMD[DType.float64, W](0.0)
            while i + W <= end:
                var x0 = xs.load[width=W](i)
                var y0 = ys.load[width=W](i)
                var x1 = xs.load[width=W](i + 1)
                var y1 = ys.load[width=W](i + 1)
                var cross = x0 * y1 - x1 * y0
                area_v += cross
                x_v += (x0 + x1) * cross
                y_v += (y0 + y1) * cross
                i += W
            var area2 = area_v.reduce_add()
            var xsum = x_v.reduce_add()
            var ysum = y_v.reduce_add()
            while i < end:
                var cross = xs[i] * ys[i + 1] - xs[i + 1] * ys[i]
                area2 += cross
                xsum += (xs[i] + xs[i + 1]) * cross
                ysum += (ys[i] + ys[i + 1]) * cross
                i += 1
            var ring_area = abs(area2) * 0.5
            var sign = 1.0 if r == first_ring else -1.0
            total_area += sign * ring_area
            if area2 != 0.0:
                center_x += sign * ring_area * xsum / (3.0 * area2)
                center_y += sign * ring_area * ysum / (3.0 * area2)
        if total_area != 0.0:
            center_x /= total_area
            center_y /= total_area
        else:
            center_x = xs[begin]
            center_y = ys[begin]
    centroids[2 * g] = center_x
    centroids[2 * g + 1] = center_y


@export("mpg_centroid")
def mpg_centroid(xs_addr: Int, ys_addr: Int, coord_offsets_addr: Int, ring_offsets_addr: Int, geometry_ring_offsets_addr: Int, kinds_addr: Int, n: Int, centroid_addr: Int) abi("C"):
    var xs = Ptr(unsafe_from_address=xs_addr)
    var ys = Ptr(unsafe_from_address=ys_addr)
    var coord_offsets = IPtr(unsafe_from_address=coord_offsets_addr)
    var ring_offsets = IPtr(unsafe_from_address=ring_offsets_addr)
    var geometry_ring_offsets = IPtr(unsafe_from_address=geometry_ring_offsets_addr)
    var kinds = IPtr(unsafe_from_address=kinds_addr)
    var centroids = Ptr(unsafe_from_address=centroid_addr)
    for g in range(n):
        centroid_one(xs, ys, coord_offsets, ring_offsets, geometry_ring_offsets, kinds, g, centroids)


def on_segment(x0: Float64, y0: Float64, x1: Float64, y1: Float64, x: Float64, y: Float64) -> Bool:
    var cross = (x - x0) * (y1 - y0) - (y - y0) * (x1 - x0)
    if abs(cross) > 1e-12:
        return False
    return x >= min(x0, x1) and x <= max(x0, x1) and y >= min(y0, y1) and y <= max(y0, y1)


def point_in_ring(coords: Ptr, begin: Int, end: Int, x: Float64, y: Float64) -> Int:
    var inside = False
    for i in range(begin, end - 1):
        var x0 = coords[2 * i]
        var y0 = coords[2 * i + 1]
        var x1 = coords[2 * (i + 1)]
        var y1 = coords[2 * (i + 1) + 1]
        if on_segment(x0, y0, x1, y1, x, y):
            return 1
        if (y0 > y) != (y1 > y):
            var hit_x = (x1 - x0) * (y - y0) / (y1 - y0) + x0
            if x < hit_x:
                inside = not inside
    return 2 if inside else 0


@export("mpg_point_distance_scalar")
def mpg_point_distance_scalar(
    xs_addr: Int,
    ys_addr: Int,
    n: Int,
    x: Float64,
    y: Float64,
    result_addr: Int,
) abi("C"):
    var xs = Ptr(unsafe_from_address=xs_addr)
    var ys = Ptr(unsafe_from_address=ys_addr)
    var result = Ptr(unsafe_from_address=result_addr)
    comptime W = simd_width_of[DType.float64]()
    var i = 0
    while i + W <= n:
        var dx = xs.load[width=W](i) - SIMD[DType.float64, W](x)
        var dy = ys.load[width=W](i) - SIMD[DType.float64, W](y)
        result.store(i, sqrt(dx * dx + dy * dy))
        i += W
    while i < n:
        result[i] = distance(xs[i], ys[i], x, y)
        i += 1


@export("mpg_point_relation")
def mpg_point_relation(
    coords_addr: Int,
    ring_offsets_addr: Int,
    geometry_ring_offsets_addr: Int,
    kinds_addr: Int,
    n: Int,
    xs_addr: Int,
    ys_addr: Int,
    result_addr: Int,
) abi("C"):
    var coords = Ptr(unsafe_from_address=coords_addr)
    var ring_offsets = IPtr(unsafe_from_address=ring_offsets_addr)
    var geometry_ring_offsets = IPtr(unsafe_from_address=geometry_ring_offsets_addr)
    var kinds = IPtr(unsafe_from_address=kinds_addr)
    var xs = Ptr(unsafe_from_address=xs_addr)
    var ys = Ptr(unsafe_from_address=ys_addr)
    var result = IPtr(unsafe_from_address=result_addr)
    for g in range(n):
        var first = Int(geometry_ring_offsets[g])
        var last = Int(geometry_ring_offsets[g + 1])
        var x = xs[g]
        var y = ys[g]
        var value = 0
        if kinds[g] == 0:
            value = 2 if coords[2 * Int(ring_offsets[first])] == x and coords[2 * Int(ring_offsets[first]) + 1] == y else 0
        elif kinds[g] == 1:
            for i in range(Int(ring_offsets[first]), Int(ring_offsets[first + 1]) - 1):
                if on_segment(coords[2 * i], coords[2 * i + 1], coords[2 * (i + 1)], coords[2 * (i + 1) + 1], x, y):
                    value = 1
                    if x != coords[2 * Int(ring_offsets[first])] or y != coords[2 * Int(ring_offsets[first]) + 1]:
                        value = 2
        else:
            value = point_in_ring(coords, Int(ring_offsets[first]), Int(ring_offsets[first + 1]), x, y)
            if value == 2:
                for r in range(first + 1, last):
                    var hole = point_in_ring(coords, Int(ring_offsets[r]), Int(ring_offsets[r + 1]), x, y)
                    if hole == 1:
                        value = 1
                    elif hole == 2:
                        value = 0
        result[g] = Int64(value)
