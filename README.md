# mojo-pygeos

`mojo-pygeos` is a standalone Mojo implementation of a practical, vectorised
subset of [pygeos](https://github.com/pygeos/pygeos): two-dimensional planar
points, line strings, linear rings, and polygons.  The public Python layer is
named `mojo_pygeos`; import it as `pygeos` when migrating covered code.

The project compares every covered operation to the real `pygeos==0.14`
package.  PyGEOS itself is now merged into Shapely, but remains installable on
PyPI and conda-forge for direct parity tests.

## Covered subset

| area | covered operations |
| --- | --- |
| constructors | `points`, `linestrings`, `linearrings`, `polygons`, `box` |
| measurements | `area`, `length`, `bounds`, `centroid` |
| accessors | `get_coordinates`, `get_num_coordinates`, `get_type_id`, `get_x`, `get_y` |
| spatial operations | broadcasted point/point `distance`; point/polygon `contains`, `covers`, `within`, `intersects` |
| interchange | 2-D `to_wkt`, `from_wkt` for `POINT`, `LINESTRING`, and `POLYGON` |

Polygon holes, implicit ring closure, NumPy-shaped object arrays, and scalar
or broadcasted operands for the covered vectorised operations are supported.
Empty geometries, Z/M coordinates,
multi-geometries, geometry collections, prepared geometries, STRtrees,
constructive topology operations, and line/polygon distance or predicates are
not yet covered; those operations raise `NotImplementedError` rather than
quietly returning an approximation.

## Install and use

```bash
pixi install
pixi run build
```

```python
import mojo_pygeos as pygeos

parcel = pygeos.box(0, 0, 10, 5)
print(pygeos.area(parcel))                         # 50.0
print(pygeos.contains(parcel, pygeos.points([2, 3])))  # True
print(pygeos.to_wkt(pygeos.centroid(parcel)))      # POINT (5 2.5)
```

Run the parity suite with `pixi run test`.  The wrapper builds the shared
library on demand when a source file is newer than it, so importing the module
also works directly from this checkout under Pixi.

## How it works

```
python/mojo_pygeos/  Geometry objects and NumPy broadcasting
          |  ctypes: contiguous float64/int64 buffer addresses
src/capi.mojo        one Mojo compilation unit and C-ABI exports
          |  packed coordinates, coordinate/ring/geometry offsets
dist/libmojo-pygeos.so
```

Each batch is packed once into a contiguous `float64[n_coordinates, 2]`
coordinate buffer, plus int64 offsets for geometries and rings.  Mojo kernels
walk those buffers without allocating: a single pass produces area, perimeter,
bounds, and centroid; a second kernel implements boundary-aware ray casting
for point-in-polygon predicates.  ctypes passes buffer addresses as `Int`, and
the exported Mojo functions rebuild mutable pointers with
`AnyOrigin[mut=True]`, matching the Mojo nightly ABI requirements.

## Benchmark

Measured with `pixi run bench` on `leaf-gpu-dedicated-server`,
Linux 6.8.0-136-generic x86_64, Python 3.10.20.  Values are the best of three
runs and include public-API packing/broadcasting costs.

| kernel | mojo-pygeos | pygeos 0.14 | ratio |
| --- | ---: | ---: | --- |
| area, 100k quadrilaterals | 10.0 ms | 5.1 ms | 0.51x slower |
| length, 100k quadrilaterals | 10.0 ms | 5.2 ms | 0.52x slower |
| bounds, 100k quadrilaterals | 9.9 ms | 34.3 ms | 3.45x faster |
| centroid, 100k quadrilaterals | 377.4 ms | 74.2 ms | 0.20x slower |
| point distance, 100k pairs | 0.3 ms | 36.1 ms | 130.58x faster |

Packed coordinate buffers are retained for repeated vectorised operations, so
the NumPy-to-Mojo boundary remains zero-copy after the initial pack. Point
distance uses a SIMD CPU kernel with a scalar tail. GPU execution is omitted:
these geometry passes are memory-bound and below the arithmetic intensity at
which transfer and launch overhead can beat the CPU.

## License

MIT
