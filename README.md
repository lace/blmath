blmath
======

[![version](https://img.shields.io/pypi/v/blmath?style=flat-square)][pypi]
[![license](https://img.shields.io/pypi/l/blmath?style=flat-square)][pypi]
[![python versions](https://img.shields.io/pypi/pyversions/blmath?style=flat-square)][pypi]

Collection of math-related utilities developed at Body Labs.

**This library is deprecated. The following libraries were broken out from this
package and now are maintained on their own:**

* **[vg][]** is a vector-geometry toolbelt for 3D points and vectors.
   * Was `blmath.numerics.vector_shortcuts`
* **[polliwog][]** provides low-level functions for working with 2D and 3D
  geometry, optimized for cloud computation.
    * Was `blmath.geometry`
* **[ounce][]** is a simple package for manipulating units of measure.
    * Was `blmath.units`
* **[entente][]** provides functions for working with meshes and pointclouds
  having vertexwise correspondence.
    * Includes `blmath.geometry.transform.find_rigid_transform`

Also related is **[lacecore][]** (the primary successor to [lace][]) which
provides polygonal meshes optimized for cloud computation.

Special mention is given to **[hyla][]**, a TypeScript counterpart to
[polliwog][].


[lacecore]: https://github.com/lace/lacecore/
[entente]: https://github.com/lace/entente/
[polliwog]: https://github.com/lace/polliwog/
[hyla]: https://github.com/lace/hyla/
[ounce]: https://github.com/lace/ounce/
[pypi]: https://pypi.org/project/blmath/
[lace]: https://github.com/lace/lace
[vg]: https://github.com/lace/vg
[polliwog]: https://github.com/lace/polliwog


Installation
------------

### Install dependencies

On macOS:

    brew install homebrew/science/suite-sparse
    brew install homebrew/science/opencv --without-numpy

On Linux:

    sudo apt-get install python-opencv libsuitesparse-dev

### Install the library

```sh
pip install blmath
```

And import it just like the upstream library:

```py
from blmath.numerics import vx
```

A collection of math related utilities used by many bits of BodyLabs' code.


blmath.numerics
---------------

Functions for manipulating numeric arrays, numbers, and linear algebra.

The [most commonly used of these](blmath/numerics/__init__.py) are directly imported into
`blmath.numerics`.

- [blmath.numerics.vx](blmath/numerics/vector_shortcuts.py) is a namespace of common linear
  algebra operations. These are easily expressed in numpy, but abstracted for
  readability purposes.
- [blmath.numerics.coercion](blmath/numerics/coercion.py) contains a validation function
  `as_numeric_array`, which produces useful error messages up front on bad
  inputs, in place of cryptic messages like "cannot broadcast..." later on.
- [blmath.numerics.operations](blmath/numerics/operations.py) contains basic numerical
  operations such as `zero_safe_divide`.
- [blmath.numerics.predicates](blmath/numerics/predicates.py) contains functions like
  `isnumeric`.
- [blmath.numerics.rounding](blmath/numerics/rounding.py) contains functions including
  "round to nearest" and `roundedlist`.
- [blmath.numerics.numpy_ext](blmath/numerics/numpy_ext.py) contains numpy utility
  functions.
- [blmath.numerics.matlab](blmath/numerics/matlab.py) contains some matlab shortcuts which
  have no numpy equivalent. At MPI the fitting code was originally written in
  Matlab before it was ported to Python.

[blmath.numerics.linalg](blmath/numerics/linalg) contains linear algebra operations.

- [blmath.numerics.linalg.sparse_cg](blmath/numerics/linalg/sparse_cg.py) contains a faster
  matrix solve optimized for sparse Jacobians.
- [blmath.numerics.linalg.lchol](blmath/numerics/linalg/lchol.py) contains a Cythonized
  implementation of Cholesky factorization.
- [blmath.numerics.linalg.isomorphism](blmath/numerics/linalg/isomorphism.py) computes the
  isomorphism between two bases.
- [blmath.numerics.linalg.gram_schmidt](blmath/numerics/linalg/gram_schmidt.py) provides a
  function for orthonormalization.

blmath.geometry
---------------

Geometric operations, transforms, and primitives, in 2D and 3D.

The [most commonly used of these](blmath/geometry/__init__.py) are directly imported into
`blmath.geometry`.

- [blmath.geometry.Box](blmath/geometry/primitives/box.py) represents an axis-aligned
  cuboid.
- [blmath.geometry.Plane](blmath/geometry/primitives/plane.py) represents a 2-D plane in
  3-space (not a hyperplane).
- [blmath.geometry.Polyline](blmath/geometry/primitives/polyline.py) represents an
  unconstrained polygonal chain in 3-space.

`blmath.geometry.transform` includes code for 3D transforms.

- [blmath.geometry.transform.CompositeTransform](blmath/geometry/transform/composite.py)
  represents a composite transform using homogeneous coordinates. (Thanks avd!)
- [blmath.geometry.transform.CoordinateManager](blmath/geometry/transform/coordinate_manager.py)
  provides a convenient interface for named reference frames within a stack of
  transforms and projecting points from one reference frame to another.
- [blmath.geometry.transform.find_rigid_transform](blmath/geometry/transform/rigid_transform.py)
  finds a rotation and translation that closely transforms one set of points to
  another. Its cousin `find_rigid_rotation` does the same, but only allows
  rotation, not translation.
- [blmath.geometry.transform.rotation.rotation_from_up_and_look](blmath/geometry/transform/rotation.py)
  produces a rotation matrix that gets a mesh into the canonical reference frame
  from "up" and "look" vectors.

Other modules:

- [blmath.geometry.apex](blmath/geometry/apex.py) provides functions for finding the most
  extreme point.
- [blmath.geometry.barycentric](blmath/geometry/barycentric.py) provides a function for
  projecting a point to a triangle using barycentric coordinates.
- [blmath.geometry.convexify](blmath/geometry/convexify.py) provides a function for
  producing a convex hull from a mostly-planar curve.
- [blmath.geometry.segment](blmath/geometry/segment.py) provides functions for working with
  line segments in n-space.

blmath.value
------------
Class for wrapping and manipulating `value`/`units` pairs.

blmath.units
------------
TODO write something here

blmath.console
------------
- [blmath.console.input_float](blmath/console.py) reads and returns a float from console.
- [blmath.console.input_value](blmath/console.py) combines `units` with a float input from console
  and returns `Value` object.



Development
-----------

```sh
pip install -r requirements_dev.txt
pip install -e .  # builds the native extension
rake unittest
rake lint
```

Tests are configured to run in both python 2.7 and 3.6 locally via tox as well as in CircleCI.
To run tests in multiple versions of python, run `tox`:

```sh
pip install -r requirements_dev.txt
tox
```

You need to make sure that `python2.7` and `python3.6` are valid commands; this can be done in pyenv via `pyenv global 3.6.5 2.7.15`


Acknowledgements
----------------

This collection was developed at Body Labs and includes a combination of code
developed at Body Labs, from legacy code and significant new portions by
[Eric Rachlin][], [Alex Weiss][], and [Paul Melnikow][]. It was extracted
from the Body Labs codebase and open-sourced by [Alex Weiss][]. In 2018 it was
[forked by Paul Melnikow][fork] and published as [metablmath][]. Thanks to a
repository and package transfer from Body Labs, the fork has been merged back
into the original.


[eric rachlin]: https://github.com/eerac
[alex weiss]: https://github.com/algrs
[paul melnikow]: https://github.com/paulmelnikow
[fork]: https://github.com/metabolize/blmath
[metablmath]: https://pypi.org/project/metablmath/


License
-------

The project is licensed under the two-clause BSD license.
