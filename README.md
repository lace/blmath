blmath
======

[![version](https://img.shields.io/pypi/v/blmath?style=flat-square)][pypi]
[![python versions](https://img.shields.io/pypi/pyversions/blmath?style=flat-square)][pypi]
[![build status](https://img.shields.io/circleci/project/github/lace/blmath/master.svg?style=flat-square)][circle]

Collection of math-related utilities developed at Body Labs.

This is a legacy library in active maintenance. The goals are:

- Keep the library working in current versions of Python and other tools.
- Make bug fixes.
- Provide API stability and backward compatibility with the upstream version.

The main thrust is to transition the functionality in this library (and the
code which depends on it, such as [lace][] to new libraries include [vg][] and
[polliwog][]. In general, new functionality should be added in the new
libraries. As needed, additional libraries can be added to the lace org.

[circle]: https://circleci.com/gh/lace/blmath
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

The [most commonly used of these](__init__.py) are directly imported into
`blmath.numerics`.

- [blmath.numerics.vx](vector_shortcuts.py) is a namespace of common linear
  algebra operations. These are easily expressed in numpy, but abstracted for
  readability purposes.
- [blmath.numerics.coercion](coercion.py) contains a validation function
  `as_numeric_array`, which produces useful error messages up front on bad
  inputs, in place of cryptic messages like "cannot broadcast..." later on.
- [blmath.numerics.operations](operations.py) contains basic numerical
  operations such as `zero_safe_divide`.
- [blmath.numerics.predicates](predicates.py) contains functions like
  `isnumeric`.
- [blmath.numerics.rounding](rounding.py) contains functions including
  "round to nearest" and `roundedlist`.
- [blmath.numerics.numpy_ext](numpy_ext.py) contains numpy utility
  functions.
- [blmath.numerics.matlab](matlab.py) contains some matlab shortcuts which
  have no numpy equivalent. At MPI the fitting code was originally written in
  Matlab before it was ported to Python.

[blmath.numerics.linalg](linalg) contains linear algebra operations.

- [blmath.numerics.linalg.sparse_cg](linalg/sparse_cg.py) contains a faster
  matrix solve optimized for sparse Jacobians.
- [blmath.numerics.linalg.lchol](linalg/lchol.py) contains a Cythonized
  implementation of Cholesky factorization.
- [blmath.numerics.linalg.isomorphism](linalg/isomorphism.py) computes the
  isomorphism between two bases.
- [blmath.numerics.linalg.gram_schmidt](linalg/gram_schmidt.py) provides a
  function for orthonormalization.

blmath.geometry
---------------

Geometric operations, transforms, and primitives, in 2D and 3D.

The [most commonly used of these](__init__.py) are directly imported into
`blmath.geometry`.

- [blmath.geometry.Box](primitives/box.py) represents an axis-aligned
  cuboid.
- [blmath.geometry.Plane](primitives/plane.py) represents a 2-D plane in
  3-space (not a hyperplane).
- [blmath.geometry.Polyline](primitives/polyline.py) represents an
  unconstrained polygonal chain in 3-space.

`blmath.geometry.transform` includes code for 3D transforms.

- [blmath.geometry.transform.CompositeTransform](transform/composite.py)
  represents a composite transform using homogeneous coordinates. (Thanks avd!)
- [blmath.geometry.transform.CoordinateManager](transform/coordinate_manager.py)
  provides a convenient interface for named reference frames within a stack of
  transforms and projecting points from one reference frame to another.
- [blmath.geometry.transform.find_rigid_transform](transform/rigid_transform.py)
  finds a rotation and translation that closely transforms one set of points to
  another. Its cousin `find_rigid_rotation` does the same, but only allows
  rotation, not translation.
- [blmath.geometry.transform.rotation.rotation_from_up_and_look](transform/rotation.py)
  produces a rotation matrix that gets a mesh into the canonical reference frame
  from "up" and "look" vectors.

Other modules:

- [blmath.geometry.apex](apex.py) provides functions for finding the most
  extreme point.
- [blmath.geometry.barycentric](barycentric.py) provides a function for
  projecting a point to a triangle using barycentric coordinates.
- [blmath.geometry.convexify](convexify.py) provides a function for
  producing a convex hull from a mostly-planar curve.
- [blmath.geometry.segment](segment.py) provides functions for working with
  line segments in n-space.

blmath.value
------------
Class for wrapping and manipulating `value`/`units` pairs.

blmath.units
------------
TODO write something here

blmath.console
------------
- [blmath.console.input_float](console.py) reads and returns a float from console.
- [blmath.console.input_value](console.py) combines `units` with a float input from console
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

Contribute
----------

- Issue Tracker: https://github.com/lace/blmath/issues
- Source Code: https://github.com/lace/blmath

Pull requests welcome!


Support
-------

If you are having issues, please let us know.


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
