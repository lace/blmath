Changelog for the Metabolize fork
=================================

## 1.6.1 (Aug 29, 2019)

- TriNormalsScaled: Fix for Python 3


## 1.6.0 (Aug 28, 2019)

- Update for Python 3.


## 1.5.0 (Mar 24, 2019)

- Move Plane.polyline_xsection to Polyline.intersect_plane.
- Polyline: Add `reindexed()` method.
- Polyline: Add `cut_by_plane()` method.

## 1.4.0 (Mar 22, 2019)

- Plane.polyline_xsection: Add `ret_edge_indices` parameter
- Polyline: Add convenience attributes `segments` and `segment_vectors`
- Polyline: Add `flip()` method.

## 1.3.0 (Oct 4, 2018)

As of this release, break from the upstream revision history and adopt ordinary
semver.

Identical to `1.2.5-post3`.


## 1.2.5-post3 (Oct 3, 2018)

- Add blmath.geometry.shapes.create_rectangular_prism, create_cube,
  create_triangular_prism, and create_horizontal_plane
- Add blmath.geometry.surface_normals.surface_normal


## 1.2.5-post2 (Sep 9, 2018)

- Temporarily drop support for Python 3.


## 1.2.5-post1 (Sep 9, 2018)

- Fix install issue by removing extraneous dependency.
