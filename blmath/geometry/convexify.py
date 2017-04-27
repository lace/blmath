def convexify_planar_curve(polyline, flatten=False, want_vertices=False):
    '''
    Take the convex hull of an almost planar 2-D curve in three-space.

    Params:
        - polyline:
            An instance of Polyline.
        - flatten:
            Boolean; if True, rotated curve will be flattened
            on the y-axis, thereby losing length. This loss
            may not be offset by the convex hull.
        - want_vertices:
            Boolean; do you want the indices to the convex hull vertices?
    '''
    import numpy as np
    from scipy.spatial import ConvexHull  # First thought this warning was caused by a pythonpath problem, but it seems more likely that the warning is caused by scipy import hackery. pylint: disable=no-name-in-module
    from blmath.geometry import Polyline
    from blmath.geometry.transform.rotation import rotate_to_xz_plane
    from blmath.geometry.transform.translation import translation

    if len(polyline.v) <= 1:
        return polyline

    rotated, R, p0 = rotate_to_xz_plane(polyline.v)

    if flatten:
        rotated[:, 1] = np.mean(rotated[:, 1])

    projected = rotated[:, [0, 2]]

    hull_vertices = ConvexHull(projected).vertices
    rotated_hull = rotated[hull_vertices]

    R_inv = np.array(np.mat(R.T).I)
    inv_rotated = np.dot(rotated_hull, R_inv)

    if p0 is not None:
        curve_hull = translation(np.array(inv_rotated), -p0)[0] # pylint: disable=invalid-unary-operand-type
    else:
        curve_hull = inv_rotated

    result = Polyline(v=curve_hull, closed=polyline.closed)

    if want_vertices:
        return result, hull_vertices
    else:
        return result
