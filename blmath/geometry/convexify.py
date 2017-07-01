def convexify_planar_curve(polyline, flatten=False, want_vertices=False, normal=None):
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
    from blmath.geometry.transform.rotation import estimate_normal

    v = polyline.v
    if len(v) <= 1:
        return polyline

    if normal is None:
        normal = estimate_normal(v)

    # to call ConvexHull, the points must be projected to a plane.
    # with an eye toward numerical stability, this can be done by dropping 
    # whichever dimension is closest to the normal
    dim_to_drop = np.argmax(np.abs(normal))
    dims_to_keep = [i for i in range(3) if i != dim_to_drop]
    proj_v = v[:, dims_to_keep]

    hull_vertices = ConvexHull(proj_v).vertices
    result = Polyline(v=v[hull_vertices], closed=polyline.closed)

    if want_vertices:
        return result, hull_vertices
    else:
        return result
