import numpy as np
from blmath.numerics import vx

class Plane(object):
    '''
    A 2-D plane in 3-space (not a hyperplane).

    Params:
        - point_on_plane, plane_normal:
            1 x 3 np.arrays
    '''

    def __init__(self, point_on_plane, unit_normal):
        if vx.almost_zero(unit_normal):
            raise ValueError('unit_normal should not be the zero vector')

        unit_normal = vx.normalize(unit_normal)

        self._r0 = point_on_plane
        self._n = unit_normal

    @classmethod
    def from_points(cls, p1, p2, p3):
        '''
        If the points are oriented in a counterclockwise direction, the plane's
        normal extends towards you.

        '''
        from blmath.numerics import as_numeric_array

        p1 = as_numeric_array(p1, shape=(3,))
        p2 = as_numeric_array(p2, shape=(3,))
        p3 = as_numeric_array(p3, shape=(3,))

        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)

        return cls(point_on_plane=p1, unit_normal=normal)

    @classmethod
    def from_points_and_vector(cls, p1, p2, vector):
        '''
        Compute a plane which contains two given points and the given
        vector. Its reference point will be p1.

        For example, to find the vertical plane that passes through
        two landmarks:

            from_points_and_normal(p1, p2, vector)

        Another way to think about this: identify the plane to which
        your result plane should be perpendicular, and specify vector
        as its normal vector.

        '''
        from blmath.numerics import as_numeric_array

        p1 = as_numeric_array(p1, shape=(3,))
        p2 = as_numeric_array(p2, shape=(3,))

        v1 = p2 - p1
        v2 = as_numeric_array(vector, shape=(3,))
        normal = np.cross(v1, v2)

        return cls(point_on_plane=p1, unit_normal=normal)

    @classmethod
    def fit_from_points(cls, points):
        '''
        Fits a plane whose normal is orthgonal to the first two principal axes
        of variation in the data and centered on the points' centroid.
        '''
        eigval, eigvec = np.linalg.eig(np.cov(points.T))
        ordering = np.argsort(eigval)[::-1]
        normal = np.cross(eigvec[:, ordering[0]], eigvec[:, ordering[1]])
        return cls(points.mean(axis=0), normal)

    @property
    def equation(self):
        '''
        Returns parameters A, B, C, D as a 1 x 4 np.array, where

            Ax + By + Cz + D = 0

        defines the plane.

        params:
            - normalized:
                Boolean, indicates whether or not the norm of the vector [A, B, C] is 1.
                Useful when computing the distance from a point to the plane.
        '''
        A, B, C = self._n
        D = -self._r0.dot(self._n)

        return np.array([A, B, C, D])

    @property
    def reference_point(self):
        '''
        The point used to create this plane.

        '''
        return self._r0

    @property
    def canonical_point(self):
        '''
        A canonical point on the plane, the one at which the normal
        would intersect the plane if drawn from the origin (0, 0, 0).

        This is computed by projecting the reference point onto the
        normal.

        This is useful for partitioning the space between two planes,
        as we do when searching for planar cross sections.

        '''
        return self._r0.dot(self._n) * self._n

    @property
    def normal(self):
        '''
        Return the plane's normal vector.

        '''
        return np.array(self._n)

    def flipped(self):
        '''
        Creates a new Plane with an inverted orientation.
        '''
        normal = self._n * -1
        return Plane(self._r0, normal)

    def sign(self, points):
        '''
        Given an array of points, return an array with +1 for points in front
        of the plane (in the direction of the normal), -1 for points behind
        the plane (away from the normal), and 0 for points on the plane.

        '''
        return np.sign(self.signed_distance(points))

    def points_in_front(self, points, inverted=False, ret_indices=False):
        '''
        Given an array of points, return the points which lie either on the
        plane or in the half-space in front of it (i.e. in the direction of
        the plane normal).

        points: An array of points.
        inverted: When `True`, invert the logic. Return the points that lie
          behind the plane instead.
        ret_indices: When `True`, return the indices instead of the points
          themselves.

        '''
        sign = self.sign(points)

        if inverted:
            mask = np.less_equal(sign, 0)
        else:
            mask = np.greater_equal(sign, 0)

        indices = np.flatnonzero(mask)

        return indices if ret_indices else points[indices]

    def signed_distance(self, points):
        '''
        Returns the signed distances given an np.array of 3-vectors.

        Params:
            - points:
                V x 3 np.array
        '''
        return np.dot(vx.pad_with_ones(points), self.equation)

    def distance(self, points):
        return np.absolute(self.signed_distance(points))

    def project_point(self, point):
        '''
        Project a given point to the plane.

        '''
        # Translate the point back to the plane along the normal.
        signed_distance_to_point = self.signed_distance(point.reshape((-1, 3)))[0]
        return point - signed_distance_to_point * self._n

    def polyline_xsection(self, polyline):
        '''
        Returns the points of intersection between the plane and any of the
        edges of `polyline`, which should be an instance of Polyline.

        '''
        # Identify edges with endpoints that are not on the same side of the plane
        sgn_dists = self.signed_distance(polyline.v)
        which_es = np.abs(np.sign(sgn_dists)[polyline.e].sum(axis=1)) != 2
        # For the intersecting edges, compute the distance of the endpoints to the plane
        endpoint_distances = np.abs(sgn_dists[polyline.e[which_es]])
        # Normalize the rows of endpoint_distances
        t = endpoint_distances / endpoint_distances.sum(axis=1)[:, np.newaxis]
        # Take a weighted average of the endpoints to obtain the points of intersection
        intersection_points = ((1. - t[:, :, np.newaxis]) * polyline.v[polyline.e[which_es]]).sum(axis=1)
        #assert(np.all(self.distance(intersection_points) < 1e-10))
        return intersection_points

    def mesh_xsection(self, m, neighborhood=None):
        '''
        Takes a cross section of planar point cloud with a Mesh object.
        Ignore those points which intersect at a vertex - the probability of
        this event is small, and accounting for it complicates the algorithm.

        If 'neighborhood' is provided, use a KDTree to constrain the
        cross section to the closest connected component to 'neighborhood'.

        Params:
            - m:
                Mesh object
            - neigbhorhood:
                M x 3 np.array

        Returns a Polyline.

        TODO Return `None` instead of an empty polyline to signal no
        intersection.

        '''
        from blmath.geometry import Polyline

        # Step 1:
        #   Select those faces that intersect the plane, fs. Also construct
        #   the signed distances (fs_dists) and normalized signed distances
        #   (fs_norm_dists) for each such face.
        sgn_dists = self.signed_distance(m.v)
        which_fs = np.abs(np.sign(sgn_dists)[m.f].sum(axis=1)) != 3
        fs = m.f[which_fs]
        fs_dists = sgn_dists[fs]
        fs_norm_dists = np.sign(fs_dists)

        # Step 2:
        #   Build a length 3 array of edges es. Each es[i] is an np.array
        #   edge_pts of shape (fs.shape[0], 3). Each vector edge_pts[i, :]
        #   in edge_pts is an interesection of the plane with the
        #   fs[i], or [np.nan, np.nan, np.nan].
        es = []

        import itertools
        for i, j in itertools.combinations([0, 1, 2], 2):
            vi = m.v[fs[:, i]]
            vj = m.v[fs[:, j]]

            vi_dist = np.absolute(fs_dists[:, i])
            vj_dist = np.absolute(fs_dists[:, j])

            vi_norm_dist = fs_norm_dists[:, i]
            vj_norm_dist = fs_norm_dists[:, j]

            # use broadcasting to bulk traverse the edges
            t = vi_dist/(vi_dist + vj_dist)
            t = t[:, np.newaxis]

            edge_pts = t * vj + (1 - t) * vi

            # flag interior edge points that have the same sign with [nan, nan, nan].
            # note also that sum(trash.shape[0] for all i, j) == fs.shape[0], which holds.
            trash = np.nonzero(vi_norm_dist * vj_norm_dist == +1)[0]
            edge_pts[trash, :] = np.nan

            es.append(edge_pts)

        if any([edge.shape[0] == 0 for edge in es]):
            return Polyline(None)

        # Step 3:
        #   Build and return the verts v and edges e. Dump trash.
        hstacked = np.hstack(es)
        trash = np.isnan(hstacked)

        cleaned = hstacked[np.logical_not(trash)].reshape(fs.shape[0], 6)
        v1s, v2s = np.hsplit(cleaned, 2)

        v = np.empty((2 * v1s.shape[0], 3), dtype=v1s.dtype)
        v[0::2] = v1s
        v[1::2] = v2s

        if neighborhood is None:
            return Polyline(v, closed=True)

        # FIXME This e is incorrect.
        # Contains e.g.
        #   [0, 1], [2, 3], [4, 5], ...
        # But should contain
        #   [0, 1], [1, 2], [2, 3], ...
        # Leaving in place since the code below may depend on it.
        e = np.array([[i, i + 1] for i in xrange(0, v.shape[0], 2)])

        # Step 4 (optional - only if 'neighborhood' is provided):
        #   Build and return the ordered vertices cmp_v, and the
        #   edges cmp_e. Get connected components, use a KDTree
        #   to select the one with minimal distance to 'component'.
        #   Return the cmp_v and (re-indexed) edge mapping cmp_e.
        from scipy.spatial import cKDTree  # First thought this warning was caused by a pythonpath problem, but it seems more likely that the warning is caused by scipy import hackery. pylint: disable=no-name-in-module
        from scipy.sparse import csc_matrix
        from scipy.sparse.csgraph import connected_components

        from bodylabs.mesh.topology.connectivity import remove_redundant_verts

        # get rid of redundancies, or we
        # overcount connected components
        v, e = remove_redundant_verts(v, e)

        # connxns:
        #   sparse matrix of connected components.
        # ij:
        #   edges transposed
        # (connected_components needs these.)
        ij = np.vstack((
            e[:, 0].reshape(1, e.shape[0]),
            e[:, 1].reshape(1, e.shape[0]),
        ))
        connxns = csc_matrix((np.ones(len(e)), ij), shape=(len(v), len(v)))

        cmp_N, cmp_labels = connected_components(connxns)

        if cmp_N == 1:
            # no work to do, bail
            polyline = Polyline(v, closed=True)
            # This function used to return (v, e), so we include this
            # sanity check to make sure the edges match what Polyline uses.
            # np.testing.assert_all_equal(polyline.e, e)
            # Hmm, this fails.
            return polyline

        cmps = np.array([
            v[np.where(cmp_labels == cmp_i)]
            for cmp_i in range(cmp_N)
        ])

        kdtree = cKDTree(neighborhood)

        # cmp_N will not be large in
        # practice, so this loop won't hurt
        means = np.array([
            np.mean(kdtree.query(cmps[cmp_i])[0])
            for cmp_i in range(cmp_N)
        ])

        which_cmp = np.where(means == np.min(means))[0][0]

        # re-index edge mapping based on which_cmp. necessary
        # particularly when which_cmp is not contiguous in cmp_labels.
        which_vs = np.where(cmp_labels == which_cmp)[0]
        # which_es = np.logical_or(
        #     np.in1d(e[:, 0], which_vs),
        #     np.in1d(e[:, 1], which_vs),
        # )

        vmap = cmp_labels.astype(float)
        vmap[cmp_labels != which_cmp] = np.nan
        vmap[cmp_labels == which_cmp] = np.arange(which_vs.size)

        cmp_v = v[which_vs]                         # equivalently, cmp_v = cmp[which_cmp]
        # cmp_e = vmap[e[which_es]].astype(int)

        polyline = Polyline(cmp_v, closed=True)
        # This function used to return (cmp_v, cmp_e), so we include this
        # sanity check to make sure the edges match what Polyline uses.
        # Remove # this, and probably the code which creates 'vmap', when
        # we're more confident.
        # Hmm, this fails.
        # np.testing.assert_all_equal(polyline.e, cmp_e)
        return polyline


def main():
    import argparse
    from lace.mesh import Mesh

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='filepath to mesh', required=True)
    parser.add_argument('-c', '--cloud', help='display point cloud', required=False, default=False, action='store_true')
    parser.add_argument('-d', '--direction', help='direction of connected component',
                        choices=['N', 'S', 'E', 'W'], default=None, required=False)
    args = parser.parse_args()

    path_to_mesh = args.path
    mesh = Mesh(filename=path_to_mesh, vc='SteelBlue')

    point_on_plane = np.array([0., 1., 0.])

    n1 = np.array([0., 1., 0.])
    p1 = Plane(point_on_plane, n1)

    n2 = np.array([1., 0., 0.])
    p2 = Plane(point_on_plane, n2)

    n3 = np.array([1., 1., 0.])
    n3 /= np.linalg.norm(n3)
    p3 = Plane(point_on_plane, n3)

    n4 = np.array([-1., 1., 0.])
    n4 /= np.linalg.norm(n4)
    p4 = Plane(point_on_plane, n4)

    dirmap = {
        'N': [0., +100., 0.],
        'S': [0., -100., 0.],
        'E': [+100., 0., 0.],
        'W': [-100., 0., 0.],
        None: None,
    }

    neighborhood = dirmap[args.direction]
    if neighborhood != None:
        neighborhood = np.array([neighborhood])

    xs1 = p1.mesh_xsection(mesh, neighborhood=neighborhood)
    xs2 = p2.mesh_xsection(mesh, neighborhood=neighborhood)
    xs3 = p3.mesh_xsection(mesh, neighborhood=neighborhood)
    xs4 = p4.mesh_xsection(mesh, neighborhood=neighborhood)

    lines = [
        polyline.as_lines()
        for polyline in xs1, xs2, xs3, xs4
    ]

    if args.cloud:
        mesh.f = []

    from lace.meshviewer import MeshViewer
    mv = MeshViewer(keepalive=True)
    mv.set_dynamic_meshes([mesh], blocking=True)
    mv.set_dynamic_lines(lines)


if __name__ == '__main__':
    main()
