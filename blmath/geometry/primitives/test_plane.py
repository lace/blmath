import math
import unittest
import numpy as np
from blmath.geometry import Plane
from blmath.numerics import vx


class DistanceToPlaneTests(unittest.TestCase):

    def test_returns_signed_distances_for_xz_plane_at_origin(self):
        # x-z plane
        normal = np.array([0., 1., 0.])
        sample = np.array([0., 0., 0.])

        plane = Plane(sample, normal)

        pts = np.array([
            [500., 502., 503.],
            [-500., -501., -503.],
        ])

        expected = np.array([502., -501.])

        np.testing.assert_array_equal(expected, plane.signed_distance(pts))

    def test_returns_unsigned_distances_for_xz_plane_at_origin(self):
        # x-z plane
        normal = np.array([0., 1., 0.])
        sample = np.array([0., 0., 0.])

        plane = Plane(sample, normal)

        pts = np.array([
            [500., 502., 503.],
            [-500., -501., -503.],
        ])

        expected = np.array([502., 501.])

        np.testing.assert_array_equal(expected, plane.distance(pts))

    def test_returns_signed_distances_for_diagonal_plane(self):
        # diagonal plane @ origin - draw a picture!
        normal = np.array([1., 1., 0.])
        normal /= np.linalg.norm(normal)
        sample = np.array([1., 1., 0.])

        plane = Plane(sample, normal)

        pts = np.array([
            [425., 425., 25.],
            [-500., -500., 25.],
        ])

        expected = np.array([
            math.sqrt(2*(425.-1.)**2),
            -math.sqrt(2*(500.+1.)**2),
        ])

        np.testing.assert_array_almost_equal(expected, plane.signed_distance(pts))

    def test_returns_unsigned_distances_for_diagonal_plane_at_origin(self):
        # diagonal plane @ origin - draw a picture!
        normal = np.array([1., 1., 0.])
        normal /= np.linalg.norm(normal)

        sample = np.array([0., 0., 0.])

        plane = Plane(sample, normal)

        pts = np.array([
            [425., 425., 25.],
            [-500., -500., 25.],
        ])

        expected = np.array([
            math.sqrt(2*(425.**2)),
            math.sqrt(2*(500.**2))
        ])

        np.testing.assert_array_almost_equal(expected, plane.distance(pts))

    def test_returns_sign_for_diagonal_plane(self):
        # diagonal plane @ origin - draw a picture!
        normal = np.array([1., 1., 0.])
        normal /= np.linalg.norm(normal)
        sample = np.array([1., 1., 0.])

        plane = Plane(sample, normal)

        pts = np.array([
            [425., 425., 25.],
            [-500., -500., 25.],
        ])

        sign = plane.sign(pts)

        expected = np.array([1., -1.])
        np.testing.assert_array_equal(sign, expected)

class TestCanonicalPoint(unittest.TestCase):

    def test_canonical_point(self):
        normal = np.array([1., 1., 0.])
        normal /= np.linalg.norm(normal)

        sample = np.array([0., 0., 0.])

        plane = Plane(sample, normal)

        np.testing.assert_array_equal(plane.canonical_point, np.array([0., 0., 0.]))

        plane = Plane(sample, -normal)

        np.testing.assert_array_equal(plane.canonical_point, np.array([0., 0., 0.]))

        normal = np.array([1., 7., 9.])
        normal /= np.linalg.norm(normal)

        plane = Plane(sample, normal)

        np.testing.assert_array_equal(plane.canonical_point, np.array([0., 0., 0.]))

        plane = Plane(sample, -normal)

        np.testing.assert_array_equal(plane.canonical_point, np.array([0., 0., 0.]))

        normal = np.array([1., 0., 0.])
        normal /= np.linalg.norm(normal)

        sample = np.array([3., 10., 20.])

        plane = Plane(sample, normal)

        np.testing.assert_array_equal(plane.canonical_point, np.array([3, 0., 0.]))

        plane = Plane(sample, -normal)

        np.testing.assert_array_equal(plane.canonical_point, np.array([3, 0., 0.]))

        normal = np.array([1., 1., 1.])
        normal /= np.linalg.norm(normal)

        sample = np.array([1., 2., 10.])

        plane = Plane(sample, normal)

        np.testing.assert_array_almost_equal(plane.canonical_point, np.array([4.333333, 4.333333, 4.333333]))

        plane = Plane(sample, -normal)

        np.testing.assert_array_almost_equal(plane.canonical_point, np.array([4.333333, 4.333333, 4.333333]))

class TestProjectPoint(unittest.TestCase):

    def test_project_point(self):
        plane = Plane(point_on_plane=np.array([0, 10, 0]), unit_normal=vx.basis.y)

        point = np.array([10, 20, -5])

        expected = np.array([10, 10, -5])

        np.testing.assert_array_equal(plane.project_point(point), expected)

class TestPlaneFromPoints(unittest.TestCase):

    def test_plane_from_points(self):
        points = np.array([
            [1, 1, 1],
            [-1, 1, 0],
            [2, 0, 3],
        ])
        plane = Plane.from_points(*points)

        a, b, c, d = plane.equation

        plane_equation_test = [a * x + b * y + c * z + d for x, y, z in points]
        np.testing.assert_array_equal(plane_equation_test, [0, 0, 0])

        projected_points = [plane.project_point(p) for p in points]
        np.testing.assert_array_almost_equal(projected_points, points)

    def test_plane_from_points_order(self):
        points = np.array([
            [1, 0, 0],
            [0, math.sqrt(1.25), 0],
            [-1, 0, 0],
        ])
        plane = Plane.from_points(*points)

        expected_v = np.array([0, 0, 1])
        np.testing.assert_array_equal(plane.normal, expected_v)

    def test_plane_from_points_and_vector(self):
        p1 = np.array([1, 5, 7])
        p2 = np.array([-2, -2, -2])
        v = np.array([1, 0, -1])
        plane = Plane.from_points_and_vector(p1, p2, v)

        points = [p1, p2]
        projected_points = [plane.project_point(p) for p in points]
        np.testing.assert_array_almost_equal(projected_points, points)

        self.assertEqual(np.dot(v, plane.normal), 0)

    def test_fit_from_points(self):
        # Set up a collection of points in the X-Y plane.
        np.random.seed(0)
        points = np.hstack([
            np.random.random((100, 2)),
            np.zeros(100).reshape(-1, 1)
        ])
        plane = Plane.fit_from_points(points)

        # The normal vector should be closely aligned with the Z-axis.
        z_axis = np.array([0., 0., 1.])
        angle = np.arccos(
            np.dot(plane.normal, z_axis) /
            np.linalg.norm(plane.normal)
        )
        self.assertTrue(angle % np.pi < 1e-6)
