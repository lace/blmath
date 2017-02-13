import unittest
import numpy as np

class PartitionSegmentOldTests(unittest.TestCase):

    def setUp(self):
        from blmath.geometry.segment import partition_segment_old
        self.partition_segment_old = partition_segment_old

    def test_raises_exception_for_invalid_partition_size_type(self):
        p1 = np.array([0., 0., 0.])
        p2 = np.array([1., 0., 0.])

        self.assertRaises(TypeError, self.partition_segment_old, p1, p2, 'foobar')

    def test_raises_exception_for_invalid_partition_size_value(self):
        p1 = np.array([0., 0., 0.])
        p2 = np.array([1., 0., 0.])

        self.assertRaises(ValueError, self.partition_segment_old, p1, p2, 1)

    def test_returns_partition_for_odd_partition_size(self):
        p1 = np.array([0., 0., 0.])
        p2 = np.array([2., 0., 0.])

        partition_size = 4

        expected_partition_points = np.array([
            [.5, 0., 0.],
            [1., 0., 0.],
            [1.5, 0., 0.],
        ])

        np.testing.assert_array_almost_equal(
            self.partition_segment_old(p1, p2, partition_size),
            expected_partition_points,
            decimal=7
        )

    def test_returns_partition_points_for_even_partition_size(self):
        p1 = np.array([0., 0., 0.])
        p2 = np.array([1., 0., 0.])

        partition_size = 5

        expected_partition_points = np.array([
            [.2, 0., 0.],
            [.4, 0., 0.],
            [.6, 0., 0.],
            [.8, 0., 0.],
        ])

        np.testing.assert_array_almost_equal(
            self.partition_segment_old(p1, p2, partition_size),
            expected_partition_points,
            decimal=7
        )

    def test_returns_partition_points_in_oriented_order(self):
        p1 = np.array([0., 0., 0.])
        p2 = np.array([1., 0., 0.])

        partition_size = 5

        expected_partition_points = np.array([
            [.8, 0., 0.],
            [.6, 0., 0.],
            [.4, 0., 0.],
            [.2, 0., 0.],
        ])

        np.testing.assert_array_almost_equal(
            self.partition_segment_old(p2, p1, partition_size),
            expected_partition_points,
            decimal=7
        )

    def test_returns_partition_points_for_diagonal_segment(self):
        p1 = np.array([0., 0., 0.])
        p2 = np.array([1., 1., 0.])

        partition_size = 3

        dist = np.linalg.norm(p2 - p1)
        domain = [
            (1/3.0) * dist,
            (2/3.0) * dist,
        ]

        unit_direction = (p2 - p1) / dist

        expected_partition_points = np.array([
            p1 + scalar * unit_direction
            for scalar in domain
        ])

        np.testing.assert_array_almost_equal(
            self.partition_segment_old(p1, p2, partition_size),
            expected_partition_points,
            decimal=7
        )


class PartitionSegmentTests(unittest.TestCase):

    def setUp(self):
        from blmath.geometry.segment import partition_segment
        self.partition_segment = partition_segment

    def test_raises_exception_for_invalid_partition_size_type(self):
        p1 = np.array([0., 0., 0.])
        p2 = np.array([1., 0., 0.])

        self.assertRaises(TypeError, self.partition_segment, p1, p2, 'foobar')

    def test_raises_exception_for_invalid_partition_size_value(self):
        p1 = np.array([0., 0., 0.])
        p2 = np.array([1., 0., 0.])

        self.assertRaises(ValueError, self.partition_segment, p1, p2, 1)

    def test_returns_partition_for_odd_partition_size(self):
        p1 = np.array([0., 0., 0.])
        p2 = np.array([2., 0., 0.])

        partition_size = 5

        expected_partition_points = np.array([
            [0., 0., 0.],
            [.5, 0., 0.],
            [1., 0., 0.],
            [1.5, 0., 0.],
            [2., 0., 0.],
        ])

        np.testing.assert_array_almost_equal(
            self.partition_segment(p1, p2, partition_size),
            expected_partition_points,
            decimal=7
        )

    def test_returns_partition_points_for_even_partition_size(self):
        p1 = np.array([0., 0., 0.])
        p2 = np.array([1., 0., 0.])

        partition_size = 6

        expected_partition_points = np.array([
            [0., 0., 0.],
            [.2, 0., 0.],
            [.4, 0., 0.],
            [.6, 0., 0.],
            [.8, 0., 0.],
            [1., 0., 0.],
        ])

        np.testing.assert_array_almost_equal(
            self.partition_segment(p1, p2, partition_size),
            expected_partition_points,
            decimal=7
        )

    def test_returns_partition_omitting_endpoint(self):
        p1 = np.array([0., 0., 0.])
        p2 = np.array([1., 0., 0.])

        partition_size = 5

        expected_partition_points = np.array([
            [0., 0., 0.],
            [.2, 0., 0.],
            [.4, 0., 0.],
            [.6, 0., 0.],
            [.8, 0., 0.],
        ])

        np.testing.assert_array_almost_equal(
            self.partition_segment(p1, p2, partition_size, endpoint=False),
            expected_partition_points,
            decimal=7
        )


class AddPointsTests(unittest.TestCase):

    def setUp(self):
        from blmath.geometry.segment import partition
        self.partition = partition

    def test_adds_points_for_equal_length_line_segments(self):
        v = np.array([
            [0., 0., 0.],
            [1., 0., 0.],
            [1., 1., 0.],
            [1., 2., 0.],
            [1., 3., 0.],
        ])

        expected = np.array([
            [0.0, 0., 0.],
            [0.2, 0., 0.],
            [0.4, 0., 0.],
            [0.6, 0., 0.],
            [0.8, 0., 0.],
            [1., 0.0, 0.],
            [1., 0.2, 0.],
            [1., 0.4, 0.],
            [1., 0.6, 0.],
            [1., 0.8, 0.],
            [1., 1.0, 0.],
            [1., 1.2, 0.],
            [1., 1.4, 0.],
            [1., 1.6, 0.],
            [1., 1.8, 0.],
            [1., 2.0, 0.],
            [1., 2.2, 0.],
            [1., 2.4, 0.],
            [1., 2.6, 0.],
            [1., 2.8, 0.],
            [1., 3.0, 0.],
        ])

        np.testing.assert_array_almost_equal(self.partition(v), expected)

    def test_adds_points_for_nonequal_arbitrarily_oriented_line(self):
        v = np.array([
            [0., 0., 0.],
            [1., 0., 1.],
            [2., 0., 1.],
            [2., 2., 1.],
        ])

        expected = np.array([
            [.0, 0.0, .0],
            [.2, 0.0, .2],
            [.4, 0.0, .4],
            [.6, 0.0, .6],
            [.8, 0.0, .8],
            [1.0, 0., 1.],
            [1.2, 0., 1.],
            [1.4, 0., 1.],
            [1.6, 0., 1.],
            [1.8, 0., 1.],
            [2., 0.0, 1.],
            [2., 0.4, 1.],
            [2., 0.8, 1.],
            [2., 1.2, 1.],
            [2., 1.6, 1.],
            [2., 2.0, 1.],
        ])

        np.testing.assert_array_almost_equal(self.partition(v), expected)


class TestLineIntersect(unittest.TestCase):

    def test_line_intersect(self):
        from blmath.geometry.segment import line_intersect
        p0, q0 = np.array([[0., 3.], [4., 11.]])
        p1, q1 = np.array([[-2., 8.], [6., 4.]])
        result = line_intersect(p0, q0, p1, q1)
        np.testing.assert_array_equal(result, [1.6, 6.2])

    def test_line_intersect_duplicate_point(self):
        from blmath.geometry.segment import line_intersect
        p0, q0 = np.array([[0., 3.], [5., 5.]])
        p1, q1 = np.array([[5., 5.], [6., 4.]])
        result = line_intersect(p0, q0, p1, q1)
        np.testing.assert_array_equal(result, [5., 5.])


class TestLineIntersect3D(unittest.TestCase):

    def test_line_intersect3_with_colinear_lines(self):
        from blmath.geometry.segment import line_intersect3
        p0, q0 = np.array([[0., 1., 2.], [0., 10., 20.]])
        p1, q1 = np.array([[0., 2., 4.], [0., 4., 8.]])
        result = line_intersect3(p0, q0, p1, q1)
        self.assertIsNone(result)

    def test_line_intersect3_with_parallel_lines(self):
        from blmath.geometry.segment import line_intersect3
        p0, q0 = np.array([[0., 1., 2.], [0., 10., 20.]])
        p1, q1 = np.array([[1., 2., 3.], [1., 11., 21.]])
        result = line_intersect3(p0, q0, p1, q1)
        self.assertIsNone(result)

    def test_line_intersect3_with_degenerate_input_p(self):
        from blmath.geometry.segment import line_intersect3
        p0, q0 = np.array([[0., 1., 2.], [0., 10., 20.]])
        p1, q1 = np.array([[0., 1., 2.], [1., 11., 21.]])
        result = line_intersect3(p0, q0, p1, q1)
        np.testing.assert_array_equal(result, [0., 1., 2.])

    def test_line_intersect3_with_degenerate_input_q(self):
        from blmath.geometry.segment import line_intersect3
        p0, q0 = np.array([[0., 1., 2.], [0., 10., 20.]])
        p1, q1 = np.array([[1., 2., 3.], [0., 10., 20.]])
        result = line_intersect3(p0, q0, p1, q1)
        np.testing.assert_array_equal(result, [0., 10., 20.])

    def test_line_intersect3_example_1(self):
        # This example tests the codirectional cross product case
        from blmath.geometry.segment import line_intersect3
        p0, q0 = np.array([[5., 5., 4.], [10., 10., 6.]])
        p1, q1 = np.array([[5., 5., 5.], [10., 10., 3.]])
        result = line_intersect3(p0, q0, p1, q1)
        np.testing.assert_array_equal(result, [25./4, 25./4, 9./2])

    def test_line_intersect3_example_2(self):
        # This example tests the opposite direction cross product case
        from blmath.geometry.segment import line_intersect3
        p0, q0 = np.array([[5., 5., 4.], [10., 10., -6.]])
        p1, q1 = np.array([[5., 5., 5.], [10., 10., -3.]])
        result = line_intersect3(p0, q0, p1, q1)
        np.testing.assert_array_equal(result, [2.5, 2.5, 9])

    def test_line_intersect3_example_3(self):
        from blmath.geometry.segment import line_intersect3
        p0, q0 = np.array([[6., 8., 4.], [12., 15., 4.]])
        p1, q1 = np.array([[6., 8., 2.], [12., 15., 6.]])
        result = line_intersect3(p0, q0, p1, q1)
        np.testing.assert_array_equal(result, [9., 23./2, 4.])
