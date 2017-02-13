import unittest
import numpy as np
from blmath.geometry.apex import apex
from blmath.geometry.apex import farthest

class TestApex(unittest.TestCase):

    def test_apex(self):
        points = np.array([
            [-0.97418884, -0.79808404, -0.18545491],
            [0.60675227, 0.32673201, -0.20369793],
            [0.67040405, 0.19267665, -0.56983579],
            [-0.68038753, -0.90011588, 0.4649872],
            [-0.62813991, -0.23947753, 0.07933854],
            [0.26348356, 0.23701114, -0.38230596],
            [0.08302473, 0.2784907, 0.09308946],
            [0.58695587, -0.33253376, -0.33493078],
            [-0.39221704, -0.45240036, 0.25284163],
            [0.46270635, -0.3865265, -0.98106526],
        ])

        np.testing.assert_array_equal(apex(points, [1, 0, 0]), [0.67040405, 0.19267665, -0.56983579])
        np.testing.assert_array_equal(apex(points, [-1, 0, 0]), [-0.97418884, -0.79808404, -0.18545491])
        np.testing.assert_array_equal(apex(points, [0, 1, 0]), [0.60675227, 0.32673201, -0.20369793])
        np.testing.assert_array_equal(apex(points, [0, -1, 0]), [-0.68038753, -0.90011588, 0.4649872])
        np.testing.assert_array_equal(apex(points, [0, 0, 1]), [-0.68038753, -0.90011588, 0.4649872])
        np.testing.assert_array_equal(apex(points, [0, 0, -1]), [0.46270635, -0.3865265, -0.98106526])

        v = [1/3 ** .5] * 3
        expected = points[np.argmax(points.sum(axis=1))]
        np.testing.assert_array_equal(apex(points, v), expected)

        # Test non-normalized too.
        np.testing.assert_array_equal(apex(points, [1, 1, 1]), expected)

    def test_farthest(self):
        from_point = np.array([-1., 0., 0.])

        to_points = np.array([
            [1., -2., -3.],
            [-1., -20., -30.],
        ])

        point, index = farthest(from_point, to_points)

        np.testing.assert_array_equal(point, to_points[1])
        np.testing.assert_array_equal(index, 1)
