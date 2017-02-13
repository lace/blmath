import unittest
import math
import numpy as np

from blmath.numerics.linalg.isomorphism import isomorphism
from blmath.numerics.linalg.isomorphism import DimensionalityError, LinearDependenceError

class TestIsomorphism(unittest.TestCase):

    def setUp(self):
        self.isomorphism = isomorphism
        self.decimal_accuracy = 7

    def test_raises_error_for_dependent_frame(self):
        self.assertRaises(
            LinearDependenceError,
            self.isomorphism,
            np.array([[1., 1.], [2., 2.]]),
            np.array([[1., 0.], [0., 1.]]),
        )

    def test_raises_error_for_mismatched_dimensions(self):
        self.assertRaises(
            DimensionalityError,
            self.isomorphism,
            np.array([[1]]),
            np.array([[1., 0.], [0., 1.]]),
        )

    def test_returns_correct_matrix(self):
        # 45 degree rotation

        frame1 = np.array([
            [1./math.sqrt(2), 1./math.sqrt(2)],
            [-1./math.sqrt(2), 1./math.sqrt(2)]
        ])

        frame2 = np.array([
            [0., 1.],
            [-1., 0.]
        ])

        T = self.isomorphism(frame1, frame2)

        np.testing.assert_array_almost_equal(
            T,
            np.array([
                [1./math.sqrt(2), -1./math.sqrt(2)],
                [1./math.sqrt(2), 1./math.sqrt(2)],
            ]),
            decimal=self.decimal_accuracy
        )
