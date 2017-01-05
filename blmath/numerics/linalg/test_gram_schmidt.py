import unittest
import math
import numpy as np

from blmath.numerics.linalg import gram_schmidt

class TestGramSchmidt(unittest.TestCase):

    def setUp(self):
        self.gs = gram_schmidt
        self.decimal_accuracy = 7

    def test_orthogonalizes(self):
        v = np.array([[1., 1., 1.], [2., 1., 0.], [5., 1., 3.]])

        result = self.gs.orthogonalize(v)

        self.assertEqual(np.dot(result[0], result[1]), 0)
        self.assertEqual(np.dot(result[1], result[2]), 0)
        self.assertEqual(np.dot(result[0], result[2]), 0)

    def test_orthormalizes(self):
        v = np.array([[1., 1., 1.], [2., 1., 0.], [5., 1., 3.]])

        result = self.gs.orthonormalize(v)

        self.assertEqual(np.dot(result[0], result[1]), 0)
        self.assertEqual(np.dot(result[1], result[2]), 0)
        self.assertEqual(np.dot(result[0], result[2]), 0)

        self.assertAlmostEqual(np.linalg.norm(result[0]), 1, places=self.decimal_accuracy)
        self.assertAlmostEqual(np.linalg.norm(result[1]), 1, places=self.decimal_accuracy)
        self.assertAlmostEqual(np.linalg.norm(result[2]), 1, places=self.decimal_accuracy)

    def test_orthormalizes_given_indices(self):
        v = np.array([[1., 1., 1.], [2., 1., 0.], [5., 1., 3.]])

        result = self.gs.orthonormalize(v, indices=[1])

        self.assertEqual(np.dot(result[0], result[1]), 0)
        self.assertEqual(np.dot(result[1], result[2]), 0)
        self.assertEqual(np.dot(result[0], result[2]), 0)

        self.assertNotEqual(np.linalg.norm(result[0]), 1)
        self.assertAlmostEqual(np.linalg.norm(result[1]), 1, places=self.decimal_accuracy)
        self.assertNotEqual(np.linalg.norm(result[2]), 1)

    def test_returns_np_array_with_correct_shape(self):
        v = np.array([[1., 1., 1.], [2., 1., 0.], [5., 1., 3.]])

        result = self.gs.orthogonalize(v)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (3, 3))

        result = self.gs.orthonormalize(v)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (3, 3))

    def test_gram_schmidt_implemented(self):
        # indirectly test correct algorithm by testing exact result. more effort not worth it.
        # see http://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process#Example
        v = np.array([[3, 1], [2, 2]], dtype=float)

        orthogonalized = self.gs.orthogonalize(v)
        np.testing.assert_array_almost_equal(
            orthogonalized,
            np.array([
                [3., 1.],
                [-2/5., 6/5.],
            ]),
            decimal=self.decimal_accuracy
        )

        orthonormalized = self.gs.orthonormalize(v)
        np.testing.assert_array_almost_equal(
            orthonormalized,
            np.array([
                [3/math.sqrt(10), 1/math.sqrt(10)],
                [-1/math.sqrt(10), 3/math.sqrt(10)],
            ]),
            decimal=self.decimal_accuracy
        )
