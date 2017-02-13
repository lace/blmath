import unittest
import warnings
import numpy as np


class TestOperations(unittest.TestCase):
    def test_zero_safe_divide(self):
        from blmath.numerics.operations import zero_safe_divide

        numerator = np.ones((5, 5))
        numerator[3, 3] = 0.

        denominator = np.ones((5, 5))
        denominator[2, 2] = 0.
        denominator[3, 3] = 0.

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            true_divide = np.true_divide(numerator, denominator)
        safe_divide = zero_safe_divide(numerator, denominator)
        self.assertTrue(np.isinf(true_divide[2, 2]))
        self.assertEqual(safe_divide[2, 2], 0.)
        self.assertTrue(np.isnan(true_divide[3, 3]))
        self.assertEqual(safe_divide[3, 3], 0.)
