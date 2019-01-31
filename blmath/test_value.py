import unittest
import numpy as np
from blmath.value import Value
from blmath.util import json

class TestValueClass(unittest.TestCase):
    def test_value_initializes_correctly(self):
        with self.assertRaises(TypeError):
            _ = Value()  # It's a failure test. pylint: disable=no-value-for-parameter
        with self.assertRaises(ValueError):
            _ = Value(13, 'mugglemeters')
        with self.assertRaises(ValueError):
            _ = Value(None, 'mugglemeters')
        with self.assertRaises(ValueError):
            _ = Value(None, 'mm')

        x = Value(13, 'mm')
        self.assertIsInstance(x, Value)
        self.assertFalse(isinstance(x, float))
        self.assertTrue(hasattr(x, 'units'))
        self.assertEqual(x.units, 'mm')

    def test_value_initializes_with_unitless_values(self):
        _ = Value(1, None)

    def test_exception_raised_if_value_is_nonsense(self):
        with self.assertRaises(ValueError):
            Value('x', 'mm')
        v = Value(0, 'mm')
        with self.assertRaises(ValueError):
            v.value = 'x'

    def test_conversions(self):
        x = Value(25, 'cm')
        self.assertAlmostEqual(x.convert('m'), 0.25)
        self.assertAlmostEqual(x.convert('cm'), 25)
        self.assertAlmostEqual(x.convert('mm'), 250)
        self.assertAlmostEqual(x.convert('in'), 9.8425197) # from google
        self.assertAlmostEqual(x.convert('ft'), 0.82021) # from google
        self.assertAlmostEqual(x.convert('fathoms'), 0.136701662) # from google
        self.assertAlmostEqual(x.convert('cubits'), 0.546806649) # from google
        x = Value(1, 'kg')
        self.assertAlmostEqual(x.convert('kg'), 1)
        self.assertAlmostEqual(x.convert('g'), 1000)
        self.assertAlmostEqual(x.convert('lbs'), 2.20462) # from google
        self.assertAlmostEqual(x.convert('stone'), 0.157473) # from google
        x = Value(90, 'deg')
        self.assertAlmostEqual(x.convert('rad'), np.pi/2)
        self.assertAlmostEqual(x.convert('deg'), 90)
        x = Value(30, 'min')
        self.assertAlmostEqual(x.convert('sec'), 30*60)
        self.assertAlmostEqual(x.convert('minutes'), 30)
        self.assertAlmostEqual(x.convert('hours'), 0.5)
        x = Value(2, 'days')
        self.assertAlmostEqual(x.convert('min'), 2*24*60)
        x = Value(1, 'year')
        self.assertAlmostEqual(x.convert('min'), 525948.48)

    def test_value_does_not_convert_unitless_values(self):
        x = Value(1, None)
        with self.assertRaises(ValueError):
            x.convert('kg')

    def test_behaves_like_tuple(self):
        x = Value(25, 'cm')
        self.assertAlmostEqual(x[0], 25)
        self.assertAlmostEqual(x[1], 'cm')

    def test_easy_conversion_properties(self):
        x = Value(25, 'cm')
        self.assertAlmostEqual(x.m, 0.25)
        self.assertAlmostEqual(x.cm, 25)
        self.assertAlmostEqual(x.mm, 250)
        with self.assertRaises(AttributeError):
            _ = x.mugglemeters

    def test_comparison(self):
        self.assertTrue(Value(25, 'cm') == Value(25, 'cm'))
        self.assertTrue(Value(25, 'cm') == Value(250, 'mm'))
        self.assertTrue(Value(25, 'cm') > Value(240, 'mm'))
        self.assertTrue(Value(25, 'cm') < Value(260, 'mm'))
        self.assertTrue(Value(25, 'cm') != Value(260, 'mm'))
        # When comparison is to a number, we assume that the units are the same as ours
        self.assertTrue(Value(25, 'cm') == 25)

    def test_multiplication(self):
        x = Value(25, 'cm')
        self.assertEqual(x * 2, 50)
        self.assertIsInstance(x * 2, Value)
        self.assertEqual(2 * x, 50)
        self.assertIsInstance(2 * x, Value)
        with self.assertRaises(ValueError):
            # This would be cm^2; for our present purposes, multpying Value*Value is an error
            _ = x * x
        self.assertEqual([1, 2, 3] * x, [Value(25, 'cm'), Value(50, 'cm'), Value(75, 'cm')])
        for y in [1, 2, 3] * x:
            self.assertIsInstance(y, Value)

    def test_addition_and_subtraction(self):
        x = Value(25, 'cm')
        self.assertEqual(x + 2, Value(27, 'cm'))
        self.assertIsInstance(x + 2, Value)
        self.assertEqual(2 + x, Value(27, 'cm'))
        self.assertIsInstance(2 + x, Value)
        self.assertEqual(x - 2, Value(23, 'cm'))
        self.assertIsInstance(x - 2, Value)
        # Note that although this is sort of poorly defined, we need to support this case in order to make comparisons easy
        self.assertEqual(2 - x, Value(-23, 'cm'))
        self.assertIsInstance(2 - x, Value)
        self.assertEqual(x + Value(25, 'cm'), Value(50, 'cm'))
        self.assertIsInstance(x + Value(25, 'cm'), Value)
        self.assertEqual(x + Value(5, 'mm'), Value(255, 'mm'))
        self.assertEqual([1, 2, 3] + x, [Value(26, 'cm'), Value(27, 'cm'), Value(28, 'cm')])
        for y in [1, 2, 3] + x:
            self.assertIsInstance(y, Value)

    def test_other_numeric_methods(self):
        x = Value(25, 'cm')
        self.assertEqual(str(x), "25.000000 cm")
        self.assertEqual(x / 2, Value(12.5, 'cm'))
        self.assertIsInstance(x / 2, Value)
        self.assertEqual(x / Value(1, 'cm'), Value(25, None))
        self.assertEqual(x / Value(1, 'm'), Value(0.25, None))
        self.assertEqual(x // 2, Value(12, 'cm'))
        self.assertEqual(x // Value(1, 'cm'), Value(25, None))
        self.assertEqual(x // Value(1, 'm'), Value(0, None))
        self.assertIsInstance(x // 2, Value)
        with self.assertRaises(AttributeError):
            _ = x % 2
        with self.assertRaises(AttributeError):
            _ = x ** 2
        with self.assertRaises(ValueError):
            _ = 2 / x
        self.assertEqual(+x, Value(25, 'cm'))
        self.assertIsInstance(+x, Value)
        self.assertEqual(-x, Value(-25, 'cm'))
        self.assertIsInstance(-x, Value)
        self.assertEqual(abs(Value(-25, 'cm')), Value(25, 'cm'))
        self.assertIsInstance(abs(Value(-25, 'cm')), Value)

    def test_cast(self):
        x = Value(25, 'cm')
        self.assertEqual(float(x), x.value)
        self.assertEqual(int(x), x.value)
        self.assertEqual(int(Value(25.5, 'cm')), 25)

    def test_make_numpy_array_out_of_values(self):
        x = np.array([Value(i, 'cm') for i in range(10)])
        res = np.sum(x)
        self.assertIsInstance(res, Value)

class TestValueSerialization(unittest.TestCase):

    def test_basic_serialization(self):
        x = Value(25, 'cm')
        x_json = json.dumps(x)

        self.assertEqual(x_json, '{"__value__": {"units": "cm", "value": 25.0}}')
        x_obj = json.loads(x_json)
        self.assertEqual(x, x_obj)

    def test_complex_serialization(self):
        x = {str(i): Value(i, 'cm') for i in range(10)}
        x_json = json.dumps(x)
        x_obj = json.loads(x_json)
        self.assertEqual(x, x_obj)

class TestValueDeserialization(unittest.TestCase):

    def test_loads(self):
        x_str = json.dumps({'__value__': {'value': 25.0, 'units': 'cm'}})
        x = json.loads(x_str)
        self.assertEqual(x.value, 25.0)
        self.assertEqual(x.units, 'cm')

    def test_from_json(self):
        x = Value.from_json({'__value__': {'value': 25.0, 'units': 'cm'}})
        self.assertEqual(x.value, 25.0)
        self.assertEqual(x.units, 'cm')

if __name__ == '__main__':
    unittest.main()
