import unittest
import QualityControl as qc

class TestNullValues(unittest.TestCase):
    def setUp(self):
        self.null_values = qc.NullValues()
        self.data_with_nulls = [1, None, 2, None, 3]
        self.data_without_nulls = [1, 2, 3, 4, 5]

    def test_count_nulls(self):
        result = self.null_values.count_nulls(self.data_with_nulls)
        self.assertEqual(result, 2)

    def test_count_nulls_no_nulls(self):
        result = self.null_values.count_nulls(self.data_without_nulls)
        self.assertEqual(result, 0)

    def test_has_nulls(self):
        result = self.null_values.has_nulls(self.data_with_nulls)
        self.assertTrue(result)

    def test_has_nulls_no_nulls(self):
        result = self.null_values.has_nulls(self.data_without_nulls)
        self.assertFalse(result)

    def test_remove_nulls(self):
        result = self.null_values.remove_nulls(self.data_with_nulls)
        self.assertEqual(result, [1, 2, 3])

if __name__ == '__main__':
    unittest.main()
