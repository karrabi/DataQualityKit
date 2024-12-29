import unittest
from pyspark.sql import SparkSession
import QualityControl as qc

class TestRangeValidity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.master("local[1]").appName("RangeValidityTest").getOrCreate()
        cls.df = cls.spark.createDataFrame([
            (1, 25, 1000.0),
            (2, 35, 2000.0),
            (3, 45, 3000.0),
            (4, 55, 4000.0),
            (5, 65, 5000.0),
            (6, 75, 6000.0),
            (7, 85, 7000.0),
            (8, 95, 8000.0),
            (9, 105, 9000.0),
            (10, 115, 10000.0)
        ], ["id", "age", "salary"])

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_check(self):
        rv = qc.RangeValidity()
        result = rv.check(self.df, columns=['age', 'salary'], boundaries={'age': {'min': 20, 'max': 100}})
        self.assertIn('age', result)
        self.assertIn('salary', result)
        self.assertEqual(result['age']['violations']['below_min'], 0)
        self.assertEqual(result['age']['violations']['above_max'], 2)

    def test_fix_cap(self):
        rv = qc.RangeValidity()
        fixed_df = rv.fix(self.df, columns=['age'], strategy='cap', boundaries={'age': {'min': 20, 'max': 100}})
        result = fixed_df.select("age").collect()
        self.assertEqual(result[0]['age'], 25)
        self.assertEqual(result[-1]['age'], 100)

    def test_fix_remove(self):
        rv = qc.RangeValidity()
        fixed_df = rv.fix(self.df, columns=['age'], strategy='remove', boundaries={'age': {'min': 20, 'max': 100}})
        result = fixed_df.count()
        self.assertEqual(result, 8)

    def test_fix_transform(self):
        rv = qc.RangeValidity()
        fixed_df = rv.fix(self.df, columns=['salary'], strategy='transform', transform_method='log')
        result = fixed_df.select("salary").collect()
        self.assertAlmostEqual(result[0]['salary'], 6.907755, places=5)

    def test_suggest_boundaries(self):
        rv = qc.RangeValidity()
        suggested = rv.suggest_boundaries(self.df, columns=['age'], method='statistical')
        self.assertIn('age', suggested)
        self.assertGreaterEqual(suggested['age']['min'], 25)
        self.assertLessEqual(suggested['age']['max'], 115)

if __name__ == '__main__':
    unittest.main()
