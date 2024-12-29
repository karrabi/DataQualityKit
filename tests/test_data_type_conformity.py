import unittest
from pyspark.sql import SparkSession
import QualityControl as qc

class TestDataTypeConformity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.master("local").appName("DataTypeConformityTest").getOrCreate()
        cls.dtc = qc.DataTypeConformity()

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def setUp(self):
        self.df = self.spark.createDataFrame([
            (1, "2021-01-01", "10.5"),
            (2, "2021-01-02", "20.5"),
            (3, "invalid_date", "invalid_double"),
            (4, None, None)
        ], ["id", "date", "value"])

    def test_check(self):
        result = self.dtc.check(self.df, columns=["date", "value"], expected_types={"date": "date", "value": "double"})
        self.assertIn("date", result)
        self.assertIn("value", result)
        self.assertEqual(result["date"]["expected_type"], "date")
        self.assertEqual(result["value"]["expected_type"], "double")
        self.assertGreater(result["date"]["violation_percentage"], 0)
        self.assertGreater(result["value"]["violation_percentage"], 0)

    def test_fix_convert(self):
        fixed_df = self.dtc.fix(self.df, columns=["value"], target_types={"value": "double"}, strategy="convert")
        self.assertEqual(fixed_df.schema["value"].dataType, DoubleType())

    def test_fix_parse(self):
        fixed_df = self.dtc.fix(self.df, columns=["date"], strategy="parse", string_pattern=r"(\d{4})-(\d{2})-(\d{2})")
        self.assertTrue("date" in fixed_df.columns)

    def test_fix_clean(self):
        fixed_df = self.dtc.fix(self.df, columns=["value"], strategy="clean")
        self.assertTrue("value" in fixed_df.columns)

    def test_fix_split(self):
        fixed_df = self.dtc.fix(self.df, columns=["value"], strategy="split", split_columns=True, string_pattern=r"\.")
        self.assertTrue("value_part1" in fixed_df.columns)
        self.assertTrue("value_part2" in fixed_df.columns)

    def test_infer_types(self):
        inferred_types = self.dtc.infer_types(self.df, columns=["id", "date", "value"])
        self.assertEqual(inferred_types["id"], "integer")
        self.assertEqual(inferred_types["date"], "string")
        self.assertEqual(inferred_types["value"], "string")

if __name__ == '__main__':
    unittest.main()
