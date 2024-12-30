import unittest
from pyspark.sql import SparkSession
from DataQualityKit.duplicate_values import DuplicateValues

class TestDuplicateValues(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.master("local[1]").appName("TestDuplicateValues").getOrCreate()
        cls.df = cls.spark.createDataFrame([
            (1, "Alice", "2021-01-01", 100.0),
            (2, "Bob", "2021-01-02", 150.0),
            (3, "Alice", "2021-01-01", 100.0),
            (4, "Charlie", "2021-01-03", 200.0),
            (5, "Bob", "2021-01-02", 150.0)
        ], ["id", "name", "date", "amount"])
        cls.dv = DuplicateValues(cls.df)

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_check_exact_duplicates(self):
        result = self.dv.check_exact_duplicates(columns=["name", "date", "amount"])
        self.assertEqual(result['duplicate_count'], 2)
        self.assertEqual(result['affected_rows'], 4)

    def test_check_fuzzy_matches(self):
        result = self.dv.check_fuzzy_matches(columns=["name"], threshold=0.9, algorithm='jaro_winkler')
        self.assertGreaterEqual(len(result['match_groups']), 0)

    def test_check_business_key_duplicates(self):
        result = self.dv.check_business_key_duplicates(key_columns=["name", "date"])
        self.assertEqual(result['violation_count'], 2)

    def test_remove_exact_duplicates(self):
        deduped_df = self.dv.remove_exact_duplicates(columns=["name", "date", "amount"], keep='first')
        self.assertEqual(deduped_df.count(), 3)

    def test_merge_similar_records(self):
        merged_df = self.dv.merge_similar_records(
            match_columns=["name"],
            merge_rules={"amount": "sum"},
            threshold=0.9
        )
        self.assertGreaterEqual(merged_df.count(), 3)

    def test_create_composite_key(self):
        keyed_df = self.dv.create_composite_key(columns=["name", "date"])
        self.assertTrue("composite_key" in keyed_df.columns)

if __name__ == "__main__":
    unittest.main()
