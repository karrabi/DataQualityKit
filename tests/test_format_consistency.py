import unittest
from pyspark.sql import SparkSession
from DataQualityKit.format_consistency import FormatConsistency

class TestFormatConsistency(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.master("local[1]").appName("FormatConsistencyTest").getOrCreate()
        cls.fc = FormatConsistency()

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_check_date_format(self):
        data = [("2021-01-01",), ("2021/01/01",), ("01-01-2021",), ("invalid_date",)]
        df = self.spark.createDataFrame(data, ["birth_date"])
        format_types = {"birth_date": "date"}
        results = self.fc.check(df, columns="birth_date", format_types=format_types)
        self.assertIn("birth_date", results)
        self.assertEqual(results["birth_date"]["violations"]["total_count"], 4)
        self.assertEqual(results["birth_date"]["violations"]["invalid_format"], 3)

    def test_fix_date_format(self):
        data = [("2021-01-01",), ("2021/01/01",), ("01-01-2021",), ("invalid_date",)]
        df = self.spark.createDataFrame(data, ["birth_date"])
        format_types = {"birth_date": "date"}
        target_formats = {"date": "yyyy-MM-dd"}
        fixed_df = self.fc.fix(df, columns="birth_date", format_types=format_types, target_formats=target_formats)
        fixed_data = [row["birth_date"] for row in fixed_df.collect()]
        self.assertIn("2021-01-01", fixed_data)
        self.assertNotIn("invalid_date", fixed_data)

    def test_check_phone_format(self):
        data = [("123-456-7890",), ("(123) 456-7890",), ("1234567890",), ("invalid_phone",)]
        df = self.spark.createDataFrame(data, ["phone"])
        format_types = {"phone": "phone"}
        results = self.fc.check(df, columns="phone", format_types=format_types)
        self.assertIn("phone", results)
        self.assertEqual(results["phone"]["violations"]["total_count"], 4)
        self.assertEqual(results["phone"]["violations"]["invalid_format"], 1)

    def test_fix_phone_format(self):
        data = [("123-456-7890",), ("(123) 456-7890",), ("1234567890",), ("invalid_phone",)]
        df = self.spark.createDataFrame(data, ["phone"])
        format_types = {"phone": "phone"}
        target_formats = {"phone": "+1-XXX-XXX-XXXX"}
        fixed_df = self.fc.fix(df, columns="phone", format_types=format_types, target_formats=target_formats)
        fixed_data = [row["phone"] for row in fixed_df.collect()]
        self.assertIn("+1-123-456-7890", fixed_data)
        self.assertNotIn("invalid_phone", fixed_data)

    def test_add_custom_pattern(self):
        self.fc.add_pattern(name="product_code", pattern=r'^[A-Z]{2}-\d{4}$', description="Product code format: XX-9999")
        self.assertIn("product_code", self.fc._custom_patterns)
        self.assertEqual(self.fc._custom_patterns["product_code"]["pattern"], r'^[A-Z]{2}-\d{4}$')

    def test_parse_components(self):
        data = [("123 Main St, Springfield, IL, 62701",)]
        df = self.spark.createDataFrame(data, ["address"])
        parsed_df = self.fc.parse_components(df, column="address", format_type="address", output_columns=["street", "city", "state", "zip"])
        parsed_data = parsed_df.collect()[0]
        self.assertEqual(parsed_data["street"], "123 Main St")
        self.assertEqual(parsed_data["city"], "Springfield")
        self.assertEqual(parsed_data["state"], "IL")
        self.assertEqual(parsed_data["zip"], "62701")

if __name__ == "__main__":
    unittest.main()
