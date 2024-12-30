import unittest
from pyspark.sql import SparkSession
from DataQualityKit.encoding_module import EncodingConformity

class TestEncodingConformity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.master("local[1]").appName("EncodingConformityTest").getOrCreate()
        data = [("normal text",), ("text with special char é",), ("invalid \x80 text",)]
        cls.df = cls.spark.createDataFrame(data, ["text"])
        cls.enc = EncodingConformity()

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_check(self):
        results = self.enc.check(df=self.df, columns="text", target_encoding="UTF-8")
        self.assertIn("text", results)
        self.assertIn("current_encoding", results["text"])
        self.assertIn("detected_encodings", results["text"])
        self.assertIn("special_chars", results["text"])
        self.assertIn("invalid_chars", results["text"])
        self.assertIn("encoding_frequencies", results["text"])
        self.assertIn("conversion_possible", results["text"])
        self.assertIn("problematic_values", results["text"])
        self.assertIn("sample_violations", results["text"])
        self.assertIn("total_violations", results["text"])
        self.assertIn("violation_percentage", results["text"])

    def test_fix_convert(self):
        fixed_df = self.enc.fix(df=self.df, columns="text", strategy="convert", target_encoding="UTF-8")
        fixed_data = [row["text"] for row in fixed_df.collect()]
        self.assertIn("normal text", fixed_data)
        self.assertIn("text with special char é", fixed_data)
        self.assertNotIn("invalid \x80 text", fixed_data)

    def test_fix_remove(self):
        fixed_df = self.enc.fix(df=self.df, columns="text", strategy="remove")
        fixed_data = [row["text"] for row in fixed_df.collect()]
        self.assertIn("normal text", fixed_data)
        self.assertIn("text with special char é", fixed_data)
        self.assertIn("invalid  text", fixed_data)

    def test_fix_replace(self):
        fixed_df = self.enc.fix(df=self.df, columns="text", strategy="replace", replacement_char="?")
        fixed_data = [row["text"] for row in fixed_df.collect()]
        self.assertIn("normal text", fixed_data)
        self.assertIn("text with special char é", fixed_data)
        self.assertIn("invalid ? text", fixed_data)

    def test_fix_encode(self):
        fixed_df = self.enc.fix(df=self.df, columns="text", strategy="encode")
        fixed_data = [row["text"] for row in fixed_df.collect()]
        self.assertIn("normal text", fixed_data)
        self.assertIn("text with special char &#233;", fixed_data)
        self.assertIn("invalid &#128; text", fixed_data)

    def test_detect_encoding(self):
        detected_encodings = self.enc.detect_encoding(df=self.df, columns="text")
        self.assertIn("text", detected_encodings)
        self.assertEqual(detected_encodings["text"], "ascii")

if __name__ == "__main__":
    unittest.main()
