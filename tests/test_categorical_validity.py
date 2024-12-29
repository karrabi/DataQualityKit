import unittest
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from DataQualityKit.categorical_validity import CategoricalValidity

class TestCategoricalValidity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.master("local[1]").appName("CategoricalValidityTest").getOrCreate()
        data = [
            ("Electronics", "Laptop"),
            ("Electronics", "Desktop"),
            ("Clothing", "Shirt"),
            ("Clothing", "Pants"),
            ("Books", "Fiction"),
            ("Books", "Non-Fiction"),
            ("Books", "Fiction"),
            ("Electronics", "Tablet"),
            ("Clothing", "Shirt"),
            ("Books", "Fiction")
        ]
        cls.df = cls.spark.createDataFrame(data, ["category", "subcategory"])
        cls.cv = CategoricalValidity(cls.df)

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_check_category_validity(self):
        results = self.cv.check_category_validity(
            column='category',
            valid_categories=['Electronics', 'Clothing', 'Books'],
            frequency_threshold=0.1
        )
        self.assertIn('invalid_categories', results)
        self.assertIn('category_frequencies', results)
        self.assertIn('rare_categories', results)
        self.assertIn('case_variations', results)
        self.assertIn('potential_misspellings', results)
        self.assertIn('statistics', results)
        self.assertIn('suggestions', results)

    def test_check_spelling_variants(self):
        results = self.cv.check_spelling_variants(
            column='subcategory',
            reference_values=['Laptop', 'Desktop', 'Shirt', 'Pants', 'Fiction', 'Non-Fiction'],
            similarity_threshold=0.9
        )
        self.assertIn('variant_groups', results)
        self.assertIn('similarity_scores', results)
        self.assertIn('correction_suggestions', results)
        self.assertIn('confidence_scores', results)
        self.assertIn('statistics', results)

    def test_map_to_standard_categories(self):
        mapping = {
            'laptop': 'Laptop',
            'desktop': 'Desktop',
            'shirt': 'Shirt',
            'pants': 'Pants',
            'fiction': 'Fiction',
            'non-fiction': 'Non-Fiction'
        }
        standardized_df = self.cv.map_to_standard_categories(
            column='subcategory',
            mapping=mapping,
            case_sensitive=False
        )
        self.assertIsInstance(standardized_df, DataFrame)

    def test_correct_with_fuzzy_matching(self):
        corrected_df = self.cv.correct_with_fuzzy_matching(
            column='subcategory',
            reference_values=['Laptop', 'Desktop', 'Shirt', 'Pants', 'Fiction', 'Non-Fiction'],
            similarity_threshold=0.8
        )
        self.assertIsInstance(corrected_df, DataFrame)

    def test_standardize_case(self):
        standardized_df = self.cv.standardize_case(
            columns='subcategory',
            case_type='title'
        )
        self.assertIsInstance(standardized_df, DataFrame)

    def test_group_rare_categories(self):
        grouped_df = self.cv.group_rare_categories(
            column='subcategory',
            threshold=0.1,
            grouping_method='frequency'
        )
        self.assertIsInstance(grouped_df, DataFrame)

if __name__ == '__main__':
    unittest.main()
