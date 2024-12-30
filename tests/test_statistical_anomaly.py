import unittest
from pyspark.sql import SparkSession
from DataQualityKit.statistical_anomaly import StatisticalAnomaly

class TestStatisticalAnomaly(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.master("local[1]").appName("StatisticalAnomalyTest").getOrCreate()
        data = [(1,), (2,), (3,), (100,), (5,), (6,), (7,), (8,), (9,), (10,)]
        cls.df = cls.spark.createDataFrame(data, ["value"])
        cls.sa = StatisticalAnomaly()
        cls.sa.df = cls.df

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_check_distribution_anomalies(self):
        results = self.sa.check_distribution_anomalies(column="value", distribution_type="normal")
        self.assertIn("distribution_stats", results)
        self.assertIn("anomaly_scores", results)
        self.assertIn("identified_anomalies", results)
        self.assertIn("test_results", results)
        self.assertIn("visualization_data", results)
        self.assertGreater(len(results["identified_anomalies"]), 0)

    def test_check_pattern_breaks(self):
        data = [(1, "2021-01-01"), (2, "2021-01-02"), (3, "2021-01-03"), (100, "2021-01-04"), (5, "2021-01-05")]
        df = self.spark.createDataFrame(data, ["value", "date"])
        self.sa.df = df
        results = self.sa.check_pattern_breaks(column="value", time_column="date", detection_method="cusum")
        self.assertIn("detected_breaks", results)
        self.assertIn("change_points", results)
        self.assertIn("trend_analysis", results)
        self.assertGreater(len(results["detected_breaks"]), 0)

    def test_apply_statistical_smoothing(self):
        smoothed_df = self.sa.apply_statistical_smoothing(column="value", method="ema", window_size=3)
        smoothed_data = [row["value"] for row in smoothed_df.collect()]
        self.assertEqual(len(smoothed_data), 10)

    def test_remove_statistical_outliers(self):
        clean_df = self.sa.remove_statistical_outliers(column="value", method="zscore", threshold=2.0)
        clean_data = [row["value"] for row in clean_df.collect()]
        self.assertNotIn(100, clean_data)

    def test_calculate_moving_averages(self):
        ma_df = self.sa.calculate_moving_averages(column="value", window_sizes=[3, 5])
        ma_data = ma_df.collect()
        self.assertIn("ma_3", ma_data[0].asDict())
        self.assertIn("ma_5", ma_data[0].asDict())

    def test_flag_for_investigation(self):
        flagged_df = self.sa.flag_for_investigation(column="value", methods=["statistical"], thresholds={"statistical": 2.0})
        flagged_data = [row["flag"] for row in flagged_df.collect()]
        self.assertTrue(any(flagged_data))

if __name__ == "__main__":
    unittest.main()
