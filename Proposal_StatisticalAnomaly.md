# DataQualityChecker: Statistical Anomaly Module

## Overview
The Statistical Anomaly module provides sophisticated functionality for detecting and handling statistical anomalies in PySpark DataFrames. This module employs advanced statistical methods to identify distribution anomalies, detect pattern breaks, and handle outliers through various statistical techniques.

## Library Structure
```
DataQualityChecker/
├── __init__.py
├── statistical_anomaly.py
├── utils/
│   ├── __init__.py
│   ├── statistical_methods.py
│   └── pattern_detection.py
└── tests/
    ├── __init__.py
    └── test_statistical_anomaly.py
```

## Core Components

### StatisticalAnomaly Class Documentation

```python
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from typing import Dict, List, Union, Optional, Any
import numpy as np
from scipy import stats

class StatisticalAnomaly:
    """
    A comprehensive class for detecting and handling statistical anomalies in PySpark DataFrames.
    Implements advanced statistical methods for anomaly detection and correction.
    
    This class provides methods for:
    - Distribution analysis and anomaly detection
    - Time series pattern break identification
    - Statistical smoothing and outlier handling
    - Moving average calculations and trend analysis
    
    Attributes
    ----------
    df : pyspark.sql.DataFrame
        The input DataFrame to analyze
    _cache : Dict
        Cache for computed statistical measures
    """

    def check_distribution_anomalies(
            self,
            column: str,
            distribution_type: Optional[str] = None,
            time_column: Optional[str] = None,
            segment_columns: Optional[List[str]] = None,
            confidence_level: float = 0.95
        ) -> Dict[str, Any]:
        """
        Performs comprehensive analysis of distribution anomalies in numerical data.
        
        This method analyzes the statistical distribution of values to identify anomalies
        through multiple statistical tests and methods. It can perform both static
        distribution analysis and time-series based anomaly detection.
        
        Parameters
        ----------
        column : str
            Column to analyze for anomalies
        distribution_type : Optional[str], default=None
            Expected distribution type:
            - 'normal': Gaussian distribution
            - 'uniform': Uniform distribution
            - 'poisson': Poisson distribution
            - 'auto': Automatically determine best fit
        time_column : Optional[str], default=None
            Time-based column for temporal analysis
        segment_columns : Optional[List[str]], default=None
            Columns to use for segmenting the analysis
        confidence_level : float, default=0.95
            Confidence level for statistical tests
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing detailed anomaly analysis:
            {
                'distribution_stats': {
                    'mean': float,
                    'median': float,
                    'std_dev': float,
                    'skewness': float,
                    'kurtosis': float
                },
                'anomaly_scores': Dict[str, float],
                'identified_anomalies': List[Dict],
                'temporal_patterns': Dict[str, Any],
                'segment_analysis': Dict[str, Dict],
                'test_results': {
                    'shapiro_wilk': float,
                    'anderson_darling': float,
                    'kolmogorov_smirnov': float
                },
                'visualization_data': Dict[str, List]
            }
            
        Examples
        --------
        >>> sa = StatisticalAnomaly(spark_df)
        >>> # Analyze sales distribution with temporal aspects
        >>> results = sa.check_distribution_anomalies(
        ...     column='daily_sales',
        ...     distribution_type='normal',
        ...     time_column='date',
        ...     segment_columns=['region'],
        ...     confidence_level=0.95
        ... )
        
        Notes
        -----
        The method implements comprehensive statistical analysis:
        1. Distribution fitting and testing
        2. Temporal pattern analysis
        3. Segmented distribution analysis
        4. Multiple statistical tests for robustness
        5. Visualization data preparation
        """
        pass

    def check_pattern_breaks(
            self,
            column: str,
            time_column: str,
            window_size: Optional[int] = None,
            detection_method: str = 'cusum',
            sensitivity: float = 1.0
        ) -> Dict[str, Any]:
        """
        Identifies significant breaks or changes in data patterns over time.
        
        This method analyzes time series data to detect pattern changes using
        various statistical methods. It can identify trend changes, level shifts,
        and seasonal pattern breaks.
        
        Parameters
        ----------
        column : str
            Column to analyze for pattern breaks
        time_column : str
            Column containing timestamp information
        window_size : Optional[int], default=None
            Size of the sliding window for analysis
            If None, automatically determined based on data
        detection_method : str, default='cusum'
            Method for detecting pattern breaks:
            - 'cusum': Cumulative sum control chart
            - 'ewma': Exponentially weighted moving average
            - 'changepoint': Changepoint detection
            - 'regression': Regression-based detection
        sensitivity : float, default=1.0
            Sensitivity of the detection algorithm (0.0 to 2.0)
            Higher values increase sensitivity to smaller changes
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing pattern break analysis:
            {
                'detected_breaks': List[Dict],
                'change_points': List[Dict],
                'trend_analysis': Dict[str, Any],
                'seasonality_breaks': List[Dict],
                'statistical_measures': Dict[str, float],
                'confidence_intervals': Dict[str, List],
                'metadata': Dict[str, Any]
            }
            
        Examples
        --------
        >>> sa = StatisticalAnomaly(spark_df)
        >>> # Detect pattern breaks in weekly sales
        >>> breaks = sa.check_pattern_breaks(
        ...     column='weekly_sales',
        ...     time_column='week',
        ...     detection_method='changepoint',
        ...     sensitivity=1.2
        ... )
        
        Notes
        -----
        Implements sophisticated pattern break detection:
        1. Multiple detection algorithms
        2. Trend and seasonality decomposition
        3. Confidence interval calculation
        4. Change point significance testing
        """
        pass

    def apply_statistical_smoothing(
            self,
            column: str,
            method: str = 'ema',
            window_size: Optional[int] = None,
            preserve_edges: bool = True,
            handle_nulls: str = 'interpolate'
        ) -> DataFrame:
        """
        Applies statistical smoothing techniques to reduce noise and anomalies.
        
        This method implements various smoothing algorithms to reduce noise while
        preserving underlying patterns and trends in the data.
        
        Parameters
        ----------
        column : str
            Column to smooth
        method : str, default='ema'
            Smoothing method to apply:
            - 'ema': Exponential moving average
            - 'kalman': Kalman filter
            - 'lowess': Locally weighted regression
            - 'gaussian': Gaussian filter
            - 'savitzky_golay': Savitzky-Golay filter
        window_size : Optional[int], default=None
            Size of the smoothing window
            If None, automatically determined
        preserve_edges : bool, default=True
            Whether to preserve edge behavior
        handle_nulls : str, default='interpolate'
            How to handle null values:
            - 'interpolate': Linear interpolation
            - 'forward': Forward fill
            - 'backward': Backward fill
            - 'drop': Remove nulls
            
        Returns
        -------
        pyspark.sql.DataFrame
            DataFrame with smoothed values
            
        Examples
        --------
        >>> sa = StatisticalAnomaly(spark_df)
        >>> # Apply EMA smoothing to sensor readings
        >>> smoothed_df = sa.apply_statistical_smoothing(
        ...     column='sensor_reading',
        ...     method='ema',
        ...     window_size=12
        ... )
        
        Notes
        -----
        Implements sophisticated smoothing techniques:
        1. Multiple smoothing algorithms
        2. Edge case handling
        3. Null value management
        4. Window size optimization
        """
        pass

    def remove_statistical_outliers(
            self,
            column: str,
            method: str = 'zscore',
            threshold: float = 3.0,
            handle_outliers: str = 'null'
        ) -> DataFrame:
        """
        Identifies and removes statistical outliers using various methods.
        
        This method implements multiple statistical approaches for identifying
        and handling outliers while preserving data integrity.
        
        Parameters
        ----------
        column : str
            Column to process for outliers
        method : str, default='zscore'
            Method for identifying outliers:
            - 'zscore': Z-score method
            - 'iqr': Interquartile range
            - 'isolation_forest': Isolation Forest
            - 'dbscan': DBSCAN clustering
            - 'mad': Median Absolute Deviation
        threshold : float, default=3.0
            Threshold for outlier detection
            For z-score, number of standard deviations
            For IQR, multiple of IQR
        handle_outliers : str, default='null'
            How to handle identified outliers:
            - 'null': Replace with null
            - 'remove': Remove rows
            - 'clip': Clip to threshold values
            - 'mean': Replace with mean
            - 'median': Replace with median
            
        Returns
        -------
        pyspark.sql.DataFrame
            DataFrame with outliers handled
            
        Examples
        --------
        >>> sa = StatisticalAnomaly(spark_df)
        >>> # Remove outliers using IQR method
        >>> clean_df = sa.remove_statistical_outliers(
        ...     column='transaction_amount',
        ...     method='iqr',
        ...     threshold=1.5,
        ...     handle_outliers='clip'
        ... )
        
        Notes
        -----
        Implements robust outlier detection:
        1. Multiple detection methods
        2. Configurable thresholds
        3. Various handling strategies
        4. Data integrity preservation
        """
        pass

    def calculate_moving_averages(
            self,
            column: str,
            window_sizes: List[int],
            weighted: bool = False,
            center: bool = True
        ) -> DataFrame:
        """
        Calculates various types of moving averages for trend analysis.
        
        This method computes different types of moving averages to help identify
        trends and smooth out short-term fluctuations.
        
        Parameters
        ----------
        column : str
            Column to calculate moving averages for
        window_sizes : List[int]
            List of window sizes for different averages
        weighted : bool, default=False
            Whether to apply linear weights to averages
        center : bool, default=True
            Whether to center the window
            
        Returns
        -------
        pyspark.sql.DataFrame
            DataFrame with added moving average columns
            
        Examples
        --------
        >>> sa = StatisticalAnomaly(spark_df)
        >>> # Calculate multiple moving averages
        >>> ma_df = sa.calculate_moving_averages(
        ...     column='daily_value',
        ...     window_sizes=[7, 14, 30],
        ...     weighted=True
        ... )
        
        Notes
        -----
        Implements versatile moving average calculation:
        1. Multiple window sizes
        2. Weighted and unweighted options
        3. Window positioning options
        4. Efficient computation
        """
        pass

    def flag_for_investigation(
            self,
            column: str,
            methods: List[str],
            thresholds: Dict[str, float],
            min_confidence: float = 0.8
        ) -> DataFrame:
        """
        Flags suspicious values for further investigation using multiple criteria.
        
        This method combines multiple anomaly detection approaches to identify
        values that warrant further investigation, with confidence scoring.
        
        Parameters
        ----------
        column : str
            Column to analyze for suspicious values
        methods : List[str]
            List of detection methods to apply:
            - 'statistical': Statistical tests
            - 'pattern': Pattern-based detection
            - 'forecast': Forecast-based detection
            - 'clustering': Cluster-based detection
        thresholds : Dict[str, float]
            Method-specific thresholds
        min_confidence : float, default=0.8
            Minimum confidence score to flag a value
            
        Returns
        -------
        pyspark.sql.DataFrame
            DataFrame with investigation flags and confidence scores
            
        Examples
        --------
        >>> sa = StatisticalAnomaly(spark_df)
        >>> # Flag suspicious values using multiple methods
        >>> flagged_df = sa.flag_for_investigation(
        ...     column='metric_value',
        ...     methods=['statistical', 'pattern'],
        ...     thresholds={
        ...         'statistical': 2.5,
        ...         'pattern': 0.9
        ...     }
        ... )
        
        Notes
        -----
        Implements comprehensive anomaly flagging:
        1. Multiple detection methods
        2. Confidence scoring
        3. Threshold customization
        4. Detailed flagging metadata
        """
        pass
```

## Usage Examples

```python
from DataQualityChecker import StatisticalAnomaly

# Initialize checker
stat_checker = StatisticalAnomaly(spark_df)

# Check for distribution anomalies
distribution_results = stat_checker.check_distribution_anomalies(
    column='daily_sales',
    distribution_type='normal',
    time_column='date',
    segment_columns=['region']
)

# Detect pattern breaks
pattern_breaks = stat_checker.check_pattern_breaks(
    column='weekly_sales',
    time_column='week',
    detection_method='changepoint',
    sensitivity=1.2
)

# Apply statistical smoothing
smoothed_df = stat_checker.apply_statistical_smoothing(
    column='sensor_reading',
    method='ema',
    window_size=12,
    preserve_edges=True
)

# Remove statistical outliers
clean_df = stat_checker.remove_statistical_outliers(
    column='transaction_amount',
    method='iqr',
    threshold=1.5,
    handle_outliers='clip'
)

# Calculate moving averages
ma_df = stat_checker.calculate_moving_averages(
    column='daily_value',
    window_sizes=[7, 14, 30],
    weighted=True
)

# Flag suspicious values
flagged_df = stat_checker.flag_for_investigation(
    column='metric_value',
    methods=['statistical', 'pattern'],
    thresholds={
        'statistical': 2.5,
        'pattern': 0.9
    }
)
```

## Dependencies

- pyspark >= 3.0.0
- numpy >= 1.20.0
- scipy >= 1.7.0
- statsmodels >= 0.13.0
- scikit-learn >= 1.0.0

## Installation

```bash
pip install DataQualityChecker
```
