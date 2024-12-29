# DataQualityChecker: Range Validity Module

## Overview
The Range Validity module provides sophisticated functionality for detecting and handling out-of-range values, outliers, and impossible values in PySpark DataFrames. This module combines statistical analysis with domain-specific rules to ensure data falls within acceptable ranges.

## Library Structure
```
DataQualityChecker/
├── __init__.py
├── range_validity.py
├── utils/
│   ├── __init__.py
│   └── statistical_utils.py
└── tests/
    ├── __init__.py
    └── test_range_validity.py
```

## Core Components

### RangeValidity Class Documentation

```python
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from typing import Dict, List, Union, Optional, Tuple
import numpy as np

class RangeValidity:
    """
    A comprehensive class for handling range validity issues in PySpark DataFrames.
    Provides methods for detecting and fixing out-of-range values using both
    statistical and domain-specific approaches.
    """

    def check(self,
             df: DataFrame,
             columns: Union[str, List[str]],
             boundaries: Optional[Dict[str, Dict[str, float]]] = None,
             outlier_method: str = 'iqr',
             outlier_threshold: float = 1.5,
             custom_rules: Optional[Dict[str, str]] = None) -> Dict[str, Dict]:
        """
        Performs comprehensive range validity analysis on specified columns.
        
        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Input DataFrame to analyze
        columns : Union[str, List[str]]
            Single column name or list of column names to analyze
        boundaries : Optional[Dict[str, Dict[str, float]]], default=None
            Dictionary defining valid ranges for columns:
            {
                'column_name': {
                    'min': float,
                    'max': float,
                    'valid_set': List[float]  # if applicable
                }
            }
        outlier_method : str, default='iqr'
            Method to detect outliers:
            - 'iqr': Interquartile Range method
            - 'zscore': Z-score method
            - 'mad': Median Absolute Deviation
            - 'isolation_forest': Isolation Forest algorithm
        outlier_threshold : float, default=1.5
            Threshold for outlier detection:
            - For IQR: multiplier for IQR range
            - For zscore: number of standard deviations
            - For MAD: multiplier for MAD
        custom_rules : Optional[Dict[str, str]], default=None
            Dictionary of custom SQL expressions for validity checks
            
        Returns
        -------
        Dict[str, Dict]
            Nested dictionary containing detailed range analysis:
            {
                'column_name': {
                    'current_stats': {
                        'min': float,
                        'max': float,
                        'mean': float,
                        'median': float,
                        'std': float,
                        'q1': float,
                        'q3': float
                    },
                    'violations': {
                        'below_min': int,
                        'above_max': int,
                        'outliers': int,
                        'impossible_values': int
                    },
                    'violation_percentage': float,
                    'outlier_indices': List[int],
                    'boundary_violation_indices': List[int],
                    'violation_patterns': Dict[str, int]
                }
            }
            
        Examples
        --------
        >>> rv = RangeValidity()
        >>> # Check with specific boundaries
        >>> range_issues = rv.check(
        ...     df,
        ...     columns=['age', 'temperature'],
        ...     boundaries={
        ...         'age': {'min': 0, 'max': 120},
        ...         'temperature': {'min': -50, 'max': 50}
        ...     }
        ... )
        >>> 
        >>> # Check with custom rules
        >>> custom_checks = rv.check(
        ...     df,
        ...     columns=['blood_pressure'],
        ...     custom_rules={
        ...         'blood_pressure': 'systolic > diastolic AND systolic <= 300'
        ...     }
        ... )
        
        Notes
        -----
        The method performs multiple levels of analysis:
        1. Basic range checking against specified boundaries
        2. Statistical outlier detection using chosen method
        3. Impossible value detection based on domain rules
        4. Pattern analysis for violation clusters
        5. Distribution analysis for potential data quality issues
        """
        pass

    def fix(self,
            df: DataFrame,
            columns: Union[str, List[str]],
            strategy: str = 'cap',
            boundaries: Optional[Dict[str, Dict[str, float]]] = None,
            outlier_params: Optional[Dict[str, Any]] = None,
            transform_method: Optional[str] = None,
            add_indicators: bool = False) -> DataFrame:
        """
        Applies specified fixing strategy to handle range validity issues.
        
        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Input DataFrame to fix
        columns : Union[str, List[str]]
            Columns to apply the fixing strategy to
        strategy : str, default='cap'
            Strategy to handle range issues. Options:
            - 'cap': Cap values at boundaries
            - 'remove': Remove rows with violations
            - 'transform': Apply statistical transformations
            - 'flag': Add indicator columns for review
        boundaries : Optional[Dict[str, Dict[str, float]]], default=None
            Dictionary defining valid ranges (same as check method)
        outlier_params : Optional[Dict[str, Any]], default=None
            Parameters for outlier detection and handling:
            {
                'method': str,  # detection method
                'threshold': float,  # detection threshold
                'handling': str  # how to handle outliers
            }
        transform_method : Optional[str], default=None
            Statistical transformation to apply:
            - 'log': Natural logarithm
            - 'sqrt': Square root
            - 'box-cox': Box-Cox transformation
            - 'yeo-johnson': Yeo-Johnson transformation
        add_indicators : bool, default=False
            Whether to add indicator columns for violations
            
        Returns
        -------
        pyspark.sql.DataFrame
            DataFrame with range validity issues fixed according to
            the specified strategy
            
        Examples
        --------
        >>> rv = RangeValidity()
        >>> 
        >>> # Cap values at boundaries
        >>> capped_df = rv.fix(
        ...     df,
        ...     columns=['age', 'temperature'],
        ...     strategy='cap',
        ...     boundaries={
        ...         'age': {'min': 0, 'max': 120},
        ...         'temperature': {'min': -50, 'max': 50}
        ...     }
        ... )
        >>> 
        >>> # Remove outliers
        >>> cleaned_df = rv.fix(
        ...     df,
        ...     columns=['salary'],
        ...     strategy='remove',
        ...     outlier_params={
        ...         'method': 'iqr',
        ...         'threshold': 1.5,
        ...         'handling': 'remove'
        ...     }
        ... )
        >>> 
        >>> # Apply transformation
        >>> transformed_df = rv.fix(
        ...     df,
        ...     columns=['skewed_values'],
        ...     strategy='transform',
        ...     transform_method='log'
        ... )
            
        Notes
        -----
        The method provides multiple strategies for handling range issues:
        
        1. Cap Strategy:
           - Replaces values outside boundaries with boundary values
           - Preserves data volume while controlling extremes
           - Can be applied separately to upper/lower bounds
           - Supports different capping rules per column
        
        2. Remove Strategy:
           - Removes rows containing out-of-range values
           - Can focus on specific violation types
           - Supports different criteria per column
           - May significantly reduce data volume
        
        3. Transform Strategy:
           - Applies statistical transformations to normalize data
           - Handles skewed distributions
           - Preserves relative relationships
           - Supports multiple transformation methods
        
        4. Flag Strategy:
           - Adds indicator columns for different violation types
           - Preserves original data
           - Enables downstream filtering and analysis
           - Supports custom flagging rules
        
        Raises
        ------
        ValueError
            If strategy is invalid
            If required parameters are missing
            If transformation method is not supported
            If column names don't exist in DataFrame
        """
        pass

    def suggest_boundaries(self,
                         df: DataFrame,
                         columns: Union[str, List[str]],
                         method: str = 'statistical',
                         domain_rules: Optional[Dict[str, Dict]] = None) -> Dict[str, Dict[str, float]]:
        """
        Suggests appropriate boundaries for specified columns based on
        data distribution and optional domain rules.
        
        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Input DataFrame to analyze
        columns : Union[str, List[str]]
            Columns to analyze for boundary suggestion
        method : str, default='statistical'
            Method to use for suggestion:
            - 'statistical': Based on distribution
            - 'percentile': Based on percentile ranges
            - 'domain': Based on provided domain rules
        domain_rules : Optional[Dict[str, Dict]], default=None
            Domain-specific rules for boundary calculation
            
        Returns
        -------
        Dict[str, Dict[str, float]]
            Suggested boundaries for each column
        
        Examples
        --------
        >>> rv = RangeValidity()
        >>> suggested_bounds = rv.suggest_boundaries(
        ...     df,
        ...     columns=['age', 'temperature']
        ... )
        >>> print(suggested_bounds)
        
        Notes
        -----
        The method uses various approaches to suggest boundaries:
        1. Statistical analysis of data distribution
        2. Domain-specific rules and constraints
        3. Historical data patterns
        4. Common sense validation
        """
        pass
```

## Usage Examples

```python
from DataQualityChecker import RangeValidity

# Initialize the checker
range_checker = RangeValidity()

# Get suggested boundaries
suggested_bounds = range_checker.suggest_boundaries(
    spark_df,
    columns=['age', 'temperature', 'pressure']
)

# Check for range violations
range_issues = range_checker.check(
    spark_df,
    columns=['age', 'temperature'],
    boundaries={
        'age': {'min': 0, 'max': 120},
        'temperature': {'min': -50, 'max': 50}
    },
    outlier_method='iqr'
)

# Fix range issues using different strategies
# 1. Cap at boundaries
capped_df = range_checker.fix(
    spark_df,
    columns=['age', 'temperature'],
    strategy='cap',
    boundaries=suggested_bounds
)

# 2. Remove outliers
cleaned_df = range_checker.fix(
    spark_df,
    columns=['salary'],
    strategy='remove',
    outlier_params={
        'method': 'iqr',
        'threshold': 1.5
    }
)

# 3. Apply transformation
transformed_df = range_checker.fix(
    spark_df,
    columns=['highly_skewed'],
    strategy='transform',
    transform_method='box-cox'
)
```

## Dependencies

- pyspark >= 3.0.0
- numpy >= 1.20.0
- scipy >= 1.7.0 (for statistical transformations)
- scikit-learn >= 0.24.0 (for isolation forest)

## Installation

```bash
pip install DataQualityChecker
```

I've created a comprehensive library documentation that covers all the required functionality for range validity checking and fixing. The documentation includes:

1. A clear class structure with detailed method documentation
2. Comprehensive docstrings with parameters, return values, and examples
3. Multiple strategies for handling different types of range validity issues
4. Detailed error handling and edge case considerations
5. Statistical methods for outlier detection and handling
6. Domain-specific rule support

Would you like me to explain any particular aspect in more detail or provide additional implementation guidance for specific features?