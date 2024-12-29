# DataQualityChecker: PySpark Data Quality Library

## Overview
DataQualityChecker is a Python library designed to perform comprehensive data quality checks and remediation actions for PySpark DataFrames

## Library Structure
```
DataQualityChecker/
├── __init__.py
├── QualityControl.py
├── utils/
│   ├── __init__.py
│   └── spark_helpers.py
└── tests/
    ├── __init__.py
    └── test_null_values.py
```

## Core Components

### NullValues Class Documentation

```python
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from typing import Dict, List, Union, Optional
from pyspark.ml.feature import Imputer

class NullValues:
    """
    A comprehensive class for handling missing values in PySpark DataFrames.
    Provides methods for detecting, analyzing, and fixing missing values.
    """

    def list_all(self, df: DataFrame) -> DataFrame:
        """
        Analyzes all columns in a DataFrame and returns missing value statistics.
        
        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Input DataFrame to analyze
            
        Returns
        -------
        pyspark.sql.DataFrame
            A new DataFrame containing the following columns:
            - column_name: string (Name of the analyzed column)
            - total_count: long (Total number of rows)
            - missing_count: long (Number of missing values)
            - missing_percentage: double (Percentage of missing values)
            - data_type: string (Data type of the column)
            
        Examples
        --------
        >>> from DataQualityChecker import NullValues
        >>> nv = NullValues()
        >>> missing_stats = nv.list_all(spark_df)
        >>> missing_stats.show()
        
        Notes
        -----
        The method considers the following as missing values:
        - NULL values
        - Empty strings ('')
        - Whitespace-only strings
        - Special characters representing missing values ('NA', 'N/A', etc.)
        """
        pass

    def check(self, 
              df: DataFrame, 
              columns: Union[str, List[str]],
              include_empty_strings: bool = True,
              include_whitespace: bool = True,
              custom_missing_values: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Performs detailed missing value analysis on specified columns.
        
        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Input DataFrame to analyze
        columns : Union[str, List[str]]
            Single column name or list of column names to analyze
        include_empty_strings : bool, default=True
            Whether to consider empty strings as missing values
        include_whitespace : bool, default=True
            Whether to consider whitespace-only strings as missing values
        custom_missing_values : Optional[List[str]], default=None
            Additional string values to consider as missing
            (e.g., ['NA', 'N/A', 'null', 'none'])
            
        Returns
        -------
        Dict[str, Dict]
            Nested dictionary containing detailed statistics for each column:
            {
                'column_name': {
                    'null_count': int,
                    'empty_string_count': int,
                    'whitespace_count': int,
                    'custom_missing_count': int,
                    'total_missing': int,
                    'missing_percentage': float,
                    'distinct_missing_patterns': List[str]
                }
            }
            
        Examples
        --------
        >>> nv = NullValues()
        >>> stats = nv.check(df, ['col1', 'col2'])
        >>> print(stats['col1']['missing_percentage'])
        
        Notes
        -----
        The method provides granular information about different types
        of missing values to help determine the appropriate fixing strategy.
        """
        pass

    def fix(self,
            df: DataFrame,
            columns: Union[str, List[str]],
            strategy: str = 'delete',
            fill_value: Optional[Union[str, float, Dict]] = None,
            imputation_method: str = 'mean',
            subset: Optional[List[str]] = None) -> DataFrame:
        """
        Applies specified fixing strategy to handle missing values.
        
        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Input DataFrame to fix
        columns : Union[str, List[str]]
            Columns to apply the fixing strategy to
        strategy : str, default='delete'
            Strategy to handle missing values. Options:
            - 'delete': Remove rows with missing values
            - 'impute': Use statistical imputation
            - 'fill': Fill with specified values
            - 'flag': Add indicator columns for missing values
        fill_value : Optional[Union[str, float, Dict]], default=None
            - If string/float: Use this value for all specified columns
            - If dict: Map of column names to fill values
            Required when strategy='fill'
        imputation_method : str, default='mean'
            Method to use when strategy='impute'. Options:
            - 'mean': Use column mean
            - 'median': Use column median
            - 'mode': Use most frequent value
            - 'ml': Use ML-based imputation
        subset : Optional[List[str]], default=None
            Only consider these columns when removing rows
            Only applicable when strategy='delete'
            
        Returns
        -------
        pyspark.sql.DataFrame
            DataFrame with missing values handled according to
            the specified strategy
            
        Examples
        --------
        >>> nv = NullValues()
        >>> # Remove rows with missing values
        >>> clean_df = nv.fix(df, 'col1', strategy='delete')
        >>> 
        >>> # Impute missing values with mean
        >>> imputed_df = nv.fix(df, ['col1', 'col2'], 
        ...                     strategy='impute', 
        ...                     imputation_method='mean')
        >>> 
        >>> # Fill missing values with specific values
        >>> filled_df = nv.fix(df, ['col1', 'col2'],
        ...                    strategy='fill',
        ...                    fill_value={'col1': 0, 'col2': 'unknown'})
            
        Notes
        -----
        The method provides multiple strategies for handling missing values:
        
        1. Delete Strategy:
           - Removes rows where specified columns have missing values
           - Can specify subset of columns to consider
           - Useful when missing data makes the entire row invalid
        
        2. Impute Strategy:
           - Uses statistical methods to estimate missing values
           - Supports mean, median, mode for numeric columns
           - ML-based imputation uses iterative regression
           - Preserves data volume but may introduce bias
        
        3. Fill Strategy:
           - Replaces missing values with specified constants
           - Can specify different values for different columns
           - Useful when you have domain-specific default values
        
        4. Flag Strategy:
           - Adds binary indicator columns for missing values
           - Preserves information about missingness
           - Useful for downstream analysis of missing patterns
        
        Raises
        ------
        ValueError
            If strategy is 'fill' but no fill_value is provided
            If invalid strategy or imputation_method is specified
            If column names don't exist in DataFrame
        """
        pass
```

## Usage Examples

```python
import QualityControl as qc

# Initialize the checker
null_checker = qc.NullValues()

# List all missing values in the DataFrame
missing_stats = null_checker.list_all(spark_df)
missing_stats.show()

# Check specific columns
detailed_stats = null_checker.check(
    spark_df,
    columns=['age', 'income'],
    include_empty_strings=True,
    custom_missing_values=['NA', 'N/A']
)

# Fix missing values using different strategies
# 1. Delete rows with missing values
clean_df = null_checker.fix(
    spark_df,
    columns=['age', 'income'],
    strategy='delete'
)

# 2. Impute missing values
imputed_df = null_checker.fix(
    spark_df,
    columns=['age', 'income'],
    strategy='impute',
    imputation_method='mean'
)

# 3. Fill with specific values
filled_df = null_checker.fix(
    spark_df,
    columns=['age', 'income'],
    strategy='fill',
    fill_value={'age': 0, 'income': 0.0}
)
```

## Dependencies

- pyspark >= 3.0.0
- numpy >= 1.20.0
- pandas >= 1.3.0 (for certain statistical operations)

## Installation

```bash
pip install DataQualityChecker
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
