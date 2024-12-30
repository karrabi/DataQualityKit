# DataQualityChecker: Data Type Conformity Module

## Overview
The Data Type Conformity module provides comprehensive functionality for detecting and fixing data type inconsistencies in PySpark DataFrames. This module is particularly useful when dealing with mixed data types, incorrect type assignments, and structured string parsing needs.

## Library Structure
```
DataQualityChecker/
├── __init__.py
├── data_types.py
├── utils/
│   ├── __init__.py
│   └── type_parsers.py
└── tests/
    ├── __init__.py
    └── test_data_types.py
```

## Core Components

### DataTypeConformity Class Documentation

```python
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import *
from typing import Dict, List, Union, Optional, Any

class DataTypeConformity:
    """
    A comprehensive class for handling data type conformity issues in PySpark DataFrames.
    Provides methods for detecting type mismatches and applying type-related transformations.
    """

    def check(self,
             df: DataFrame,
             columns: Union[str, List[str]],
             expected_types: Optional[Dict[str, str]] = None,
             detect_mixed: bool = True,
             sample_size: Optional[int] = 1000) -> Dict[str, Dict]:
        """
        Performs detailed analysis of data type conformity issues in specified columns.
        
        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Input DataFrame to analyze
        columns : Union[str, List[str]]
            Single column name or list of column names to analyze
        expected_types : Optional[Dict[str, str]], default=None
            Dictionary mapping column names to expected data types
            Example: {'age': 'integer', 'salary': 'double'}
        detect_mixed : bool, default=True
            Whether to detect mixed data types within columns
        sample_size : Optional[int], default=1000
            Number of rows to sample for detailed type analysis
            Set to None to analyze entire DataFrame
            
        Returns
        -------
        Dict[str, Dict]
            Nested dictionary containing detailed type information for each column:
            {
                'column_name': {
                    'current_type': str,
                    'expected_type': str,
                    'detected_types': List[str],
                    'type_frequencies': Dict[str, int],
                    'conversion_possible': bool,
                    'problematic_values': List[str],
                    'sample_violations': List[Dict],
                    'total_violations': int,
                    'violation_percentage': float
                }
            }
            
        Examples
        --------
        >>> dtc = DataTypeConformity()
        >>> # Check specific columns with expected types
        >>> type_issues = dtc.check(
        ...     df,
        ...     columns=['age', 'salary'],
        ...     expected_types={'age': 'integer', 'salary': 'double'}
        ... )
        >>> 
        >>> # Print violation percentage for age column
        >>> print(type_issues['age']['violation_percentage'])
        
        Notes
        -----
        The method performs several levels of analysis:
        1. Basic type checking against expected types
        2. Mixed type detection within columns
        3. Pattern recognition for structured strings
        4. Conversion possibility assessment
        5. Statistical analysis of type distributions
        
        For structured strings, it attempts to identify common patterns
        (dates, numbers with units, composite values, etc.)
        """
        pass

    def fix(self,
            df: DataFrame,
            columns: Union[str, List[str]],
            target_types: Optional[Dict[str, str]] = None,
            strategy: str = 'convert',
            handling_method: str = 'coerce',
            split_columns: bool = False,
            string_pattern: Optional[str] = None) -> DataFrame:
        """
        Applies specified fixing strategy to handle data type conformity issues.
        
        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Input DataFrame to fix
        columns : Union[str, List[str]]
            Columns to apply the fixing strategy to
        target_types : Optional[Dict[str, str]], default=None
            Dictionary mapping column names to target data types
            Required when strategy='convert'
        strategy : str, default='convert'
            Strategy to handle type issues. Options:
            - 'convert': Convert to specified target types
            - 'parse': Parse structured strings
            - 'clean': Remove non-conforming characters
            - 'split': Split mixed data into separate columns
        handling_method : str, default='coerce'
            How to handle conversion errors. Options:
            - 'coerce': Replace failed conversions with null
            - 'raise': Raise error on conversion failure
            - 'preserve': Keep original value if conversion fails
        split_columns : bool, default=False
            Whether to create new columns for different parts
            Only applicable when strategy='split'
        string_pattern : Optional[str], default=None
            Regex pattern for parsing structured strings
            Required when strategy='parse'
            
        Returns
        -------
        pyspark.sql.DataFrame
            DataFrame with data type issues fixed according to
            the specified strategy
            
        Examples
        --------
        >>> dtc = DataTypeConformity()
        >>> 
        >>> # Convert columns to specific types
        >>> fixed_df = dtc.fix(
        ...     df,
        ...     columns=['age', 'salary'],
        ...     target_types={'age': 'integer', 'salary': 'double'},
        ...     strategy='convert'
        ... )
        >>> 
        >>> # Parse structured date strings
        >>> parsed_df = dtc.fix(
        ...     df,
        ...     columns='date_string',
        ...     strategy='parse',
        ...     string_pattern=r'(\d{2})-(\d{2})-(\d{4})'
        ... )
        >>> 
        >>> # Split composite column into parts
        >>> split_df = dtc.fix(
        ...     df,
        ...     columns='dimension',
        ...     strategy='split',
        ...     split_columns=True
        ... )
            
        Notes
        -----
        The method provides multiple strategies for handling type issues:
        
        1. Convert Strategy:
           - Attempts to convert values to specified target types
           - Handles conversion errors according to handling_method
           - Supports all standard Spark data types
           - Can perform smart type inference if target_type not specified
        
        2. Parse Strategy:
           - Extracts structured data from string columns
           - Uses regex patterns for parsing
           - Can handle common formats (dates, numbers with units)
           - Creates properly typed columns from parsed components
        
        3. Clean Strategy:
           - Removes characters that prevent proper type conversion
           - Handles common issues like currency symbols, units
           - Preserves semantic meaning where possible
           - Can be combined with 'convert' strategy
        
        4. Split Strategy:
           - Identifies and separates mixed data types
           - Creates new columns for different components
           - Maintains relationships between split values
           - Useful for composite fields like dimensions or ranges
        
        Raises
        ------
        ValueError
            If strategy is 'convert' but no target_types provided
            If strategy is 'parse' but no string_pattern provided
            If invalid strategy or handling_method specified
            If column names don't exist in DataFrame
        """
        pass

    def infer_types(self,
                   df: DataFrame,
                   columns: Union[str, List[str]],
                   sample_size: Optional[int] = 1000) -> Dict[str, str]:
        """
        Attempts to infer the most appropriate data types for specified columns.
        
        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Input DataFrame to analyze
        columns : Union[str, List[str]]
            Columns to analyze for type inference
        sample_size : Optional[int], default=1000
            Number of rows to sample for type analysis
            
        Returns
        -------
        Dict[str, str]
            Dictionary mapping column names to inferred data types
            
        Examples
        --------
        >>> dtc = DataTypeConformity()
        >>> inferred_types = dtc.infer_types(df, ['col1', 'col2'])
        >>> print(inferred_types)
        
        Notes
        -----
        The method uses various heuristics to determine appropriate types:
        1. Pattern matching for common formats
        2. Statistical analysis of value distributions
        3. Semantic analysis of content
        4. Consideration of precision requirements
        """
        pass
```

## Usage Examples

```python
from DataQualityChecker import DataTypeConformity

# Initialize the checker
type_checker = DataTypeConformity()

# Check for type issues
type_issues = type_checker.check(
    spark_df,
    columns=['age', 'salary', 'date_joined'],
    expected_types={
        'age': 'integer',
        'salary': 'double',
        'date_joined': 'date'
    }
)

# Infer appropriate types
inferred_types = type_checker.infer_types(
    spark_df,
    columns=['unnamed_col1', 'unnamed_col2']
)

# Fix type issues using different strategies
# 1. Convert to proper types
converted_df = type_checker.fix(
    spark_df,
    columns=['age', 'salary'],
    target_types={'age': 'integer', 'salary': 'double'},
    strategy='convert'
)

# 2. Parse structured strings
parsed_df = type_checker.fix(
    spark_df,
    columns='date_joined',
    strategy='parse',
    string_pattern=r'(\d{2})/(\d{2})/(\d{4})'
)

# 3. Split mixed data
split_df = type_checker.fix(
    spark_df,
    columns='dimensions',
    strategy='split',
    split_columns=True
)
```

## Dependencies

- pyspark >= 3.0.0
- numpy >= 1.20.0
- pandas >= 1.3.0 (for certain type inference operations)

## Installation

```bash
pip install DataQualityChecker
```

I've created a comprehensive library documentation that covers all the required functionality for data type conformity checking and fixing. The documentation includes:

1. A clear class structure with detailed method documentation
2. Comprehensive docstrings with parameters, return values, and examples
3. Multiple strategies for handling different types of data type issues
4. Detailed error handling and edge case considerations
5. Practical usage examples

Would you like me to explain any particular aspect in more detail or provide additional implementation guidance for specific features?