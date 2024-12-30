# DataQualityChecker: Encoding Conformity Module

## Overview
The Encoding Conformity module provides comprehensive functionality for detecting and fixing character encoding issues in PySpark DataFrames. This module specializes in handling encoding-related problems such as mixed character sets, invalid characters, and non-standard encodings.

## Library Structure
```
DataQualityChecker/
├── __init__.py
├── encoding.py
├── utils/
│   ├── __init__.py
│   └── encoding_utils.py
└── tests/
    ├── __init__.py
    └── test_encoding.py
```

## Core Components

### EncodingConformity Class Documentation

```python
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from typing import Dict, List, Union, Optional, Set

class EncodingConformity:
    """
    A comprehensive class for handling character encoding issues in PySpark DataFrames.
    Provides methods for detecting encoding problems and applying encoding-related transformations.
    """

    def check(self,
             df: DataFrame,
             columns: Union[str, List[str]],
             target_encoding: str = 'UTF-8',
             detect_special_chars: bool = True,
             sample_size: Optional[int] = 1000) -> Dict[str, Dict]:
        """
        Performs detailed analysis of encoding issues in specified string columns.
        
        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Input DataFrame to analyze
        columns : Union[str, List[str]]
            Single column name or list of column names to analyze
        target_encoding : str, default='UTF-8'
            Expected character encoding for the columns
        detect_special_chars : bool, default=True
            Whether to detect and report special characters
        sample_size : Optional[int], default=1000
            Number of rows to sample for detailed encoding analysis
            Set to None to analyze entire DataFrame
            
        Returns
        -------
        Dict[str, Dict]
            Nested dictionary containing detailed encoding information for each column:
            {
                'column_name': {
                    'current_encoding': str,
                    'detected_encodings': List[str],
                    'special_chars': Set[str],
                    'invalid_chars': Set[str],
                    'encoding_frequencies': Dict[str, int],
                    'conversion_possible': bool,
                    'problematic_values': List[str],
                    'sample_violations': List[Dict],
                    'total_violations': int,
                    'violation_percentage': float
                }
            }
            
        Examples
        --------
        >>> enc = EncodingConformity()
        >>> # Check specific columns for encoding issues
        >>> encoding_issues = enc.check(
        ...     df,
        ...     columns=['name', 'description'],
        ...     target_encoding='UTF-8'
        ... )
        >>> 
        >>> # Print detected special characters
        >>> print(encoding_issues['name']['special_chars'])
        
        Notes
        -----
        The method performs several levels of analysis:
        1. Basic encoding validation against target encoding
        2. Special character detection
        3. Invalid character identification
        4. Statistical analysis of character distributions
        5. Encoding conversion possibility assessment
        
        Special attention is given to common problematic characters
        and encoding-specific issues (e.g., Windows-1252 vs UTF-8)
        """
        pass

    def fix(self,
            df: DataFrame,
            columns: Union[str, List[str]],
            strategy: str = 'convert',
            target_encoding: str = 'UTF-8',
            handling_method: str = 'replace',
            replacement_char: str = '?',
            preserve_special_chars: bool = True) -> DataFrame:
        """
        Applies specified fixing strategy to handle encoding issues.
        
        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Input DataFrame to fix
        columns : Union[str, List[str]]
            Columns to apply the fixing strategy to
        strategy : str, default='convert'
            Strategy to handle encoding issues. Options:
            - 'convert': Convert to target encoding
            - 'remove': Remove invalid characters
            - 'replace': Replace invalid characters with replacement_char
            - 'encode': Replace with encoded equivalents
        target_encoding : str, default='UTF-8'
            Target character encoding for conversion
        handling_method : str, default='replace'
            How to handle conversion errors. Options:
            - 'replace': Replace invalid characters
            - 'ignore': Skip invalid characters
            - 'strict': Raise error on invalid characters
        replacement_char : str, default='?'
            Character to use for replacement when handling_method='replace'
        preserve_special_chars : bool, default=True
            Whether to preserve valid special characters during conversion
            
        Returns
        -------
        pyspark.sql.DataFrame
            DataFrame with encoding issues fixed according to
            the specified strategy
            
        Examples
        --------
        >>> enc = EncodingConformity()
        >>> 
        >>> # Convert columns to UTF-8
        >>> fixed_df = enc.fix(
        ...     df,
        ...     columns=['name', 'description'],
        ...     strategy='convert',
        ...     target_encoding='UTF-8'
        ... )
        >>> 
        >>> # Remove invalid characters
        >>> cleaned_df = enc.fix(
        ...     df,
        ...     columns=['text'],
        ...     strategy='remove'
        ... )
        >>> 
        >>> # Replace invalid characters with encoded equivalents
        >>> encoded_df = enc.fix(
        ...     df,
        ...     columns=['content'],
        ...     strategy='encode',
        ...     preserve_special_chars=True
        ... )
            
        Notes
        -----
        The method provides multiple strategies for handling encoding issues:
        
        1. Convert Strategy:
           - Converts text to specified target encoding
           - Handles conversion errors according to handling_method
           - Preserves valid special characters if specified
           - Supports all standard Python encodings
        
        2. Remove Strategy:
           - Removes invalid characters and unprintable characters
           - Preserves valid special characters if specified
           - Can be combined with 'convert' strategy
           - Maintains string length information
        
        3. Replace Strategy:
           - Replaces invalid characters with specified replacement
           - Handles common encoding-specific issues
           - Preserves string semantics where possible
           - Supports custom replacement characters
        
        4. Encode Strategy:
           - Replaces characters with encoded equivalents
           - Uses HTML/XML encoding where appropriate
           - Maintains readability of special characters
           - Useful for web-safe content
        
        Raises
        ------
        ValueError
            If invalid strategy or handling_method specified
            If invalid target_encoding specified
            If column names don't exist in DataFrame
            If replacement_char is more than one character
        """
        pass

    def detect_encoding(self,
                       df: DataFrame,
                       columns: Union[str, List[str]],
                       sample_size: Optional[int] = 1000) -> Dict[str, str]:
        """
        Attempts to detect the character encoding of specified columns.
        
        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Input DataFrame to analyze
        columns : Union[str, List[str]]
            Columns to analyze for encoding detection
        sample_size : Optional[int], default=1000
            Number of rows to sample for encoding analysis
            
        Returns
        -------
        Dict[str, str]
            Dictionary mapping column names to detected encodings
            
        Examples
        --------
        >>> enc = EncodingConformity()
        >>> detected_encodings = enc.detect_encoding(df, ['col1', 'col2'])
        >>> print(detected_encodings)
        
        Notes
        -----
        The method uses various heuristics to determine encodings:
        1. Character set analysis
        2. Byte order mark detection
        3. Statistical analysis of byte patterns
        4. Common encoding signatures
        """
        pass
```

## Usage Examples

```python
from DataQualityChecker import EncodingConformity

# Initialize the checker
encoding_checker = EncodingConformity()

# Check for encoding issues
encoding_issues = encoding_checker.check(
    spark_df,
    columns=['name', 'description'],
    target_encoding='UTF-8',
    detect_special_chars=True
)

# Detect encodings
detected_encodings = encoding_checker.detect_encoding(
    spark_df,
    columns=['text_col1', 'text_col2']
)

# Fix encoding issues using different strategies
# 1. Convert to UTF-8
converted_df = encoding_checker.fix(
    spark_df,
    columns=['name', 'description'],
    strategy='convert',
    target_encoding='UTF-8'
)

# 2. Remove invalid characters
cleaned_df = encoding_checker.fix(
    spark_df,
    columns=['text'],
    strategy='remove',
    preserve_special_chars=True
)

# 3. Replace with encoded equivalents
encoded_df = encoding_checker.fix(
    spark_df,
    columns=['content'],
    strategy='encode'
)
```

## Dependencies

- pyspark >= 3.0.0
- chardet >= 4.0.0
- ftfy >= 6.0.0

## Installation

```bash
pip install DataQualityChecker
```

The documentation provides:

1. A clear class structure with detailed method documentation
2. Comprehensive docstrings with parameters, return values, and examples
3. Multiple strategies for handling different types of encoding issues
4. Detailed error handling and edge case considerations
5. Practical usage examples
