# DataQualityChecker: Format Consistency Module

## Overview
The Format Consistency module provides sophisticated functionality for validating and standardizing formatted data like dates, phone numbers, emails, and addresses in PySpark DataFrames. This module combines pattern matching, lookup tables, and intelligent parsing to ensure data format consistency across your dataset.

## Library Structure
```
DataQualityChecker/
├── __init__.py
├── format_consistency.py
├── utils/
│   ├── __init__.py
│   ├── format_patterns.py
│   ├── parsers.py
│   └── lookup_tables.py
└── tests/
    ├── __init__.py
    └── test_format_consistency.py
```

## Core Components

### FormatConsistency Class Documentation

```python
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from typing import Dict, List, Union, Optional, Pattern
import re
from datetime import datetime

class FormatConsistency:
    """
    A comprehensive class for handling format consistency issues in PySpark DataFrames.
    Provides methods for detecting and fixing format violations in common data types
    like dates, phone numbers, emails, and addresses.
    """

    def check(self,
             df: DataFrame,
             columns: Union[str, List[str]],
             format_types: Dict[str, str],
             custom_patterns: Optional[Dict[str, str]] = None,
             locale: str = 'en_US') -> Dict[str, Dict]:
        """
        Performs comprehensive format consistency analysis on specified columns.
        
        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Input DataFrame to analyze
        columns : Union[str, List[str]]
            Single column name or list of column names to analyze
        format_types : Dict[str, str]
            Dictionary mapping columns to their expected format types:
            {
                'column_name': 'format_type'
            }
            Supported format types:
            - 'date': Date values
            - 'datetime': Date and time values
            - 'phone': Phone numbers
            - 'email': Email addresses
            - 'address': Postal addresses
            - 'ip': IP addresses
            - 'custom': Custom format (requires custom_patterns)
        custom_patterns : Optional[Dict[str, str]], default=None
            Dictionary of custom regex patterns for validation
        locale : str, default='en_US'
            Locale for format validation rules
            
        Returns
        -------
        Dict[str, Dict]
            Nested dictionary containing detailed format analysis:
            {
                'column_name': {
                    'format_type': str,
                    'violations': {
                        'total_count': int,
                        'invalid_format': int,
                        'mixed_formats': int,
                        'unknown_formats': int
                    },
                    'detected_patterns': List[str],
                    'pattern_frequencies': Dict[str, int],
                    'example_violations': List[Dict],
                    'violation_percentage': float,
                    'suggested_formats': List[str]
                }
            }
            
        Examples
        --------
        >>> fc = FormatConsistency()
        >>> # Check date and phone number formats
        >>> format_issues = fc.check(
        ...     df,
        ...     columns=['birth_date', 'phone'],
        ...     format_types={
        ...         'birth_date': 'date',
        ...         'phone': 'phone'
        ...     }
        ... )
        >>> 
        >>> # Check with custom pattern
        >>> custom_check = fc.check(
        ...     df,
        ...     columns=['product_code'],
        ...     format_types={'product_code': 'custom'},
        ...     custom_patterns={
        ...         'product_code': r'^[A-Z]{2}-\d{4}$'
        ...     }
        ... )
        
        Notes
        -----
        The method performs multiple levels of analysis:
        1. Pattern matching against standard formats
        2. Detection of mixed format usage
        3. Identification of common format patterns
        4. Statistical analysis of format distributions
        5. Validation against locale-specific rules
        """
        pass

    def fix(self,
            df: DataFrame,
            columns: Union[str, List[str]],
            format_types: Dict[str, str],
            strategy: str = 'standardize',
            target_formats: Optional[Dict[str, str]] = None,
            lookup_tables: Optional[Dict[str, DataFrame]] = None,
            handle_errors: str = 'flag') -> DataFrame:
        """
        Applies specified fixing strategy to handle format consistency issues.
        
        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Input DataFrame to fix
        columns : Union[str, List[str]]
            Columns to apply the fixing strategy to
        format_types : Dict[str, str]
            Dictionary mapping columns to their format types
            (same as check method)
        strategy : str, default='standardize'
            Strategy to handle format issues. Options:
            - 'standardize': Convert to standard format
            - 'parse': Parse and reconstruct
            - 'lookup': Use lookup tables for standardization
            - 'regex': Apply regex transformations
        target_formats : Optional[Dict[str, str]], default=None
            Dictionary specifying target formats:
            {
                'date': 'yyyy-MM-dd',
                'phone': '+1-XXX-XXX-XXXX',
                'email': 'lowercase'
            }
        lookup_tables : Optional[Dict[str, DataFrame]], default=None
            Dictionary of lookup tables for standardization
        handle_errors : str, default='flag'
            How to handle conversion errors:
            - 'flag': Add error indicator column
            - 'null': Replace with null
            - 'preserve': Keep original value
            - 'raise': Raise error
            
        Returns
        -------
        pyspark.sql.DataFrame
            DataFrame with format consistency issues fixed according to
            the specified strategy
            
        Examples
        --------
        >>> fc = FormatConsistency()
        >>> 
        >>> # Standardize dates and phones
        >>> fixed_df = fc.fix(
        ...     df,
        ...     columns=['birth_date', 'phone'],
        ...     format_types={
        ...         'birth_date': 'date',
        ...         'phone': 'phone'
        ...     },
        ...     target_formats={
        ...         'birth_date': 'yyyy-MM-dd',
        ...         'phone': '+1-XXX-XXX-XXXX'
        ...     }
        ... )
        >>> 
        >>> # Use lookup table for address standardization
        >>> standardized_df = fc.fix(
        ...     df,
        ...     columns=['address'],
        ...     format_types={'address': 'address'},
        ...     strategy='lookup',
        ...     lookup_tables={
        ...         'address': address_standards_df
        ...     }
        ... )
            
        Notes
        -----
        The method provides multiple strategies for handling format issues:
        
        1. Standardize Strategy:
           - Converts values to specified standard formats
           - Handles common variations automatically
           - Supports locale-specific formatting
           - Preserves semantic meaning
        
        2. Parse Strategy:
           - Breaks down complex formats into components
           - Reconstructs in desired format
           - Handles nested structures
           - Validates component values
        
        3. Lookup Strategy:
           - Uses reference tables for standardization
           - Supports fuzzy matching
           - Handles abbreviations and variants
           - Maintains consistency with standards
        
        4. Regex Strategy:
           - Applies pattern-based transformations
           - Supports complex string manipulations
           - Handles structured formats
           - Validates results against patterns
        
        Raises
        ------
        ValueError
            If strategy is invalid
            If required parameters are missing
            If format type is not supported
            If column names don't exist in DataFrame
        """
        pass

    def add_pattern(self,
                   name: str,
                   pattern: str,
                   validation_func: Optional[callable] = None,
                   description: str = '') -> None:
        """
        Adds a new custom pattern for format validation.
        
        Parameters
        ----------
        name : str
            Name of the new pattern
        pattern : str
            Regex pattern string
        validation_func : Optional[callable], default=None
            Custom validation function
        description : str, default=''
            Description of the pattern and its use
            
        Examples
        --------
        >>> fc = FormatConsistency()
        >>> fc.add_pattern(
        ...     name='product_code',
        ...     pattern=r'^[A-Z]{2}-\d{4}$',
        ...     description='Product code format: XX-9999'
        ... )
        """
        pass

    def parse_components(self,
                        df: DataFrame,
                        column: str,
                        format_type: str,
                        output_columns: Optional[List[str]] = None) -> DataFrame:
        """
        Parses formatted values into their component parts.
        
        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Input DataFrame
        column : str
            Column to parse
        format_type : str
            Type of format to parse
        output_columns : Optional[List[str]], default=None
            Names for the output component columns
            
        Returns
        -------
        pyspark.sql.DataFrame
            DataFrame with additional columns for components
            
        Examples
        --------
        >>> fc = FormatConsistency()
        >>> # Parse address into components
        >>> parsed_df = fc.parse_components(
        ...     df,
        ...     column='address',
        ...     format_type='address',
        ...     output_columns=['street', 'city', 'state', 'zip']
        ... )
        """
        pass
```

## Usage Examples

```python
from DataQualityChecker import FormatConsistency

# Initialize the checker
format_checker = FormatConsistency()

# Check format consistency
format_issues = format_checker.check(
    spark_df,
    columns=['birth_date', 'phone', 'email'],
    format_types={
        'birth_date': 'date',
        'phone': 'phone',
        'email': 'email'
    }
)

# Fix format issues using different strategies
# 1. Standardize formats
standardized_df = format_checker.fix(
    spark_df,
    columns=['birth_date', 'phone'],
    format_types={
        'birth_date': 'date',
        'phone': 'phone'
    },
    target_formats={
        'birth_date': 'yyyy-MM-dd',
        'phone': '+1-XXX-XXX-XXXX'
    }
)

# 2. Parse address components
parsed_df = format_checker.parse_components(
    spark_df,
    column='address',
    format_type='address',
    output_columns=['street', 'city', 'state', 'zip']
)

# 3. Add custom pattern
format_checker.add_pattern(
    name='student_id',
    pattern=r'^[A-Z]{2}\d{6}$',
    description='Student ID format: XX999999'
)
```

## Dependencies

- pyspark >= 3.0.0
- numpy >= 1.20.0
- pandas >= 1.3.0
- phonenumbers >= 8.12.0 (for phone number parsing)
- email-validator >= 1.1.0 (for email validation)
- usaddress >= 0.5.10 (for US address parsing)

## Installation

```bash
pip install DataQualityChecker
```

I've created a comprehensive library documentation that covers all the required functionality for format consistency checking and fixing. The documentation includes:

1. A clear class structure with detailed method documentation
2. Comprehensive docstrings with parameters, return values, and examples
3. Multiple strategies for handling different types of format consistency issues
4. Support for various format types (dates, phones, emails, addresses)
5. Detailed error handling and edge case considerations
6. Pattern management and component parsing capabilities

This implementation provides a flexible and robust solution for handling format consistency issues in PySpark DataFrames. The module is designed to be extensible, allowing for the addition of new format types and patterns as needed.

Would you like me to explain any particular aspect in more detail or provide additional implementation guidance for specific features?