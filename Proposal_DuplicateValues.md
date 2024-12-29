# DataQualityChecker: Duplicate Values Module

## Overview
The Duplicate Values module provides comprehensive functionality for detecting and handling duplicate records in PySpark DataFrames. This module excels at identifying various types of duplicates, from exact matches to fuzzy duplicates and business key violations.

## Library Structure
```
DataQualityChecker/
├── __init__.py
├── duplicate_values.py
├── utils/
│   ├── __init__.py
│   └── similarity_metrics.py
└── tests/
    ├── __init__.py
    └── test_duplicate_values.py
```

## Core Components

### DuplicateValues Class Documentation

```python
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from typing import Dict, List, Union, Optional, Any
import jellyfish  # For fuzzy matching

class DuplicateValues:
    """
    A comprehensive class for detecting and handling duplicate records in PySpark DataFrames.
    Provides methods for identifying exact duplicates, fuzzy matches, and business key violations.
    
    Attributes
    ----------
    df : pyspark.sql.DataFrame
        The input DataFrame to analyze
    _timestamp_cols : List[str]
        Cache of timestamp columns for recency checks
    """

    def check_exact_duplicates(
            self,
            columns: Optional[List[str]] = None,
            sample_size: Optional[int] = 1000
        ) -> Dict[str, Any]:
        """
        Identifies exact duplicate records across specified columns or entire DataFrame.
        
        Parameters
        ----------
        columns : Optional[List[str]], default=None
            Columns to check for duplicates. If None, checks all columns
        sample_size : Optional[int], default=1000
            Number of sample duplicate records to include in results
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing detailed duplicate analysis:
            {
                'total_records': int,
                'duplicate_count': int,
                'duplicate_percentage': float,
                'affected_rows': int,
                'sample_duplicates': List[Dict],
                'column_impact': Dict[str, int],
                'group_sizes': Dict[int, int],
                'largest_groups': List[Dict]
            }
            
        Examples
        --------
        >>> dv = DuplicateValues(spark_df)
        >>> # Check specific columns for duplicates
        >>> results = dv.check_exact_duplicates(
        ...     columns=['customer_id', 'transaction_date']
        ... )
        >>> print(f"Found {results['duplicate_count']} duplicates")
        
        Notes
        -----
        The method performs comprehensive duplicate analysis:
        1. Identifies completely identical rows
        2. Analyzes duplicate patterns
        3. Provides statistical summary
        4. Samples representative duplicates
        """
        pass

    def check_fuzzy_matches(
            self,
            columns: List[str],
            threshold: float = 0.9,
            algorithm: str = 'levenshtein',
            blocking_columns: Optional[List[str]] = None
        ) -> Dict[str, Any]:
        """
        Identifies records that are similar but not exactly identical using fuzzy matching.
        
        Parameters
        ----------
        columns : List[str]
            Columns to analyze for fuzzy matches
        threshold : float, default=0.9
            Similarity threshold (0.0 to 1.0) for matching
        algorithm : str, default='levenshtein'
            Similarity algorithm to use:
            - 'levenshtein': Edit distance-based
            - 'jaro_winkler': Position-based
            - 'soundex': Phonetic similarity
            - 'ngram': N-gram-based similarity
        blocking_columns : Optional[List[str]], default=None
            Columns to use for blocking to improve performance
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing fuzzy match analysis:
            {
                'match_groups': List[Dict],
                'similarity_scores': Dict[str, float],
                'match_statistics': Dict[str, Any],
                'column_correlations': Dict[str, float],
                'suggested_thresholds': Dict[str, float],
                'sample_matches': List[Dict]
            }
            
        Examples
        --------
        >>> dv = DuplicateValues(spark_df)
        >>> # Find similar company names
        >>> fuzzy_results = dv.check_fuzzy_matches(
        ...     columns=['company_name'],
        ...     threshold=0.85,
        ...     algorithm='jaro_winkler'
        ... )
        
        Notes
        -----
        The method implements sophisticated fuzzy matching:
        1. Applies specified similarity algorithm
        2. Uses blocking for performance optimization
        3. Provides detailed match analysis
        4. Suggests optimal thresholds
        """
        pass

    def check_business_key_duplicates(
            self,
            key_columns: List[str],
            tolerance_rules: Optional[Dict[str, Any]] = None,
            temporal_constraints: Optional[Dict[str, str]] = None
        ) -> Dict[str, Any]:
        """
        Identifies duplicate records based on business keys with custom validation rules.
        
        Parameters
        ----------
        key_columns : List[str]
            Columns that form the business key
        tolerance_rules : Optional[Dict[str, Any]], default=None
            Rules for acceptable variations in non-key columns:
            {
                'column_name': {
                    'type': 'numeric|categorical|temporal',
                    'tolerance': value,
                    'unit': 'absolute|percentage'
                }
            }
        temporal_constraints : Optional[Dict[str, str]], default=None
            Time-based rules for duplicate validation:
            {
                'valid_from': 'column_name',
                'valid_to': 'column_name',
                'overlap_allowed': bool
            }
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing business key analysis:
            {
                'violation_count': int,
                'violation_groups': List[Dict],
                'key_statistics': Dict[str, Any],
                'tolerance_breaches': List[Dict],
                'temporal_violations': List[Dict],
                'impact_analysis': Dict[str, Any]
            }
            
        Examples
        --------
        >>> dv = DuplicateValues(spark_df)
        >>> # Check order duplicates with amount tolerance
        >>> biz_key_results = dv.check_business_key_duplicates(
        ...     key_columns=['order_id', 'customer_id'],
        ...     tolerance_rules={
        ...         'amount': {
        ...             'type': 'numeric',
        ...             'tolerance': 0.01,
        ...             'unit': 'absolute'
        ...         }
        ...     }
        ... )
        
        Notes
        -----
        Implements sophisticated business key validation:
        1. Validates composite business keys
        2. Applies tolerance rules
        3. Handles temporal aspects
        4. Provides violation analysis
        """
        pass

    def remove_exact_duplicates(
            self,
            columns: Optional[List[str]] = None,
            keep: str = 'first',
            order_by: Optional[List[str]] = None
        ) -> DataFrame:
        """
        Removes exact duplicate records based on specified criteria.
        
        Parameters
        ----------
        columns : Optional[List[str]], default=None
            Columns to consider for duplicate removal
        keep : str, default='first'
            Which record to keep:
            - 'first': Keep first occurrence
            - 'last': Keep last occurrence
            - 'most_complete': Keep record with most non-null values
            - 'most_recent': Keep most recent based on timestamp
        order_by : Optional[List[str]], default=None
            Columns to use for ordering when keep='first'|'last'
            
        Returns
        -------
        pyspark.sql.DataFrame
            DataFrame with duplicates removed
            
        Examples
        --------
        >>> dv = DuplicateValues(spark_df)
        >>> # Remove duplicates keeping most complete record
        >>> deduped_df = dv.remove_exact_duplicates(
        ...     columns=['customer_id', 'order_id'],
        ...     keep='most_complete'
        ... )
        
        Notes
        -----
        Implements intelligent duplicate removal:
        1. Considers specified columns
        2. Applies sophisticated keeping logic
        3. Preserves data integrity
        4. Optimizes performance
        """
        pass

    def merge_similar_records(
            self,
            match_columns: List[str],
            merge_rules: Dict[str, str],
            threshold: float = 0.9,
            conflict_resolution: str = 'most_frequent'
        ) -> DataFrame:
        """
        Merges records identified as similar based on specified rules.
        
        Parameters
        ----------
        match_columns : List[str]
            Columns to use for similarity matching
        merge_rules : Dict[str, str]
            Rules for merging column values:
            {
                'column_name': 'most_frequent|longest|newest|sum|average'
            }
        threshold : float, default=0.9
            Similarity threshold for matching
        conflict_resolution : str, default='most_frequent'
            Strategy for resolving conflicting values:
            - 'most_frequent': Use most common value
            - 'longest': Use longest string
            - 'newest': Use most recent value
            - 'manual': Raise exception for manual review
            
        Returns
        -------
        pyspark.sql.DataFrame
            DataFrame with similar records merged
            
        Examples
        --------
        >>> dv = DuplicateValues(spark_df)
        >>> # Merge similar customer records
        >>> merged_df = dv.merge_similar_records(
        ...     match_columns=['customer_name', 'address'],
        ...     merge_rules={
        ...         'customer_name': 'longest',
        ...         'email': 'most_frequent',
        ...         'total_purchases': 'sum'
        ...     }
        ... )
        
        Notes
        -----
        Implements sophisticated record merging:
        1. Identifies similar records
        2. Applies merge rules
        3. Resolves conflicts
        4. Maintains data consistency
        """
        pass

    def create_composite_key(
            self,
            columns: List[str],
            transformations: Optional[Dict[str, str]] = None,
            separator: str = '_'
        ) -> DataFrame:
        """
        Creates composite keys from multiple columns for unique record identification.
        
        Parameters
        ----------
        columns : List[str]
            Columns to combine into composite key
        transformations : Optional[Dict[str, str]], default=None
            Transformations to apply to columns:
            {
                'column_name': 'upper|lower|trim|clean|hash'
            }
        separator : str, default='_'
            Character to use between combined values
            
        Returns
        -------
        pyspark.sql.DataFrame
            DataFrame with new composite key column
            
        Examples
        --------
        >>> dv = DuplicateValues(spark_df)
        >>> # Create composite key from multiple columns
        >>> keyed_df = dv.create_composite_key(
        ...     columns=['region', 'customer_id', 'order_date'],
        ...     transformations={
        ...         'region': 'upper',
        ...         'customer_id': 'clean'
        ...     }
        ... )
        
        Notes
        -----
        Implements robust composite key creation:
        1. Combines multiple columns
        2. Applies transformations
        3. Ensures uniqueness
        4. Optimizes for performance
        """
        pass
```

## Usage Examples

```python
from DataQualityChecker import DuplicateValues

# Initialize checker
dup_checker = DuplicateValues(spark_df)

# Check for exact duplicates
dup_results = dup_checker.check_exact_duplicates(
    columns=['customer_id', 'order_date', 'amount']
)

# Find similar customer records
fuzzy_results = dup_checker.check_fuzzy_matches(
    columns=['customer_name', 'address'],
    threshold=0.85,
    algorithm='jaro_winkler'
)

# Check business key duplicates with tolerance
biz_key_results = dup_checker.check_business_key_duplicates(
    key_columns=['order_id'],
    tolerance_rules={
        'amount': {
            'type': 'numeric',
            'tolerance': 0.01,
            'unit': 'absolute'
        }
    }
)

# Remove duplicates keeping most complete records
deduped_df = dup_checker.remove_exact_duplicates(
    columns=['customer_id', 'order_id'],
    keep='most_complete'
)

# Merge similar records
merged_df = dup_checker.merge_similar_records(
    match_columns=['customer_name', 'address'],
    merge_rules={
        'customer_name': 'longest',
        'email': 'most_frequent',
        'total_purchases': 'sum'
    }
)

# Create composite keys
keyed_df = dup_checker.create_composite_key(
    columns=['region', 'customer_id', 'order_date'],
    transformations={
        'region': 'upper',
        'customer_id': 'clean'
    }
)
```

## Dependencies

- pyspark >= 3.0.0
- jellyfish >= 0.9.0
- pandas >= 1.3.0
- numpy >= 1.20.0

## Installation

```bash
pip install DataQualityChecker
```