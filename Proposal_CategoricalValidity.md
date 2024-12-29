# DataQualityChecker: Categorical Validity Module

## Overview
The Categorical Validity module provides comprehensive functionality for validating and standardizing categorical data in PySpark DataFrames. This module excels at identifying and correcting category inconsistencies, handling misspellings, standardizing formats, and managing rare categories.

## Library Structure
```
DataQualityChecker/
├── __init__.py
├── categorical_validity.py
├── utils/
│   ├── __init__.py
│   ├── string_matching.py
│   └── category_grouping.py
└── tests/
    ├── __init__.py
    └── test_categorical_validity.py
```

## Core Components

### CategoricalValidity Class Documentation

```python
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from typing import Dict, List, Union, Optional, Any
import jellyfish  # For fuzzy matching

class CategoricalValidity:
    """
    A comprehensive class for validating and standardizing categorical data in PySpark DataFrames.
    Provides methods for detecting and correcting category inconsistencies, misspellings,
    and format variations.
    
    Attributes
    ----------
    df : pyspark.sql.DataFrame
        The input DataFrame to analyze
    _cache : Dict
        Cache for frequently used computations
    """

    def check_category_validity(
            self,
            column: str,
            valid_categories: Optional[List[str]] = None,
            case_sensitive: bool = False,
            frequency_threshold: Optional[float] = None
        ) -> Dict[str, Any]:
        """
        Performs comprehensive analysis of categorical validity issues.
        
        Parameters
        ----------
        column : str
            Column name to analyze
        valid_categories : Optional[List[str]], default=None
            List of valid category values. If None, infers from data
        case_sensitive : bool, default=False
            Whether to treat different cases as distinct categories
        frequency_threshold : Optional[float], default=None
            Threshold for identifying rare categories (0.0 to 1.0)
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing detailed category analysis:
            {
                'invalid_categories': List[str],
                'category_frequencies': Dict[str, int],
                'rare_categories': List[str],
                'case_variations': Dict[str, List[str]],
                'potential_misspellings': Dict[str, List[Dict]],
                'statistics': {
                    'total_categories': int,
                    'invalid_count': int,
                    'rare_count': int,
                    'case_inconsistencies': int
                },
                'suggestions': {
                    'mappings': Dict[str, str],
                    'groupings': List[List[str]]
                }
            }
            
        Examples
        --------
        >>> cv = CategoricalValidity(spark_df)
        >>> results = cv.check_category_validity(
        ...     column='product_category',
        ...     valid_categories=['Electronics', 'Clothing', 'Books'],
        ...     frequency_threshold=0.01
        ... )
        >>> print(f"Found {len(results['invalid_categories'])} invalid categories")
        
        Notes
        -----
        The method performs comprehensive categorical analysis:
        1. Validates against known categories
        2. Identifies rare categories
        3. Detects case inconsistencies
        4. Suggests potential corrections
        """
        pass

    def check_spelling_variants(
            self,
            column: str,
            reference_values: Optional[List[str]] = None,
            similarity_threshold: float = 0.85,
            algorithm: str = 'jaro_winkler'
        ) -> Dict[str, Any]:
        """
        Identifies potential misspellings and variants in categorical values.
        
        Parameters
        ----------
        column : str
            Column to analyze for spelling variants
        reference_values : Optional[List[str]], default=None
            Known correct spellings. If None, uses most frequent values
        similarity_threshold : float, default=0.85
            Threshold for considering values as variants (0.0 to 1.0)
        algorithm : str, default='jaro_winkler'
            Algorithm for string similarity:
            - 'levenshtein': Edit distance-based
            - 'jaro_winkler': Position-based
            - 'soundex': Phonetic similarity
            - 'ngram': N-gram-based similarity
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing spelling analysis:
            {
                'variant_groups': List[Dict],
                'similarity_scores': Dict[str, float],
                'correction_suggestions': Dict[str, str],
                'confidence_scores': Dict[str, float],
                'statistics': {
                    'total_variants': int,
                    'unique_base_values': int,
                    'average_group_size': float
                }
            }
            
        Examples
        --------
        >>> cv = CategoricalValidity(spark_df)
        >>> variants = cv.check_spelling_variants(
        ...     column='country',
        ...     reference_values=['United States', 'United Kingdom', 'Canada'],
        ...     similarity_threshold=0.9
        ... )
        
        Notes
        -----
        Implements sophisticated spelling analysis:
        1. Identifies similar values
        2. Groups related variants
        3. Suggests corrections
        4. Provides confidence scores
        """
        pass

    def map_to_standard_categories(
            self,
            column: str,
            mapping: Dict[str, str],
            handle_unknown: str = 'keep',
            case_sensitive: bool = False
        ) -> DataFrame:
        """
        Maps categorical values to standardized categories using a provided mapping.
        
        Parameters
        ----------
        column : str
            Column to standardize
        mapping : Dict[str, str]
            Dictionary mapping current values to standard categories
        handle_unknown : str, default='keep'
            How to handle values not in mapping:
            - 'keep': Preserve original value
            - 'null': Set to null
            - 'error': Raise error
            - 'other': Map to 'Other' category
        case_sensitive : bool, default=False
            Whether to perform case-sensitive mapping
            
        Returns
        -------
        pyspark.sql.DataFrame
            DataFrame with standardized categories
            
        Examples
        --------
        >>> cv = CategoricalValidity(spark_df)
        >>> standardized_df = cv.map_to_standard_categories(
        ...     column='status',
        ...     mapping={
        ...         'in_progress': 'In Progress',
        ...         'in progress': 'In Progress',
        ...         'done': 'Completed',
        ...         'finished': 'Completed'
        ...     }
        ... )
        
        Notes
        -----
        Implements robust category standardization:
        1. Applies mapping rules
        2. Handles unknown values
        3. Manages case sensitivity
        4. Preserves data integrity
        """
        pass

    def correct_with_fuzzy_matching(
            self,
            column: str,
            reference_values: List[str],
            similarity_threshold: float = 0.85,
            max_suggestions: int = 1
        ) -> DataFrame:
        """
        Corrects categorical values using fuzzy matching against reference values.
        
        Parameters
        ----------
        column : str
            Column to correct
        reference_values : List[str]
            List of correct reference values
        similarity_threshold : float, default=0.85
            Minimum similarity score for matching (0.0 to 1.0)
        max_suggestions : int, default=1
            Maximum number of correction suggestions to return
            
        Returns
        -------
        pyspark.sql.DataFrame
            DataFrame with corrected categories and confidence scores
            
        Examples
        --------
        >>> cv = CategoricalValidity(spark_df)
        >>> corrected_df = cv.correct_with_fuzzy_matching(
        ...     column='product_type',
        ...     reference_values=['Laptop', 'Desktop', 'Tablet'],
        ...     similarity_threshold=0.8
        ... )
        
        Notes
        -----
        Implements intelligent fuzzy correction:
        1. Computes similarity scores
        2. Suggests corrections
        3. Handles ambiguous cases
        4. Provides confidence metrics
        """
        pass

    def standardize_case(
            self,
            columns: Union[str, List[str]],
            case_type: str = 'title',
            custom_rules: Optional[Dict[str, str]] = None
        ) -> DataFrame:
        """
        Standardizes the case format of categorical values.
        
        Parameters
        ----------
        columns : Union[str, List[str]]
            Columns to standardize
        case_type : str, default='title'
            Type of case standardization:
            - 'lower': All lowercase
            - 'upper': All uppercase
            - 'title': Title Case
            - 'sentence': Sentence case
            - 'custom': Use custom_rules
        custom_rules : Optional[Dict[str, str]], default=None
            Custom case mapping rules
            
        Returns
        -------
        pyspark.sql.DataFrame
            DataFrame with standardized case format
            
        Examples
        --------
        >>> cv = CategoricalValidity(spark_df)
        >>> standardized_df = cv.standardize_case(
        ...     columns=['category', 'subcategory'],
        ...     case_type='title'
        ... )
        
        Notes
        -----
        Implements comprehensive case standardization:
        1. Applies case rules
        2. Handles special cases
        3. Preserves acronyms
        4. Supports custom formatting
        """
        pass

    def group_rare_categories(
            self,
            column: str,
            threshold: float = 0.01,
            grouping_method: str = 'frequency',
            other_category_name: str = 'Other'
        ) -> DataFrame:
        """
        Groups infrequent categories into a single category.
        
        Parameters
        ----------
        column : str
            Column containing categories to group
        threshold : float, default=0.01
            Frequency threshold for considering a category rare (0.0 to 1.0)
        grouping_method : str, default='frequency'
            Method for identifying rare categories:
            - 'frequency': Based on occurrence count
            - 'percentage': Based on percentage of total
            - 'rank': Based on frequency rank
        other_category_name : str, default='Other'
            Name for the grouped category
            
        Returns
        -------
        pyspark.sql.DataFrame
            DataFrame with rare categories grouped
            
        Examples
        --------
        >>> cv = CategoricalValidity(spark_df)
        >>> grouped_df = cv.group_rare_categories(
        ...     column='product_subcategory',
        ...     threshold=0.05,
        ...     grouping_method='percentage'
        ... )
        
        Notes
        -----
        Implements intelligent category grouping:
        1. Identifies rare categories
        2. Applies grouping rules
        3. Preserves data distribution
        4. Maintains traceability
        """
        pass
```

## Usage Examples

```python
from DataQualityChecker import CategoricalValidity

# Initialize checker
cat_checker = CategoricalValidity(spark_df)

# Check category validity
validity_results = cat_checker.check_category_validity(
    column='product_category',
    valid_categories=['Electronics', 'Clothing', 'Books'],
    frequency_threshold=0.01
)

# Check for spelling variants
spelling_results = cat_checker.check_spelling_variants(
    column='country',
    reference_values=['United States', 'United Kingdom', 'Canada'],
    similarity_threshold=0.9
)

# Map to standard categories
standardized_df = cat_checker.map_to_standard_categories(
    column='status',
    mapping={
        'in_progress': 'In Progress',
        'done': 'Completed',
        'finished': 'Completed'
    }
)

# Correct using fuzzy matching
corrected_df = cat_checker.correct_with_fuzzy_matching(
    column='product_type',
    reference_values=['Laptop', 'Desktop', 'Tablet'],
    similarity_threshold=0.8
)

# Standardize case format
case_standardized_df = cat_checker.standardize_case(
    columns=['category', 'subcategory'],
    case_type='title'
)

# Group rare categories
grouped_df = cat_checker.group_rare_categories(
    column='product_subcategory',
    threshold=0.05,
    grouping_method='percentage'
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
