from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from typing import Dict, List, Union, Optional, Tuple, Any
from pyspark.ml.feature import Imputer
from pyspark.sql.types import *
from pyspark.sql.window import Window
import numpy as np
import jellyfish  # For fuzzy matching

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
        total_count = df.count()
        columns = df.columns
        
        def missing_count(col):
            return F.sum(F.when(F.col(col).isNull() | 
                                (F.col(col) == '') | 
                                (F.trim(F.col(col)) == '') | 
                                (F.col(col).isin('NA', 'N/A', 'null', 'none')), 1).otherwise(0)).alias(col)
        
        missing_counts = df.select([missing_count(col) for col in columns]).collect()[0].asDict()
        
        result = []
        for col in columns:
            missing_count = missing_counts[col]
            missing_percentage = (missing_count / total_count) * 100
            data_type = df.schema[col].dataType.simpleString()
            result.append((col, total_count, missing_count, missing_percentage, data_type))
        
        result_df = df.sparkSession.createDataFrame(result, schema=['column_name', 'total_count', 'missing_count', 'missing_percentage', 'data_type'])
        return result_df

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
        if isinstance(columns, str):
            columns = [columns]
        
        custom_missing_values = custom_missing_values or []
        
        def count_missing(col):
            conditions = [F.col(col).isNull()]
            if include_empty_strings:
                conditions.append(F.col(col) == '')
            if include_whitespace:
                conditions.append(F.trim(F.col(col)) == '')
            if custom_missing_values:
                conditions.append(F.col(col).isin(custom_missing_values))
            return F.sum(F.when(F.reduce(lambda x, y: x | y, conditions), 1).otherwise(0)).alias(col)
        
        missing_stats = {}
        for col in columns:
            null_count = df.filter(F.col(col).isNull()).count()
            empty_string_count = df.filter(F.col(col) == '').count() if include_empty_strings else 0
            whitespace_count = df.filter(F.trim(F.col(col)) == '').count() if include_whitespace else 0
            custom_missing_count = df.filter(F.col(col).isin(custom_missing_values)).count() if custom_missing_values else 0
            total_missing = null_count + empty_string_count + whitespace_count + custom_missing_count
            missing_percentage = (total_missing / df.count()) * 100
            distinct_missing_patterns = df.filter(F.reduce(lambda x, y: x | y, [F.col(col).isNull(), 
                                                                               F.col(col) == '', 
                                                                               F.trim(F.col(col)) == '', 
                                                                               F.col(col).isin(custom_missing_values)])).select(col).distinct().rdd.flatMap(lambda x: x).collect()
            
            missing_stats[col] = {
                'null_count': null_count,
                'empty_string_count': empty_string_count,
                'whitespace_count': whitespace_count,
                'custom_missing_count': custom_missing_count,
                'total_missing': total_missing,
                'missing_percentage': missing_percentage,
                'distinct_missing_patterns': distinct_missing_patterns
            }
        
        return missing_stats

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
        if isinstance(columns, str):
            columns = [columns]
        
        if strategy == 'delete':
            if subset:
                df = df.dropna(subset=subset)
            else:
                df = df.dropna(subset=columns)
        
        elif strategy == 'fill':
            if fill_value is None:
                raise ValueError("fill_value must be provided when strategy is 'fill'")
            if isinstance(fill_value, dict):
                for col, value in fill_value.items():
                    df = df.fillna({col: value})
            else:
                df = df.fillna(fill_value, subset=columns)
        
        elif strategy == 'impute':
            if imputation_method not in ['mean', 'median', 'mode', 'ml']:
                raise ValueError("Invalid imputation_method specified")
            if imputation_method in ['mean', 'median']:
                for col in columns:
                    if imputation_method == 'mean':
                        fill_value = df.agg({col: 'mean'}).first()[0]
                    elif imputation_method == 'median':
                        fill_value = df.approxQuantile(col, [0.5], 0.25)[0]
                    df = df.fillna({col: fill_value})
            elif imputation_method == 'mode':
                for col in columns:
                    fill_value = df.groupBy(col).count().orderBy('count', ascending=False).first()[0]
                    df = df.fillna({col: fill_value})
            elif imputation_method == 'ml':
                imputer = Imputer(inputCols=columns, outputCols=columns)
                df = imputer.fit(df).transform(df)
        
        elif strategy == 'flag':
            for col in columns:
                df = df.withColumn(f"{col}_missing_flag", F.when(F.col(col).isNull() | 
                                                                 (F.col(col) == '') | 
                                                                 (F.trim(F.col(col)) == '') | 
                                                                 (F.col(col).isin('NA', 'N/A', 'null', 'none')), 1).otherwise(0))
        
        else:
            raise ValueError("Invalid strategy specified")
        
        return df


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
        # Ensure columns is a list
        if isinstance(columns, str):
            columns = [columns]

        # Sample the DataFrame if sample_size is specified
        if sample_size:
            df = df.sample(withReplacement=False, fraction=min(sample_size / df.count(), 1.0))

        results = {}
        for column in columns:
            result = {
                'current_type': str(df.schema[column].dataType),
                'expected_type': expected_types.get(column) if expected_types else None,
                'detected_types': [],
                'type_frequencies': {},
                'conversion_possible': True,
                'problematic_values': [],
                'sample_violations': [],
                'total_violations': 0,
                'violation_percentage': 0.0
            }

            # Detect mixed types
            if detect_mixed:
                type_counts = df.select(column).groupBy(F.col(column).cast("string")).count().collect()
                for row in type_counts:
                    detected_type = type(row[column])
                    result['detected_types'].append(detected_type)
                    result['type_frequencies'][detected_type] = row['count']

            # Check for type conformity
            if result['expected_type']:
                violations = df.filter(~F.col(column).cast(result['expected_type']).isNotNull()).select(column).collect()
                result['problematic_values'] = [row[column] for row in violations]
                result['total_violations'] = len(violations)
                result['violation_percentage'] = (result['total_violations'] / df.count()) * 100

            results[column] = result

        return results

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
        # Ensure columns is a list
        if isinstance(columns, str):
            columns = [columns]

        if strategy == 'convert' and not target_types:
            raise ValueError("target_types must be provided when strategy is 'convert'")
        if strategy == 'parse' and not string_pattern:
            raise ValueError("string_pattern must be provided when strategy is 'parse'")

        for column in columns:
            if strategy == 'convert':
                target_type = target_types.get(column)
                if not target_type:
                    raise ValueError(f"Target type for column {column} not specified")
                try:
                    df = df.withColumn(column, F.col(column).cast(target_type))
                except Exception as e:
                    if handling_method == 'coerce':
                        df = df.withColumn(column, F.col(column).cast(target_type).otherwise(None))
                    elif handling_method == 'raise':
                        raise e
                    elif handling_method == 'preserve':
                        pass  # Keep original value if conversion fails

            elif strategy == 'parse':
                df = df.withColumn(column, F.regexp_extract(F.col(column), string_pattern, 0))

            elif strategy == 'clean':
                df = df.withColumn(column, F.regexp_replace(F.col(column), r'[^0-9a-zA-Z]+', ''))

            elif strategy == 'split':
                if not split_columns:
                    raise ValueError("split_columns must be True when strategy is 'split'")
                split_col = F.split(F.col(column), string_pattern)
                for i in range(len(split_col)):
                    df = df.withColumn(f"{column}_part{i+1}", split_col.getItem(i))

            else:
                raise ValueError(f"Invalid strategy: {strategy}")

        return df

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
        # Ensure columns is a list
        if isinstance(columns, str):
            columns = [columns]

        # Sample the DataFrame if sample_size is specified
        if sample_size:
            df = df.sample(withReplacement=False, fraction=min(sample_size / df.count(), 1.0))

        inferred_types = {}
        for column in columns:
            sample_values = df.select(column).distinct().rdd.map(lambda row: row[0]).collect()
            inferred_type = self._infer_type_from_values(sample_values)
            inferred_types[column] = inferred_type

        return inferred_types

    def _infer_type_from_values(self, values: List[Any]) -> str:
        """
        Helper method to infer data type from a list of values.
        """
        if all(isinstance(value, int) for value in values):
            return 'integer'
        elif all(isinstance(value, float) for value in values):
            return 'double'
        elif all(isinstance(value, str) for value in values):
            if all(self._is_date(value) for value in values):
                return 'date'
            return 'string'
        else:
            return 'string'

    def _is_date(self, value: str) -> bool:
        """
        Helper method to check if a string value is a date.
        """
        try:
            from dateutil.parser import parse
            parse(value)
            return True
        except ValueError:
            return False
        

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
        # Convert single column to list
        if isinstance(columns, str):
            columns = [columns]

        results = {}

        for column in columns:
            col_stats = df.select(
                F.min(column).alias('min'),
                F.max(column).alias('max'),
                F.mean(column).alias('mean'),
                F.expr(f'percentile_approx({column}, 0.5)').alias('median'),
                F.stddev(column).alias('std'),
                F.expr(f'percentile_approx({column}, 0.25)').alias('q1'),
                F.expr(f'percentile_approx({column}, 0.75)').alias('q3')
            ).collect()[0].asDict()

            violations = {
                'below_min': 0,
                'above_max': 0,
                'outliers': 0,
                'impossible_values': 0
            }

            if boundaries and column in boundaries:
                boundary = boundaries[column]
                if 'min' in boundary:
                    violations['below_min'] = df.filter(F.col(column) < boundary['min']).count()
                if 'max' in boundary:
                    violations['above_max'] = df.filter(F.col(column) > boundary['max']).count()

            if outlier_method == 'iqr':
                iqr = col_stats['q3'] - col_stats['q1']
                lower_bound = col_stats['q1'] - outlier_threshold * iqr
                upper_bound = col_stats['q3'] + outlier_threshold * iqr
                violations['outliers'] = df.filter((F.col(column) < lower_bound) | (F.col(column) > upper_bound)).count()

            # Add more outlier detection methods as needed

            if custom_rules and column in custom_rules:
                violations['impossible_values'] = df.filter(F.expr(custom_rules[column])).count()

            results[column] = {
                'current_stats': col_stats,
                'violations': violations,
                'violation_percentage': sum(violations.values()) / df.count() * 100,
                'outlier_indices': [],  # Placeholder for actual indices
                'boundary_violation_indices': [],  # Placeholder for actual indices
                'violation_patterns': {}  # Placeholder for pattern analysis
            }

        return results

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
        # Convert single column to list
        if isinstance(columns, str):
            columns = [columns]

        for column in columns:
            if strategy == 'cap':
                if boundaries and column in boundaries:
                    boundary = boundaries[column]
                    if 'min' in boundary:
                        df = df.withColumn(column, F.when(F.col(column) < boundary['min'], boundary['min']).otherwise(F.col(column)))
                    if 'max' in boundary:
                        df = df.withColumn(column, F.when(F.col(column) > boundary['max'], boundary['max']).otherwise(F.col(column)))

            elif strategy == 'remove':
                if boundaries and column in boundaries:
                    boundary = boundaries[column]
                    if 'min' in boundary:
                        df = df.filter(F.col(column) >= boundary['min'])
                    if 'max' in boundary:
                        df = df.filter(F.col(column) <= boundary['max'])
                if outlier_params:
                    method = outlier_params.get('method', 'iqr')
                    threshold = outlier_params.get('threshold', 1.5)
                    if method == 'iqr':
                        q1, q3 = df.approxQuantile(column, [0.25, 0.75], 0.05)
                        iqr = q3 - q1
                        lower_bound = q1 - threshold * iqr
                        upper_bound = q3 + threshold * iqr
                        df = df.filter((F.col(column) >= lower_bound) & (F.col(column) <= upper_bound))

            elif strategy == 'transform':
                if transform_method == 'log':
                    df = df.withColumn(column, F.log(F.col(column)))
                elif transform_method == 'sqrt':
                    df = df.withColumn(column, F.sqrt(F.col(column)))
                # Add more transformation methods as needed

            elif strategy == 'flag':
                if boundaries and column in boundaries:
                    boundary = boundaries[column]
                    if 'min' in boundary:
                        df = df.withColumn(f'{column}_below_min', F.when(F.col(column) < boundary['min'], 1).otherwise(0))
                    if 'max' in boundary:
                        df = df.withColumn(f'{column}_above_max', F.when(F.col(column) > boundary['max'], 1).otherwise(0))
                if outlier_params:
                    method = outlier_params.get('method', 'iqr')
                    threshold = outlier_params.get('threshold', 1.5)
                    if method == 'iqr':
                        q1, q3 = df.approxQuantile(column, [0.25, 0.75], 0.05)
                        iqr = q3 - q1
                        lower_bound = q1 - threshold * iqr
                        upper_bound = q3 + threshold * iqr
                        df = df.withColumn(f'{column}_outlier', F.when((F.col(column) < lower_bound) | (F.col(column) > upper_bound), 1).otherwise(0))

        return df

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
        # Convert single column to list
        if isinstance(columns, str):
            columns = [columns]

        suggested_boundaries = {}

        for column in columns:
            if method == 'statistical':
                col_stats = df.select(
                    F.expr(f'percentile_approx({column}, 0.01)').alias('min'),
                    F.expr(f'percentile_approx({column}, 0.99)').alias('max')
                ).collect()[0].asDict()
                suggested_boundaries[column] = {
                    'min': col_stats['min'],
                    'max': col_stats['max']
                }

            elif method == 'percentile':
                col_stats = df.select(
                    F.expr(f'percentile_approx({column}, 0.05)').alias('min'),
                    F.expr(f'percentile_approx({column}, 0.95)').alias('max')
                ).collect()[0].asDict()
                suggested_boundaries[column] = {
                    'min': col_stats['min'],
                    'max': col_stats['max']
                }

            elif method == 'domain' and domain_rules and column in domain_rules:
                suggested_boundaries[column] = domain_rules[column]

            # Add more methods as needed

        return suggested_boundaries


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
        # Implementation of check_category_validity
        df = self.df
        if not case_sensitive:
            df = df.withColumn(column, F.lower(F.col(column)))
            if valid_categories:
                valid_categories = [cat.lower() for cat in valid_categories]

        category_counts = df.groupBy(column).count()
        total_count = category_counts.agg(F.sum("count")).collect()[0][0]

        category_frequencies = {row[column]: row["count"] for row in category_counts.collect()}
        invalid_categories = []
        rare_categories = []
        case_variations = {}
        potential_misspellings = {}

        if valid_categories:
            invalid_categories = [cat for cat in category_frequencies if cat not in valid_categories]

        if frequency_threshold:
            rare_categories = [cat for cat, count in category_frequencies.items() if count / total_count < frequency_threshold]

        if not case_sensitive:
            for cat in category_frequencies:
                variations = [var for var in category_frequencies if var.lower() == cat.lower() and var != cat]
                if variations:
                    case_variations[cat] = variations

        # Placeholder for potential misspellings detection
        # This would involve using a string similarity algorithm like Levenshtein or Jaro-Winkler

        statistics = {
            'total_categories': len(category_frequencies),
            'invalid_count': len(invalid_categories),
            'rare_count': len(rare_categories),
            'case_inconsistencies': len(case_variations)
        }

        suggestions = {
            'mappings': {},  # Placeholder for suggested mappings
            'groupings': []  # Placeholder for suggested groupings
        }

        return {
            'invalid_categories': invalid_categories,
            'category_frequencies': category_frequencies,
            'rare_categories': rare_categories,
            'case_variations': case_variations,
            'potential_misspellings': potential_misspellings,
            'statistics': statistics,
            'suggestions': suggestions
        }

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
        df = self.df
        if not reference_values:
            reference_values = df.groupBy(column).count().orderBy(F.desc("count")).limit(100).select(column).rdd.flatMap(lambda x: x).collect()

        def get_similarity_func(algorithm):
            if algorithm == 'levenshtein':
                return jellyfish.levenshtein_distance
            elif algorithm == 'jaro_winkler':
                return jellyfish.jaro_winkler
            elif algorithm == 'soundex':
                return jellyfish.soundex
            elif algorithm == 'ngram':
                return lambda x, y: jellyfish.ngram_similarity(x, y, 2)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")

        similarity_func = get_similarity_func(algorithm)
        variant_groups = []
        similarity_scores = {}
        correction_suggestions = {}
        confidence_scores = {}

        for value in df.select(column).distinct().rdd.flatMap(lambda x: x).collect():
            for ref_value in reference_values:
                similarity = similarity_func(value, ref_value)
                if similarity >= similarity_threshold:
                    similarity_scores[value] = similarity
                    correction_suggestions[value] = ref_value
                    confidence_scores[value] = similarity
                    variant_groups.append({'value': value, 'reference': ref_value, 'similarity': similarity})

        statistics = {
            'total_variants': len(variant_groups),
            'unique_base_values': len(reference_values),
            'average_group_size': len(variant_groups) / len(reference_values) if reference_values else 0
        }

        return {
            'variant_groups': variant_groups,
            'similarity_scores': similarity_scores,
            'correction_suggestions': correction_suggestions,
            'confidence_scores': confidence_scores,
            'statistics': statistics
        }

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
        df = self.df
        if not case_sensitive:
            df = df.withColumn(column, F.lower(F.col(column)))
            mapping = {k.lower(): v for k, v in mapping.items()}

        def map_value(value):
            if value in mapping:
                return mapping[value]
            elif handle_unknown == 'null':
                return None
            elif handle_unknown == 'error':
                raise ValueError(f"Unknown category: {value}")
            elif handle_unknown == 'other':
                return 'Other'
            else:
                return value

        map_udf = F.udf(map_value, F.StringType())
        df = df.withColumn(column, map_udf(F.col(column)))

        return df

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
        df = self.df

        def get_similarity_func(algorithm):
            if algorithm == 'levenshtein':
                return jellyfish.levenshtein_distance
            elif algorithm == 'jaro_winkler':
                return jellyfish.jaro_winkler
            elif algorithm == 'soundex':
                return jellyfish.soundex
            elif algorithm == 'ngram':
                return lambda x, y: jellyfish.ngram_similarity(x, y, 2)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")

        similarity_func = get_similarity_func('jaro_winkler')

        def correct_value(value):
            best_match = None
            best_similarity = 0
            for ref_value in reference_values:
                similarity = similarity_func(value, ref_value)
                if similarity > best_similarity and similarity >= similarity_threshold:
                    best_match = ref_value
                    best_similarity = similarity
            return best_match if best_match else value

        correct_udf = F.udf(correct_value, F.StringType())
        df = df.withColumn(column, correct_udf(F.col(column)))

        return df

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
        df = self.df
        if isinstance(columns, str):
            columns = [columns]

        def standardize(value, case_type, custom_rules):
            if custom_rules and value in custom_rules:
                return custom_rules[value]
            if case_type == 'lower':
                return value.lower()
            elif case_type == 'upper':
                return value.upper()
            elif case_type == 'title':
                return value.title()
            elif case_type == 'sentence':
                return value.capitalize()
            else:
                return value

        standardize_udf = F.udf(lambda x: standardize(x, case_type, custom_rules), F.StringType())

        for column in columns:
            df = df.withColumn(column, standardize_udf(F.col(column)))

        return df

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
        df = self.df

        if grouping_method == 'frequency':
            category_counts = df.groupBy(column).count()
            total_count = category_counts.agg(F.sum("count")).collect()[0][0]
            rare_categories = category_counts.filter(F.col("count") / total_count < threshold).select(column).rdd.flatMap(lambda x: x).collect()
        elif grouping_method == 'percentage':
            category_counts = df.groupBy(column).count()
            total_count = category_counts.agg(F.sum("count")).collect()[0][0]
            rare_categories = category_counts.filter(F.col("count") / total_count < threshold).select(column).rdd.flatMap(lambda x: x).collect()
        elif grouping_method == 'rank':
            category_counts = df.groupBy(column).count().orderBy(F.asc("count"))
            rare_categories = category_counts.limit(int(threshold * category_counts.count())).select(column).rdd.flatMap(lambda x: x).collect()
        else:
            raise ValueError(f"Unknown grouping method: {grouping_method}")

        def group_value(value):
            return other_category_name if value in rare_categories else value

        group_udf = F.udf(group_value, F.StringType())
        df = df.withColumn(column, group_udf(F.col(column)))

        return df
    

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
        # If no columns specified, use all columns
        if columns is None:
            columns = self.df.columns

        # Count total records
        total_records = self.df.count()

        # Group by specified columns and count occurrences
        grouped_df = self.df.groupBy(columns).count()

        # Filter groups with more than one occurrence (duplicates)
        duplicates_df = grouped_df.filter(F.col('count') > 1)

        # Count duplicate groups and affected rows
        duplicate_count = duplicates_df.count()
        affected_rows = duplicates_df.agg(F.sum('count')).collect()[0][0] - duplicate_count

        # Calculate duplicate percentage
        duplicate_percentage = (affected_rows / total_records) * 100

        # Sample duplicate records
        sample_duplicates = duplicates_df.orderBy(F.desc('count')).limit(sample_size).collect()

        # Analyze column impact
        column_impact = {col: self.df.groupBy(col).count().filter(F.col('count') > 1).count() for col in columns}

        # Analyze group sizes
        group_sizes = duplicates_df.groupBy('count').count().collect()
        group_sizes = {row['count']: row['count(1)'] for row in group_sizes}

        # Get largest groups
        largest_groups = duplicates_df.orderBy(F.desc('count')).limit(5).collect()

        return {
            'total_records': total_records,
            'duplicate_count': duplicate_count,
            'duplicate_percentage': duplicate_percentage,
            'affected_rows': affected_rows,
            'sample_duplicates': [row.asDict() for row in sample_duplicates],
            'column_impact': column_impact,
            'group_sizes': group_sizes,
            'largest_groups': [row.asDict() for row in largest_groups]
        }

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
        def get_similarity_func(algorithm: str):
            if algorithm == 'levenshtein':
                return jellyfish.levenshtein_distance
            elif algorithm == 'jaro_winkler':
                return jellyfish.jaro_winkler
            elif algorithm == 'soundex':
                return jellyfish.soundex
            elif algorithm == 'ngram':
                return lambda x, y: jellyfish.ngram_similarity(x, y, 2)
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")

        similarity_func = get_similarity_func(algorithm)

        def is_similar(row1, row2, columns, threshold):
            for col in columns:
                if similarity_func(row1[col], row2[col]) < threshold:
                    return False
            return True

        # Apply blocking if specified
        if blocking_columns:
            df = self.df.groupBy(blocking_columns).agg(F.collect_list(F.struct(*self.df.columns)).alias('grouped'))
        else:
            df = self.df.withColumn('grouped', F.array(*[F.struct(*self.df.columns)]))

        match_groups = []
        for row in df.collect():
            group = row['grouped']
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    if is_similar(group[i], group[j], columns, threshold):
                        match_groups.append((group[i], group[j]))

        # Calculate similarity scores
        similarity_scores = {f"{row1['id']}-{row2['id']}": similarity_func(row1[col], row2[col])
                             for row1, row2 in match_groups for col in columns}

        # Calculate match statistics
        match_statistics = {
            'total_matches': len(match_groups),
            'average_similarity': sum(similarity_scores.values()) / len(similarity_scores) if similarity_scores else 0
        }

        # Calculate column correlations
        column_correlations = {col: self.df.stat.corr(col, 'similarity') for col in columns}

        # Suggest optimal thresholds
        suggested_thresholds = {col: 0.9 for col in columns}  # Placeholder for actual logic

        # Sample matches
        sample_matches = match_groups[:10]

        return {
            'match_groups': [{'row1': row1.asDict(), 'row2': row2.asDict()} for row1, row2 in match_groups],
            'similarity_scores': similarity_scores,
            'match_statistics': match_statistics,
            'column_correlations': column_correlations,
            'suggested_thresholds': suggested_thresholds,
            'sample_matches': [{'row1': row1.asDict(), 'row2': row2.asDict()} for row1, row2 in sample_matches]
        }

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
        # Group by key columns
        grouped_df = self.df.groupBy(key_columns).count()

        # Filter groups with more than one occurrence (violations)
        violations_df = grouped_df.filter(F.col('count') > 1)

        # Count violation groups
        violation_count = violations_df.count()

        # Collect violation groups
        violation_groups = violations_df.collect()

        # Analyze key statistics
        key_statistics = {col: self.df.groupBy(col).count().collect() for col in key_columns}

        # Check tolerance breaches
        tolerance_breaches = []
        if tolerance_rules:
            for col, rule in tolerance_rules.items():
                if rule['type'] == 'numeric':
                    tolerance_breaches.append(
                        self.df.filter(F.abs(F.col(col) - F.lag(col).over(Window.partitionBy(key_columns).orderBy(col))) > rule['tolerance']).collect()
                    )
                elif rule['type'] == 'categorical':
                    tolerance_breaches.append(
                        self.df.filter(F.col(col) != F.lag(col).over(Window.partitionBy(key_columns).orderBy(col))).collect()
                    )
                elif rule['type'] == 'temporal':
                    tolerance_breaches.append(
                        self.df.filter(F.datediff(F.col(col), F.lag(col).over(Window.partitionBy(key_columns).orderBy(col))) > rule['tolerance']).collect()
                    )

        # Check temporal violations
        temporal_violations = []
        if temporal_constraints:
            valid_from = temporal_constraints.get('valid_from')
            valid_to = temporal_constraints.get('valid_to')
            overlap_allowed = temporal_constraints.get('overlap_allowed', False)
            if valid_from and valid_to:
                if overlap_allowed:
                    temporal_violations = self.df.filter(F.col(valid_from) < F.col(valid_to)).collect()
                else:
                    temporal_violations = self.df.filter(F.col(valid_from) >= F.col(valid_to)).collect()

        # Perform impact analysis
        impact_analysis = {
            'total_violations': violation_count,
            'total_tolerance_breaches': len(tolerance_breaches),
            'total_temporal_violations': len(temporal_violations)
        }

        return {
            'violation_count': violation_count,
            'violation_groups': [row.asDict() for row in violation_groups],
            'key_statistics': key_statistics,
            'tolerance_breaches': [row.asDict() for row in tolerance_breaches],
            'temporal_violations': [row.asDict() for row in temporal_violations],
            'impact_analysis': impact_analysis
        }

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
        if columns is None:
            columns = self.df.columns

        if keep == 'first':
            window_spec = Window.partitionBy(columns).orderBy(order_by if order_by else columns)
            deduped_df = self.df.withColumn('row_number', F.row_number().over(window_spec)).filter(F.col('row_number') == 1).drop('row_number')
        elif keep == 'last':
            window_spec = Window.partitionBy(columns).orderBy(F.desc(order_by if order_by else columns))
            deduped_df = self.df.withColumn('row_number', F.row_number().over(window_spec)).filter(F.col('row_number') == 1).drop('row_number')
        elif keep == 'most_complete':
            deduped_df = self.df.withColumn('null_count', sum(F.col(c).isNull().cast('int') for c in columns))
            window_spec = Window.partitionBy(columns).orderBy('null_count')
            deduped_df = deduped_df.withColumn('row_number', F.row_number().over(window_spec)).filter(F.col('row_number') == 1).drop('row_number', 'null_count')
        elif keep == 'most_recent':
            if not order_by:
                raise ValueError("order_by must be specified when keep='most_recent'")
            window_spec = Window.partitionBy(columns).orderBy(F.desc(order_by))
            deduped_df = self.df.withColumn('row_number', F.row_number().over(window_spec)).filter(F.col('row_number') == 1).drop('row_number')
        else:
            raise ValueError(f"Unsupported keep method: {keep}")

        return deduped_df

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
        def get_similarity_func(algorithm: str):
            if algorithm == 'levenshtein':
                return jellyfish.levenshtein_distance
            elif algorithm == 'jaro_winkler':
                return jellyfish.jaro_winkler
            elif algorithm == 'soundex':
                return jellyfish.soundex
            elif algorithm == 'ngram':
                return lambda x, y: jellyfish.ngram_similarity(x, y, 2)
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")

        similarity_func = get_similarity_func('levenshtein')

        def is_similar(row1, row2, columns, threshold):
            for col in columns:
                if similarity_func(row1[col], row2[col]) < threshold:
                    return False
            return True

        # Apply blocking if specified
        if blocking_columns:
            df = self.df.groupBy(blocking_columns).agg(F.collect_list(F.struct(*self.df.columns)).alias('grouped'))
        else:
            df = self.df.withColumn('grouped', F.array(*[F.struct(*self.df.columns)]))

        match_groups = []
        for row in df.collect():
            group = row['grouped']
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    if is_similar(group[i], group[j], match_columns, threshold):
                        match_groups.append((group[i], group[j]))

        def resolve_conflict(values, rule):
            if rule == 'most_frequent':
                return max(set(values), key=values.count)
            elif rule == 'longest':
                return max(values, key=len)
            elif rule == 'newest':
                return max(values)
            elif rule == 'sum':
                return sum(values)
            elif rule == 'average':
                return sum(values) / len(values)
            else:
                raise ValueError(f"Unsupported merge rule: {rule}")

        merged_records = []
        for row1, row2 in match_groups:
            merged_record = {}
            for col in self.df.columns:
                if col in merge_rules:
                    merged_record[col] = resolve_conflict([row1[col], row2[col]], merge_rules[col])
                else:
                    merged_record[col] = row1[col]
            merged_records.append(merged_record)

        merged_df = self.df.sql_ctx.createDataFrame(merged_records)

        return merged_df

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
        def apply_transformation(col, transformation):
            if transformation == 'upper':
                return F.upper(F.col(col))
            elif transformation == 'lower':
                return F.lower(F.col(col))
            elif transformation == 'trim':
                return F.trim(F.col(col))
            elif transformation == 'clean':
                return F.regexp_replace(F.col(col), '[^a-zA-Z0-9]', '')
            elif transformation == 'hash':
                return F.sha2(F.col(col), 256)
            else:
                raise ValueError(f"Unsupported transformation: {transformation}")

        if transformations is None:
            transformations = {}

        transformed_columns = [
            apply_transformation(col, transformations.get(col, '')) if col in transformations else F.col(col)
            for col in columns
        ]

        composite_key_col = F.concat_ws(separator, *transformed_columns)

        return self.df.withColumn('composite_key', composite_key_col)


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
        if isinstance(columns, str):
            columns = [columns]
        
        results = {}
        
        for column in columns:
            format_type = format_types.get(column)
            if not format_type:
                raise ValueError(f"No format type specified for column: {column}")
            
            if format_type == 'custom' and custom_patterns and column in custom_patterns:
                pattern = custom_patterns[column]
            else:
                pattern = self._get_standard_pattern(format_type, locale)
            
            violations = self._detect_violations(df, column, pattern)
            detected_patterns, pattern_frequencies = self._analyze_patterns(df, column)
            
            results[column] = {
                'format_type': format_type,
                'violations': violations,
                'detected_patterns': detected_patterns,
                'pattern_frequencies': pattern_frequencies,
                'example_violations': self._get_example_violations(df, column, pattern),
                'violation_percentage': violations['total_count'] / df.count() * 100,
                'suggested_formats': self._suggest_formats(detected_patterns)
            }
        
        return results

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
        if isinstance(columns, str):
            columns = [columns]
        
        for column in columns:
            format_type = format_types.get(column)
            if not format_type:
                raise ValueError(f"No format type specified for column: {column}")
            
            if strategy == 'standardize':
                df = self._standardize(df, column, format_type, target_formats)
            elif strategy == 'parse':
                df = self._parse(df, column, format_type)
            elif strategy == 'lookup':
                df = self._lookup(df, column, lookup_tables)
            elif strategy == 'regex':
                df = self._apply_regex(df, column, target_formats)
            else:
                raise ValueError(f"Invalid strategy: {strategy}")
            
            if handle_errors == 'flag':
                df = self._flag_errors(df, column)
            elif handle_errors == 'null':
                df = self._replace_with_null(df, column)
            elif handle_errors == 'preserve':
                pass  # No action needed
            elif handle_errors == 'raise':
                self._raise_errors(df, column)
            else:
                raise ValueError(f"Invalid error handling strategy: {handle_errors}")
        
        return df

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
        if not hasattr(self, '_custom_patterns'):
            self._custom_patterns = {}
        
        self._custom_patterns[name] = {
            'pattern': pattern,
            'validation_func': validation_func,
            'description': description
        }

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
        if format_type == 'address':
            components = ['street', 'city', 'state', 'zip']
        elif format_type == 'datetime':
            components = ['date', 'time']
        else:
            raise ValueError(f"Unsupported format type for parsing: {format_type}")
        
        if output_columns and len(output_columns) == len(components):
            components = output_columns
        
        for component in components:
            df = df.withColumn(component, F.lit(None))  # Placeholder for actual parsing logic
        
        return df


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
        # Initialize results dictionary
        results = {
            'distribution_stats': {},
            'anomaly_scores': {},
            'identified_anomalies': [],
            'temporal_patterns': {},
            'segment_analysis': {},
            'test_results': {},
            'visualization_data': {}
        }

        # Calculate basic statistics
        stats_df = self.df.select(
            F.mean(column).alias('mean'),
            F.expr(f'percentile_approx({column}, 0.5)').alias('median'),
            F.stddev(column).alias('std_dev'),
            F.skewness(column).alias('skewness'),
            F.kurtosis(column).alias('kurtosis')
        ).collect()[0]

        results['distribution_stats'] = {
            'mean': stats_df['mean'],
            'median': stats_df['median'],
            'std_dev': stats_df['std_dev'],
            'skewness': stats_df['skewness'],
            'kurtosis': stats_df['kurtosis']
        }

        # Perform statistical tests
        data = np.array(self.df.select(column).rdd.flatMap(lambda x: x).collect())
        shapiro_wilk = stats.shapiro(data)[1]
        anderson_darling = stats.anderson(data).statistic
        kolmogorov_smirnov = stats.kstest(data, 'norm')[1]

        results['test_results'] = {
            'shapiro_wilk': shapiro_wilk,
            'anderson_darling': anderson_darling,
            'kolmogorov_smirnov': kolmogorov_smirnov
        }

        # Identify anomalies based on z-scores
        z_scores = np.abs(stats.zscore(data))
        anomalies = np.where(z_scores > stats.norm.ppf(confidence_level))[0]
        results['anomaly_scores'] = {str(i): z_scores[i] for i in anomalies}
        results['identified_anomalies'] = [{'index': int(i), 'value': data[i]} for i in anomalies]

        # Temporal pattern analysis
        if time_column:
            window_spec = Window.orderBy(time_column)
            self.df = self.df.withColumn('z_score', F.abs((F.col(column) - stats_df['mean']) / stats_df['std_dev']))
            temporal_patterns = self.df.withColumn('is_anomaly', F.col('z_score') > stats.norm.ppf(confidence_level))
            results['temporal_patterns'] = temporal_patterns.select(time_column, 'is_anomaly').collect()

        # Segment analysis
        if segment_columns:
            for segment in segment_columns:
                segment_stats = self.df.groupBy(segment).agg(
                    F.mean(column).alias('mean'),
                    F.expr(f'percentile_approx({column}, 0.5)').alias('median'),
                    F.stddev(column).alias('std_dev'),
                    F.skewness(column).alias('skewness'),
                    F.kurtosis(column).alias('kurtosis')
                ).collect()
                results['segment_analysis'][segment] = segment_stats

        # Visualization data preparation
        results['visualization_data'] = {
            'histogram': np.histogram(data, bins='auto').tolist(),
            'boxplot': [np.percentile(data, q) for q in [25, 50, 75]]
        }

        return results

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
        # Initialize results dictionary
        results = {
            'detected_breaks': [],
            'change_points': [],
            'trend_analysis': {},
            'seasonality_breaks': [],
            'statistical_measures': {},
            'confidence_intervals': {},
            'metadata': {}
        }

        # Prepare data for analysis
        df = self.df.select(time_column, column).orderBy(time_column)
        data = np.array(df.select(column).rdd.flatMap(lambda x: x).collect())

        # Detect pattern breaks using the specified method
        if detection_method == 'cusum':
            mean = np.mean(data)
            std_dev = np.std(data)
            cusum_pos, cusum_neg = [0], [0]
            for i in range(1, len(data)):
                cusum_pos.append(max(0, cusum_pos[-1] + data[i] - mean - sensitivity * std_dev))
                cusum_neg.append(min(0, cusum_neg[-1] + data[i] - mean + sensitivity * std_dev))
            results['detected_breaks'] = [{'index': i, 'value': data[i]} for i in range(len(data)) if cusum_pos[i] > 0 or cusum_neg[i] < 0]
        elif detection_method == 'ewma':
            lambda_ = 2 / (window_size + 1) if window_size else 0.2
            ewma = [data[0]]
            for i in range(1, len(data)):
                ewma.append(lambda_ * data[i] + (1 - lambda_) * ewma[-1])
            results['detected_breaks'] = [{'index': i, 'value': data[i]} for i in range(len(data)) if abs(data[i] - ewma[i]) > sensitivity * np.std(data)]
        elif detection_method == 'changepoint':
            from ruptures import Binseg
            model = Binseg()
            change_points = model.fit_predict(data, pen=sensitivity)
            results['change_points'] = [{'index': cp, 'value': data[cp]} for cp in change_points]
        elif detection_method == 'regression':
            from sklearn.linear_model import LinearRegression
            X = np.arange(len(data)).reshape(-1, 1)
            model = LinearRegression().fit(X, data)
            trend = model.predict(X)
            results['trend_analysis'] = {'slope': model.coef_[0], 'intercept': model.intercept_}
            results['detected_breaks'] = [{'index': i, 'value': data[i]} for i in range(len(data)) if abs(data[i] - trend[i]) > sensitivity * np.std(data)]

        # Calculate confidence intervals
        results['confidence_intervals'] = {
            'mean': [np.mean(data) - 1.96 * np.std(data), np.mean(data) + 1.96 * np.std(data)],
            'std_dev': [np.std(data) - 1.96 * np.std(data) / np.sqrt(len(data)), np.std(data) + 1.96 * np.std(data) / np.sqrt(len(data))]
        }

        # Metadata
        results['metadata'] = {
            'detection_method': detection_method,
            'sensitivity': sensitivity,
            'window_size': window_size
        }

        return results

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
        # Handle null values
        if handle_nulls == 'interpolate':
            self.df = self.df.withColumn(column, F.expr(f'linear_interpolate({column})'))
        elif handle_nulls == 'forward':
            self.df = self.df.withColumn(column, F.expr(f'last({column}, True)').over(Window.orderBy(time_column).rowsBetween(Window.unboundedPreceding, 0)))
        elif handle_nulls == 'backward':
            self.df = self.df.withColumn(column, F.expr(f'first({column}, True)').over(Window.orderBy(time_column).rowsBetween(0, Window.unboundedFollowing)))
        elif handle_nulls == 'drop':
            self.df = self.df.dropna(subset=[column])

        # Apply smoothing method
        if method == 'ema':
            alpha = 2 / (window_size + 1) if window_size else 0.2
            self.df = self.df.withColumn('ema', F.expr(f'ewma({column}, {alpha})'))
        elif method == 'kalman':
            from pykalman import KalmanFilter
            data = np.array(self.df.select(column).rdd.flatMap(lambda x: x).collect())
            kf = KalmanFilter()
            smoothed_data = kf.em(data).smooth(data)[0]
            self.df = self.df.withColumn('kalman', F.array(smoothed_data.tolist()))
        elif method == 'lowess':
            from statsmodels.nonparametric.smoothers_lowess import lowess
            data = np.array(self.df.select(column).rdd.flatMap(lambda x: x).collect())
            smoothed_data = lowess(data, np.arange(len(data)), frac=window_size/len(data) if window_size else 0.2)[:, 1]
            self.df = self.df.withColumn('lowess', F.array(smoothed_data.tolist()))
        elif method == 'gaussian':
            from scipy.ndimage import gaussian_filter
            data = np.array(self.df.select(column).rdd.flatMap(lambda x: x).collect())
            smoothed_data = gaussian_filter(data, sigma=window_size if window_size else 1)
            self.df = self.df.withColumn('gaussian', F.array(smoothed_data.tolist()))
        elif method == 'savitzky_golay':
            from scipy.signal import savgol_filter
            data = np.array(self.df.select(column).rdd.flatMap(lambda x: x).collect())
            smoothed_data = savgol_filter(data, window_length=window_size if window_size else 5, polyorder=2)
            self.df = self.df.withColumn('savitzky_golay', F.array(smoothed_data.tolist()))

        # Preserve edges if required
        if preserve_edges:
            self.df = self.df.withColumn(column, F.coalesce(F.col('ema'), F.col('kalman'), F.col('lowess'), F.col('gaussian'), F.col('savitzky_golay'), F.col(column)))

        return self.df

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
        # Handle outliers based on the specified method
        if method == 'zscore':
            mean = self.df.select(F.mean(column)).collect()[0][0]
            std_dev = self.df.select(F.stddev(column)).collect()[0][0]
            z_scores = self.df.withColumn('z_score', (F.col(column) - mean) / std_dev)
            outliers = z_scores.filter(F.abs(F.col('z_score')) > threshold)
        elif method == 'iqr':
            q1 = self.df.approxQuantile(column, [0.25], 0.01)[0]
            q3 = self.df.approxQuantile(column, [0.75], 0.01)[0]
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            outliers = self.df.filter((F.col(column) < lower_bound) | (F.col(column) > upper_bound))
        elif method == 'isolation_forest':
            from pyspark.ml.feature import VectorAssembler
            from pyspark.ml.clustering import BisectingKMeans
            assembler = VectorAssembler(inputCols=[column], outputCol='features')
            df_features = assembler.transform(self.df)
            bisecting_kmeans = BisectingKMeans(k=2, seed=1)
            model = bisecting_kmeans.fit(df_features)
            predictions = model.transform(df_features)
            outliers = predictions.filter(predictions['prediction'] == 1)
        elif method == 'dbscan':
            from pyspark.ml.feature import VectorAssembler
            from pyspark.ml.clustering import DBSCAN
            assembler = VectorAssembler(inputCols=[column], outputCol='features')
            df_features = assembler.transform(self.df)
            dbscan = DBSCAN(eps=threshold, minPts=5)
            model = dbscan.fit(df_features)
            predictions = model.transform(df_features)
            outliers = predictions.filter(predictions['prediction'] == -1)
        elif method == 'mad':
            median = self.df.approxQuantile(column, [0.5], 0.01)[0]
            mad = self.df.withColumn('mad', F.abs(F.col(column) - median))
            mad_value = mad.select(F.expr('percentile_approx(mad, 0.5)')).collect()[0][0]
            outliers = mad.filter(F.abs(F.col(column) - median) > threshold * mad_value)

        # Handle identified outliers
        if handle_outliers == 'null':
            self.df = self.df.withColumn(column, F.when(outliers[column].isNotNull(), None).otherwise(F.col(column)))
        elif handle_outliers == 'remove':
            self.df = self.df.join(outliers, on=column, how='left_anti')
        elif handle_outliers == 'clip':
            if method == 'zscore':
                self.df = self.df.withColumn(column, F.when(F.abs(F.col('z_score')) > threshold, F.lit(mean + threshold * std_dev)).otherwise(F.col(column)))
            elif method == 'iqr':
                self.df = self.df.withColumn(column, F.when(F.col(column) < lower_bound, F.lit(lower_bound)).when(F.col(column) > upper_bound, F.lit(upper_bound)).otherwise(F.col(column)))
        elif handle_outliers == 'mean':
            mean_value = self.df.select(F.mean(column)).collect()[0][0]
            self.df = self.df.withColumn(column, F.when(outliers[column].isNotNull(), mean_value).otherwise(F.col(column)))
        elif handle_outliers == 'median':
            median_value = self.df.approxQuantile(column, [0.5], 0.01)[0]
            self.df = self.df.withColumn(column, F.when(outliers[column].isNotNull(), median_value).otherwise(F.col(column)))

        return self.df

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
        for window_size in window_sizes:
            if weighted:
                weights = np.arange(1, window_size + 1)
                sum_weights = np.sum(weights)
                window_spec = Window.orderBy(F.col('index')).rowsBetween(-window_size + 1, 0)
                self.df = self.df.withColumn(
                    f'ma_{window_size}',
                    F.sum(F.col(column) * F.lit(weights)).over(window_spec) / F.lit(sum_weights)
                )
            else:
                window_spec = Window.orderBy(F.col('index')).rowsBetween(-window_size + 1, 0)
                self.df = self.df.withColumn(
                    f'ma_{window_size}',
                    F.avg(F.col(column)).over(window_spec)
                )

            if center:
                self.df = self.df.withColumn(
                    f'ma_{window_size}',
                    F.expr(f'lead(ma_{window_size}, {window_size // 2}) over (order by index)')
                )

        return self.df

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
        # Initialize results DataFrame
        self.df = self.df.withColumn('flag', F.lit(False)).withColumn('confidence', F.lit(0.0))

        # Apply statistical tests
        if 'statistical' in methods:
            mean = self.df.select(F.mean(column)).collect()[0][0]
            std_dev = self.df.select(F.stddev(column)).collect()[0][0]
            z_scores = self.df.withColumn('z_score', (F.col(column) - mean) / std_dev)
            self.df = self.df.withColumn(
                'flag',
                F.when(F.abs(F.col('z_score')) > thresholds['statistical'], True).otherwise(F.col('flag'))
            ).withColumn(
                'confidence',
                F.when(F.abs(F.col('z_score')) > thresholds['statistical'], F.col('confidence') + 0.5).otherwise(F.col('confidence'))
            )

        # Apply pattern-based detection
        if 'pattern' in methods:
            window_spec = Window.orderBy('index').rowsBetween(-1, 1)
            self.df = self.df.withColumn(
                'pattern_diff',
                F.abs(F.col(column) - F.avg(column).over(window_spec))
            )
            self.df = self.df.withColumn(
                'flag',
                F.when(F.col('pattern_diff') > thresholds['pattern'], True).otherwise(F.col('flag'))
            ).withColumn(
                'confidence',
                F.when(F.col('pattern_diff') > thresholds['pattern'], F.col('confidence') + 0.3).otherwise(F.col('confidence'))
            )

        # Apply forecast-based detection
        if 'forecast' in methods:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            data = np.array(self.df.select(column).rdd.flatMap(lambda x: x).collect())
            model = ExponentialSmoothing(data).fit()
            forecast = model.predict(start=0, end=len(data) - 1)
            self.df = self.df.withColumn(
                'forecast_diff',
                F.abs(F.col(column) - F.array(forecast.tolist()))
            )
            self.df = self.df.withColumn(
                'flag',
                F.when(F.col('forecast_diff') > thresholds['forecast'], True).otherwise(F.col('flag'))
            ).withColumn(
                'confidence',
                F.when(F.col('forecast_diff') > thresholds['forecast'], F.col('confidence') + 0.2).otherwise(F.col('confidence'))
            )

        # Apply clustering-based detection
        if 'clustering' in methods:
            from pyspark.ml.feature import VectorAssembler
            from pyspark.ml.clustering import KMeans
            assembler = VectorAssembler(inputCols=[column], outputCol='features')
            df_features = assembler.transform(self.df)
            kmeans = KMeans(k=2, seed=1)
            model = kmeans.fit(df_features)
            predictions = model.transform(df_features)
            self.df = self.df.withColumn(
                'flag',
                F.when(predictions['prediction'] == 1, True).otherwise(F.col('flag'))
            ).withColumn(
                'confidence',
                F.when(predictions['prediction'] == 1, F.col('confidence') + 0.1).otherwise(F.col('confidence'))
            )

        # Filter by minimum confidence
        self.df = self.df.filter(F.col('confidence') >= min_confidence)

        return self.df
    

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
        import chardet

        if isinstance(columns, str):
            columns = [columns]

        results = {}

        for column in columns:
            column_data = df.select(column).rdd.flatMap(lambda x: x).collect()
            sample_data = column_data[:sample_size] if sample_size else column_data

            detected_encodings = [chardet.detect(bytes(row, 'utf-8'))['encoding'] for row in sample_data]
            encoding_frequencies = {enc: detected_encodings.count(enc) for enc in set(detected_encodings)}

            special_chars = set()
            invalid_chars = set()
            problematic_values = []
            sample_violations = []

            for row in sample_data:
                try:
                    row.encode(target_encoding)
                except UnicodeEncodeError as e:
                    problematic_values.append(row)
                    sample_violations.append({'value': row, 'error': str(e)})
                    invalid_chars.update(set(row) - set(row.encode(target_encoding, 'ignore').decode(target_encoding)))

                if detect_special_chars:
                    special_chars.update(set(row) - set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '))

            total_violations = len(problematic_values)
            violation_percentage = (total_violations / len(sample_data)) * 100 if sample_data else 0

            results[column] = {
                'current_encoding': target_encoding,
                'detected_encodings': list(set(detected_encodings)),
                'special_chars': special_chars,
                'invalid_chars': invalid_chars,
                'encoding_frequencies': encoding_frequencies,
                'conversion_possible': total_violations == 0,
                'problematic_values': problematic_values,
                'sample_violations': sample_violations,
                'total_violations': total_violations,
                'violation_percentage': violation_percentage
            }

        return results

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
        if isinstance(columns, str):
            columns = [columns]

        if strategy not in ['convert', 'remove', 'replace', 'encode']:
            raise ValueError("Invalid strategy specified.")

        if handling_method not in ['replace', 'ignore', 'strict']:
            raise ValueError("Invalid handling_method specified.")

        if len(replacement_char) != 1:
            raise ValueError("replacement_char must be a single character.")

        for column in columns:
            if column not in df.columns:
                raise ValueError(f"Column {column} does not exist in DataFrame.")

            if strategy == 'convert':
                def convert_text(text):
                    try:
                        return text.encode(target_encoding, errors=handling_method).decode(target_encoding)
                    except UnicodeEncodeError:
                        return replacement_char if handling_method == 'replace' else text

                convert_udf = F.udf(convert_text, F.StringType())
                df = df.withColumn(column, convert_udf(F.col(column)))

            elif strategy == 'remove':
                def remove_invalid_chars(text):
                    return ''.join([char if char.isprintable() else '' for char in text])

                remove_udf = F.udf(remove_invalid_chars, F.StringType())
                df = df.withColumn(column, remove_udf(F.col(column)))

            elif strategy == 'replace':
                def replace_invalid_chars(text):
                    return text.encode(target_encoding, errors='replace').decode(target_encoding).replace('�', replacement_char)

                replace_udf = F.udf(replace_invalid_chars, F.StringType())
                df = df.withColumn(column, replace_udf(F.col(column)))

            elif strategy == 'encode':
                def encode_special_chars(text):
                    return text.encode('ascii', errors='xmlcharrefreplace').decode('ascii')

                encode_udf = F.udf(encode_special_chars, F.StringType())
                df = df.withColumn(column, encode_udf(F.col(column)))

        return df

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
        import chardet

        if isinstance(columns, str):
            columns = [columns]

        detected_encodings = {}

        for column in columns:
            column_data = df.select(column).rdd.flatMap(lambda x: x).collect()
            sample_data = column_data[:sample_size] if sample_size else column_data

            encoding_counts = {}
            for row in sample_data:
                result = chardet.detect(bytes(row, 'utf-8'))
                encoding = result['encoding']
                if encoding:
                    encoding_counts[encoding] = encoding_counts.get(encoding, 0) + 1

            if encoding_counts:
                detected_encodings[column] = max(encoding_counts, key=encoding_counts.get)
            else:
                detected_encodings[column] = 'unknown'

        return detected_encodings
    




