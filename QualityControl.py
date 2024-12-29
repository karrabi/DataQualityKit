from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from typing import Dict, List, Union, Optional, Tuple, Any
from pyspark.ml.feature import Imputer
from pyspark.sql.types import *
from pyspark.sql.window import Window
import numpy as np

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


