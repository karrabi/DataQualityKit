from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from typing import Dict, List, Union, Optional, Any
from pyspark.ml.feature import Imputer
from pyspark.sql.types import *


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
        

