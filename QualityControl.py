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