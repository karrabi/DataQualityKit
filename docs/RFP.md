List of column-level data quality checks and their corresponding remediation actions:

1. Missing Values
- ListAll: list all columns of dataset and shows number and percentage of missnig values
- Check: Null count, empty strings, whitespace-only values
- Actions:
  * Delete rows if missing data is critical
  * Impute with mean/median/mode
  * Use ML-based imputation
  * Replace with default values
  * Flag for manual review

2. Data Type Conformity
- Check: Incorrect data types, mixed types in column
- Actions:
  * Convert to correct data type
  * Parse structured strings
  * Remove non-conforming characters
  * Split mixed data into separate columns

3. Range Validity
- Check: Values outside expected min/max, outliers, impossible values
- Actions:
  * Cap at valid boundaries
  * Remove outliers
  * Apply statistical transformations
  * Flag for domain expert review

4. Format Consistency
- Check: Dates, phone numbers, emails, addresses format violations
- Actions:
  * Standardize formats
  * Apply regex transformations
  * Use lookup tables for standardization
  * Parse and reconstruct

5. Business Rule Compliance
- Check: Domain-specific rules violations
- Actions:
  * Apply business logic corrections
  * Use mapping tables
  * Replace with valid alternatives
  * Quarantine violations

6. Duplicate Values
- Check: Exact duplicates, fuzzy matches, business-key duplicates
- Actions:
  * Remove exact duplicates
  * Merge similar records
  * Keep most recent/complete version
  * Create composite keys

7. Categorical Validity
- Check: Invalid categories, misspellings, case inconsistency
- Actions:
  * Map to standard categories
  * Fuzzy matching correction
  * Standardize case
  * Group rare categories

8. Statistical Anomalies
- Check: Distribution anomalies, sudden changes, pattern breaks
- Actions:
  * Apply statistical smoothing
  * Remove statistical outliers
  * Use moving averages
  * Flag for investigation

9. Cross-Column Consistency
- Check: Logical relationships, derived value accuracy
- Actions:
  * Recalculate derived values
  * Enforce relationship rules
  * Correct dependent fields
  * Remove inconsistent rows

10. Encoding Issues
- Check: Character encoding problems, special characters
- Actions:
  * Convert to standard encoding
  * Remove invalid characters
  * Replace with encoded equivalents
  * Standardize to UTF-8