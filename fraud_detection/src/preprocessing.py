import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
warnings.filterwarnings('ignore')

class DataCleaner:
    """
    Comprehensive data cleaning class for credit default datasets
    """
    
    def __init__(self, data_path=None, df=None):
        """
        Initialize the DataCleaner
        
        Parameters:
        data_path (str): Path to the dataset
        df (DataFrame): Pre-loaded DataFrame
        """
        if df is not None:
            self.df = df.copy()
        elif data_path:
            self.df = pd.read_csv(data_path)
        else:
            raise ValueError("Either data_path or df must be provided")
        
        self.original_shape = self.df.shape
        self.cleaning_log = []
        
    def log_step(self, step_description, before_shape, after_shape):
        """Log cleaning steps for tracking"""
        self.cleaning_log.append({
            'step': step_description,
            'before_shape': before_shape,
            'after_shape': after_shape,
            'rows_removed': before_shape[0] - after_shape[0],
            'columns_removed': before_shape[1] - after_shape[1]
        })
    
    def initial_assessment(self):
        """
        Perform initial data quality assessment
        """
        print("=== INITIAL DATA ASSESSMENT ===")
        print(f"Dataset shape: {self.df.shape}")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Basic info
        print("\n--- Data Types ---")
        print(self.df.dtypes.value_counts())
        
        # Missing values
        print("\n--- Missing Values ---")
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing_Count': missing_data,
            'Missing_Percentage': missing_percent
        }).sort_values('Missing_Percentage', ascending=False)
        
        print(missing_df[missing_df['Missing_Count'] > 0])
        
        # Duplicates
        duplicates = self.df.duplicated().sum()
        print(f"\n--- Duplicates ---")
        print(f"Duplicate rows: {duplicates} ({duplicates/len(self.df)*100:.2f}%)")
        
        return missing_df
    
    def remove_duplicates(self):
        """
        Remove duplicate records
        """
        before_shape = self.df.shape
        initial_duplicates = self.df.duplicated().sum()
        
        if initial_duplicates > 0:
            self.df = self.df.drop_duplicates()
            after_shape = self.df.shape
            self.log_step("Remove duplicates", before_shape, after_shape)
            print(f"Removed {initial_duplicates} duplicate rows")
        else:
            print("No duplicate rows found")
    
    def handle_missing_values(self, strategy_dict=None):
        """
        Handle missing values with different strategies for different columns
        
        Parameters:
        strategy_dict (dict): Dictionary mapping column names to strategies
        """
        if strategy_dict is None:
            # Default strategies based on common patterns
            strategy_dict = self._determine_missing_strategies()
        
        before_shape = self.df.shape
        
        for column, strategy in strategy_dict.items():
            if column not in self.df.columns:
                continue
                
            missing_count = self.df[column].isnull().sum()
            if missing_count == 0:
                continue
            
            if strategy == 'drop_column':
                self.df = self.df.drop(columns=[column])
                print(f"Dropped column '{column}' (too many missing values)")
                
            elif strategy == 'drop_rows':
                self.df = self.df.dropna(subset=[column])
                print(f"Dropped rows with missing '{column}'")
                
            elif strategy == 'mode':
                mode_value = self.df[column].mode().iloc[0] if not self.df[column].mode().empty else 'Unknown'
                self.df[column] = self.df[column].fillna(mode_value)
                print(f"Filled '{column}' missing values with mode: {mode_value}")
                
            elif strategy == 'median':
                median_value = self.df[column].median()
                self.df[column] = self.df[column].fillna(median_value)
                print(f"Filled '{column}' missing values with median: {median_value}")
                
            elif strategy == 'mean':
                mean_value = self.df[column].mean()
                self.df[column] = self.df[column].fillna(mean_value)
                print(f"Filled '{column}' missing values with mean: {mean_value:.2f}")
                
            elif strategy == 'zero':
                self.df[column] = self.df[column].fillna(0)
                print(f"Filled '{column}' missing values with 0")
                
            elif strategy == 'unknown':
                self.df[column] = self.df[column].fillna('Unknown')
                print(f"Filled '{column}' missing values with 'Unknown'")
        
        after_shape = self.df.shape
        self.log_step("Handle missing values", before_shape, after_shape)
    
    def _determine_missing_strategies(self):
        """
        Automatically determine missing value strategies based on data types and patterns
        """
        strategies = {}
        missing_threshold = 0.5  # Drop columns with >50% missing
        
        for column in self.df.columns:
            missing_pct = self.df[column].isnull().sum() / len(self.df)
            
            if missing_pct > missing_threshold:
                strategies[column] = 'drop_column'
            elif self.df[column].dtype == 'object':
                strategies[column] = 'mode'
            elif self.df[column].dtype in ['int64', 'float64']:
                # Use median for numerical data (more robust to outliers)
                strategies[column] = 'median'
            else:
                strategies[column] = 'mode'
        
        return strategies
    
    def detect_outliers(self, method='iqr', columns=None):
        """
        Detect outliers using specified method
        
        Parameters:
        method (str): 'iqr', 'zscore', or 'isolation'
        columns (list): Columns to check for outliers
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        outliers_dict = {}
        
        for column in columns:
            if column not in self.df.columns:
                continue
                
            if method == 'iqr':
                Q1 = self.df[column].quantile(0.25)
                Q3 = self.df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)].index
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(self.df[column].dropna()))
                outliers = self.df[column].dropna().index[z_scores > 3]
            
            outliers_dict[column] = outliers
            print(f"Column '{column}': {len(outliers)} outliers detected ({len(outliers)/len(self.df)*100:.2f}%)")
        
        return outliers_dict
    
    def handle_outliers(self, outliers_dict, method='cap'):
        """
        Handle outliers using specified method
        
        Parameters:
        outliers_dict (dict): Dictionary of outliers from detect_outliers
        method (str): 'remove', 'cap', or 'transform'
        """
        before_shape = self.df.shape
        
        if method == 'remove':
            # Remove rows with outliers in any column
            all_outlier_indices = set()
            for indices in outliers_dict.values():
                all_outlier_indices.update(indices)
            
            self.df = self.df.drop(index=list(all_outlier_indices))
            print(f"Removed {len(all_outlier_indices)} rows containing outliers")
            
        elif method == 'cap':
            # Cap outliers at 1st and 99th percentiles
            for column in outliers_dict.keys():
                if column in self.df.columns:
                    lower_cap = self.df[column].quantile(0.01)
                    upper_cap = self.df[column].quantile(0.99)
                    
                    self.df[column] = self.df[column].clip(lower=lower_cap, upper=upper_cap)
                    print(f"Capped outliers in '{column}' between {lower_cap:.2f} and {upper_cap:.2f}")
        
        after_shape = self.df.shape
        self.log_step(f"Handle outliers ({method})", before_shape, after_shape)
    
    def optimize_data_types(self):
        """
        Optimize data types for memory efficiency
        """
        before_memory = self.df.memory_usage(deep=True).sum() / 1024**2
        
        # Convert object columns with few unique values to category
        for column in self.df.select_dtypes(include=['object']).columns:
            unique_ratio = self.df[column].nunique() / len(self.df)
            if unique_ratio < 0.5:  # Less than 50% unique values
                self.df[column] = self.df[column].astype('category')
        
        # Optimize integer columns
        for column in self.df.select_dtypes(include=['int64']).columns:
            col_min = self.df[column].min()
            col_max = self.df[column].max()
            
            if col_min >= 0:
                if col_max < 255:
                    self.df[column] = self.df[column].astype('uint8')
                elif col_max < 65535:
                    self.df[column] = self.df[column].astype('uint16')
                elif col_max < 4294967295:
                    self.df[column] = self.df[column].astype('uint32')
            else:
                if col_min > -128 and col_max < 127:
                    self.df[column] = self.df[column].astype('int8')
                elif col_min > -32768 and col_max < 32767:
                    self.df[column] = self.df[column].astype('int16')
                elif col_min > -2147483648 and col_max < 2147483647:
                    self.df[column] = self.df[column].astype('int32')
        
        # Optimize float columns
        for column in self.df.select_dtypes(include=['float64']).columns:
            self.df[column] = self.df[column].astype('float32')
        
        after_memory = self.df.memory_usage(deep=True).sum() / 1024**2
        print(f"Memory usage reduced from {before_memory:.2f} MB to {after_memory:.2f} MB")
        print(f"Memory reduction: {(before_memory - after_memory)/before_memory*100:.2f}%")
    
    def encode_datetime_features(self, datetime_column='TransactionDT'):
        """
        Encode datetime features from TransactionDT column
        
        Parameters:
        datetime_column (str): Name of the datetime column to encode
        """
        before_shape = self.df.shape
        
        if datetime_column not in self.df.columns:
            print(f"Warning: Column '{datetime_column}' not found")
            return
        
        print(f"=== ENCODING DATETIME FEATURES: {datetime_column} ===")
        
        # Convert to datetime if it's not already
        if self.df[datetime_column].dtype != 'datetime64[ns]':
            # Try different conversion methods
            try:
                # If it's Unix timestamp (seconds since epoch)
                self.df[datetime_column] = pd.to_datetime(self.df[datetime_column], unit='s')
            except:
                try:
                    # If it's already in datetime format
                    self.df[datetime_column] = pd.to_datetime(self.df[datetime_column])
                except:
                    print(f"Could not convert {datetime_column} to datetime")
                    return
        
        # Extract useful datetime features
        dt_col = self.df[datetime_column]
        
        # Basic time components
        self.df[f'{datetime_column}_hour'] = dt_col.dt.hour
        self.df[f'{datetime_column}_day'] = dt_col.dt.day
        self.df[f'{datetime_column}_month'] = dt_col.dt.month
        self.df[f'{datetime_column}_year'] = dt_col.dt.year
        self.df[f'{datetime_column}_dayofweek'] = dt_col.dt.dayofweek  # 0=Monday, 6=Sunday
        self.df[f'{datetime_column}_dayofyear'] = dt_col.dt.dayofyear
        
        # Cyclical encoding for better ML performance
        # Hour (0-23)
        self.df[f'{datetime_column}_hour_sin'] = np.sin(2 * np.pi * dt_col.dt.hour / 24)
        self.df[f'{datetime_column}_hour_cos'] = np.cos(2 * np.pi * dt_col.dt.hour / 24)
        
        # Day of week (0-6)
        self.df[f'{datetime_column}_dow_sin'] = np.sin(2 * np.pi * dt_col.dt.dayofweek / 7)
        self.df[f'{datetime_column}_dow_cos'] = np.cos(2 * np.pi * dt_col.dt.dayofweek / 7)
        
        # Month (1-12)
        self.df[f'{datetime_column}_month_sin'] = np.sin(2 * np.pi * dt_col.dt.month / 12)
        self.df[f'{datetime_column}_month_cos'] = np.cos(2 * np.pi * dt_col.dt.month / 12)
        
        # Business vs weekend
        self.df[f'{datetime_column}_is_weekend'] = (dt_col.dt.dayofweek >= 5).astype(int)
        
        # Time of day categories
        hour = dt_col.dt.hour
        self.df[f'{datetime_column}_time_of_day'] = pd.cut(hour, 
                                                          bins=[0, 6, 12, 18, 24], 
                                                          labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                                                          include_lowest=True)
        
        # Time since first transaction (relative timing)
        min_dt = dt_col.min()
        self.df[f'{datetime_column}_days_since_first'] = (dt_col - min_dt).dt.days
        self.df[f'{datetime_column}_seconds_since_first'] = (dt_col - min_dt).dt.total_seconds()
        
        # Drop original datetime column if desired (optional)
        # self.df = self.df.drop(columns=[datetime_column])
        
        after_shape = self.df.shape
        self.log_step(f"Encode datetime features from {datetime_column}", before_shape, after_shape)
        
        print(f"Created {after_shape[1] - before_shape[1]} new datetime features")
        print("New datetime features:")
        new_cols = [col for col in self.df.columns if datetime_column in col and col != datetime_column]
        for col in new_cols:
            print(f"  - {col}")

    
    def validate_data(self):
        """
        Perform data validation checks
        """
        print("=== DATA VALIDATION ===")
        
        # Check for negative values in columns that shouldn't have them
        amount_columns = [col for col in self.df.columns if 'amount' in col.lower() or 'balance' in col.lower()]
        for column in amount_columns:
            if column in self.df.columns:
                negative_count = (self.df[column] < 0).sum()
                if negative_count > 0:
                    print(f"Warning: {negative_count} negative values in '{column}'")
        
        # Check for dates in the future (if any date columns exist)
        date_columns = self.df.select_dtypes(include=['datetime64']).columns
        for column in date_columns:
            future_dates = (self.df[column] > pd.Timestamp.now()).sum()
            if future_dates > 0:
                print(f"Warning: {future_dates} future dates in '{column}'")
        
        # Check for suspicious patterns
        print("\n--- Final Data Summary ---")
        print(f"Final shape: {self.df.shape}")
        print(f"Rows removed: {self.original_shape[0] - self.df.shape[0]}")
        print(f"Columns removed: {self.original_shape[1] - self.df.shape[1]}")
    
    def create_cleaning_report(self):
        """
        Generate a comprehensive cleaning report
        """
        print("\n=== CLEANING REPORT ===")
        for step in self.cleaning_log:
            print(f"Step: {step['step']}")
            print(f"  Before: {step['before_shape']}")
            print(f"  After: {step['after_shape']}")
            print(f"  Rows removed: {step['rows_removed']}")
            print(f"  Columns removed: {step['columns_removed']}")
            print()
    
    def execute_cleaning_pipeline(self, custom_strategies=None):
        """
        Execute the complete cleaning pipeline
        
        Parameters:
        custom_strategies (dict): Custom missing value strategies
        """
        print("Starting comprehensive data cleaning pipeline...")
        
        # Step 1: Initial assessment
        missing_df = self.initial_assessment()
        
        # Step 2: Remove duplicates
        print("\n=== STEP 1: REMOVING DUPLICATES ===")
        self.remove_duplicates()
        
        # Step 3: Handle missing values
        print("\n=== STEP 2: HANDLING MISSING VALUES ===")
        self.handle_missing_values(custom_strategies)
        
        # Step 4: Encode datetime features (NEW STEP)
        print("\n=== STEP 3: ENCODING DATETIME FEATURES ===")
        self.encode_datetime_features('TransactionDT')
        
        # Step 5: Detect and handle outliers
        print("\n=== STEP 4: HANDLING OUTLIERS ===")
        outliers = self.detect_outliers(method='iqr')
        self.handle_outliers(outliers, method='cap')
        
        # Step 6: Optimize data types
        print("\n=== STEP 5: OPTIMIZING DATA TYPES ===")
        self.optimize_data_types()
        
        # Step 7: Validate data
        print("\n=== STEP 6: DATA VALIDATION ===")
        self.validate_data()
        
        # Step 8: Generate report
        self.create_cleaning_report()
        
        return self.df

# Example usage function
def clean_credit_data(data_path, output_path=None):
    """
    Main function to clean credit default data
    
    Parameters:
    data_path (str): Path to raw data
    output_path (str): Path to save cleaned data
    """
    # Initialize cleaner
    cleaner = DataCleaner(data_path=data_path)
    
    # Define custom strategies if needed
    custom_strategies = {
        # Example custom strategies - adjust based on your specific dataset
        # 'customer_id': 'drop_rows',  # Critical field, drop if missing
        # 'income': 'median',          # Use median for income
        # 'employment_type': 'mode',   # Use mode for categorical
    }
    
    # Execute cleaning pipeline
    cleaned_df = cleaner.execute_cleaning_pipeline(custom_strategies)
    
    # Save cleaned data
    if output_path:
        cleaned_df.to_csv(output_path, index=False)
        print(f"\nCleaned data saved to: {output_path}")
    
    return cleaned_df, cleaner



# Example of how to use the cleaner
if __name__ == "__main__":
    # Example usage - uncomment and modify paths as needed
    
    # Load and clean data
    # data_path = "../raw/credit_data.csv"
    # output_path = "cleaned_credit_data.csv"
    # cleaned_data, cleaner_instance = clean_credit_data(data_path, output_path)
    
    # Or use with existing DataFrame
    # cleaner = DataCleaner(df=your_dataframe)
    # cleaned_df = cleaner.execute_cleaning_pipeline()
    
    pass