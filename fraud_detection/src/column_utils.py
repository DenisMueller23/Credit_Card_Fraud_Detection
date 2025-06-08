import pandas as pd
import numpy as np
from typing import List, Set, Dict
import logging
import re

logger = logging.getLogger(__name__)

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names by replacing hyphens with underscores and 
    ensuring consistent naming conventions.
    """
    df_copy = df.copy()
    
    # Replace hyphens with underscores
    df_copy.columns = df_copy.columns.str.replace('-', '_')
    
    # Remove any leading/trailing whitespace
    df_copy.columns = df_copy.columns.str.strip()
    
    # Optionally, you could add more standardization rules here
    # df_copy.columns = df_copy.columns.str.lower()  # Convert to lowercase
    
    return df_copy

def validate_column_consistency(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                              exclude_target: bool = True, target_col: str = 'isFraud') -> bool:
    """
    Validate that train and test datasets have consistent column structure.
    
    Args:
        train_df: Training dataset
        test_df: Test dataset  
        exclude_target: Whether to exclude target column from validation
        target_col: Name of target column to exclude
        
    Returns:
        bool: True if columns are consistent, False otherwise
    """
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    
    # Remove target column from train set if specified
    if exclude_target and target_col in train_cols:
        train_cols.remove(target_col)
    
    # Check for differences
    missing_in_test = train_cols - test_cols
    extra_in_test = test_cols - train_cols
    
    if missing_in_test:
        logger.warning(f"Missing columns in test set: {list(missing_in_test)[:10]}")
        print(f"❌ Missing in test: {list(missing_in_test)[:5]}")
        
    if extra_in_test:
        logger.warning(f"Extra columns in test set: {list(extra_in_test)[:10]}")
        print(f"⚠️  Extra in test: {list(extra_in_test)[:5]}")
    
    is_consistent = len(missing_in_test) == 0 and len(extra_in_test) == 0
    
    if is_consistent:
        print("✅ Column consistency validated")
    else:
        print(f"❌ Column inconsistency detected")
        
    return is_consistent

def get_column_differences(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                          target_col: str = 'isFraud') -> Dict[str, List[str]]:
    """
    Get detailed column differences between train and test sets.
    """
    train_cols = set(train_df.columns)
    if target_col in train_cols:
        train_cols.remove(target_col)
    test_cols = set(test_df.columns)
    
    return {
        'missing_in_test': list(train_cols - test_cols),
        'extra_in_test': list(test_cols - train_cols),
        'common_columns': list(train_cols & test_cols)
    }

def align_column_order(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                      target_col: str = 'isFraud') -> pd.DataFrame:
    """
    Align test dataset column order to match training dataset.
    """
    train_cols = [col for col in train_df.columns if col != target_col]
    
    # Only keep columns that exist in test set
    available_cols = [col for col in train_cols if col in test_df.columns]
    
    return test_df[available_cols]