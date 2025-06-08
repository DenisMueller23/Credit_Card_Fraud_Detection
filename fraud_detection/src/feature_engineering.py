"""
Advanced Feature Engineering Pipeline for Fraud Detection

This module implements a comprehensive, production-ready feature engineering pipeline
following enterprise data science best practices. All transformations are designed
to prevent data leakage and ensure reproducibility.

Author: Denis Müller
Version: 2.0.0
"""

import logging
import pickle
import warnings
import numpy as np
import pandas as pd
import traceback
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler

# Import from column_utils
from column_utils import standardize_column_names

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", message=".*Columns not found.*")

# Set debug mode
DEBUG_MODE = True

def debug_log(message: str, level: str = "INFO"):
    """Enhanced debugging function"""
    if DEBUG_MODE:
        print(f"[{level}] {message}")
        if level == "ERROR":
            traceback.print_exc()


@dataclass
class FeatureConfig:
    """Configuration class for feature engineering parameters."""
    
    # Temporal feature parameters
    velocity_windows: List[int] = field(default_factory=lambda: [1, 24, 168])  # hours
    cyclical_features: List[str] = field(default_factory=lambda: ['hour', 'day'])
    
    # Amount feature parameters
    amount_percentiles: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.75, 0.95])
    outlier_threshold: float = 2.0
    
    # Frequency encoding parameters
    bayesian_alpha: float = 10.0
    min_category_frequency: int = 5
    
    # Anomaly detection parameters
    isolation_contamination: float = 0.1
    zscore_threshold: float = 3.0
    
    # Processing parameters
    random_state: int = 42
    n_jobs: int = -1
    chunk_size: int = 10000


class BaseFeatureTransformer(BaseEstimator, TransformerMixin, ABC):
    """Abstract base class for all feature transformers."""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.is_fitted_ = False
        self.feature_names_out_ = []
        
    @abstractmethod
    def _fit_transform_train(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform training data."""
        pass
    
    @abstractmethod
    def _transform_test(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform test data using fitted parameters."""
        pass
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'BaseFeatureTransformer':
        """Fit the transformer on training data."""
        logger.info(f"Fitting {self.__class__.__name__}")
        self._validate_input(X)
        self._fit_transform_train(X, y)
        self.is_fitted_ = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted parameters."""
        if not self.is_fitted_:
            raise ValueError(f"{self.__class__.__name__} must be fitted before transform")
        
        self._validate_input(X)
        return self._transform_test(X)
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y)._fit_transform_train(X, y)
    
    def _validate_input(self, X: pd.DataFrame) -> None:
        """Validate input data."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if X.empty:
            raise ValueError("Input DataFrame is empty")


class TemporalFeatureTransformer(BaseFeatureTransformer):
    """Advanced temporal feature engineering with velocity calculations."""
    
    def __init__(self, config: FeatureConfig, time_col: str = 'TransactionDT'):
        super().__init__(config)
        self.time_col = time_col
        self.velocity_stats_ = {}
        
    def _fit_transform_train(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform training data."""
        X_transformed = X.copy()
        
        # Ensure time column exists and is numeric
        if self.time_col not in X_transformed.columns:
            logger.warning(f"Time column '{self.time_col}' not found, skipping temporal features")
            return X_transformed
            
        X_transformed[self.time_col] = pd.to_numeric(X_transformed[self.time_col], errors='coerce')
        
        # Basic temporal features
        X_transformed = self._create_basic_temporal_features(X_transformed)
        
        # Velocity features (computationally intensive - optimized)
        X_transformed = self._create_velocity_features(X_transformed, is_training=True)
        
        # Cyclical encoding
        X_transformed = self._create_cyclical_features(X_transformed)
        
        # Time-based statistics
        X_transformed = self._create_temporal_statistics(X_transformed)
        
        self.feature_names_out_ = [col for col in X_transformed.columns if col not in X.columns]
        logger.info(f"Created {len(self.feature_names_out_)} temporal features")
        
        return X_transformed
    
    def _transform_test(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform test data using fitted parameters."""
        X_transformed = X.copy()
        
        # Check if time column exists
        if self.time_col not in X_transformed.columns:
            logger.warning(f"Time column '{self.time_col}' not found, skipping temporal features")
            return X_transformed
            
        X_transformed[self.time_col] = pd.to_numeric(X_transformed[self.time_col], errors='coerce')
        
        # Apply same transformations as training
        X_transformed = self._create_basic_temporal_features(X_transformed)
        X_transformed = self._create_velocity_features(X_transformed, is_training=False)
        X_transformed = self._create_cyclical_features(X_transformed)
        X_transformed = self._create_temporal_statistics(X_transformed)
        
        return X_transformed
    
    def _create_basic_temporal_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create basic time-based features."""
        try:
            # Time decomposition
            X[f'{self.time_col}_hour'] = (X[self.time_col] / 3600) % 24
            X[f'{self.time_col}_day'] = (X[self.time_col] / (3600 * 24)) % 7
            X[f'{self.time_col}_week'] = X[self.time_col] / (3600 * 24 * 7)
            
            # Ensure numeric types
            for col in [f'{self.time_col}_hour', f'{self.time_col}_day', f'{self.time_col}_week']:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
            
            # Temporal patterns
            X['is_weekend'] = X[f'{self.time_col}_day'].isin([5, 6]).astype(np.int8)
            X['is_night'] = ((X[f'{self.time_col}_hour'] >= 22) | 
                            (X[f'{self.time_col}_hour'] <= 6)).astype(np.int8)
            X['is_business_hours'] = ((X[f'{self.time_col}_hour'] >= 9) & 
                                     (X[f'{self.time_col}_hour'] <= 17)).astype(np.int8)
            X['is_peak_hours'] = X[f'{self.time_col}_hour'].isin([12, 13, 18, 19, 20]).astype(np.int8)
            
            # Additional time features
            X['days_since_start'] = X[self.time_col] / (3600 * 24)
            X['hour_of_week'] = X[f'{self.time_col}_day'] * 24 + X[f'{self.time_col}_hour']
            
        except Exception as e:
            logger.warning(f"Error creating basic temporal features: {e}")
        
        return X
    
    def _create_velocity_features(self, X: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Create optimized velocity features."""
        if 'card1' not in X.columns:
            logger.warning("card1 column not found, skipping velocity features")
            return X
        
        try:
            # Sort by card and time for efficient calculation
            X_sorted = X.sort_values(['card1', self.time_col]).reset_index(drop=True)
            
            for hours in self.config.velocity_windows:
                window_seconds = hours * 3600
                col_name = f'velocity_{hours}h'
                
                if is_training:
                    # Calculate velocity using rolling window approach
                    velocities = self._calculate_velocities_optimized(
                        X_sorted, window_seconds, self.time_col
                    )
                    X_sorted[col_name] = velocities
                    
                    # Store statistics for test set
                    velocity_stats = X_sorted.groupby('card1')[col_name].agg(['mean', 'std']).reset_index()
                    self.velocity_stats_[col_name] = velocity_stats
                else:
                    # Use pre-calculated statistics for unseen cards
                    velocities = self._calculate_velocities_optimized(
                        X_sorted, window_seconds, self.time_col
                    )
                    X_sorted[col_name] = velocities
                    
                    # Fill missing velocities with training statistics
                    if col_name in self.velocity_stats_:
                        stats_df = self.velocity_stats_[col_name]
                        global_mean = stats_df['mean'].mean()
                        X_sorted[col_name].fillna(global_mean, inplace=True)
            
            # Time differences
            X_sorted['time_since_last_txn'] = X_sorted.groupby('card1')[self.time_col].diff().fillna(0)
            X_sorted['time_to_next_txn'] = X_sorted.groupby('card1')[self.time_col].diff(-1).abs().fillna(0)
            
            # Restore original order
            return X_sorted.sort_index()
            
        except Exception as e:
            logger.warning(f"Error creating velocity features: {e}")
            return X
    
    def _calculate_velocities_optimized(self, X: pd.DataFrame, window_seconds: int, 
                                      time_col: str) -> pd.Series:
        """Optimized velocity calculation using vectorized operations."""
        try:
            velocities = np.ones(len(X), dtype=np.int32)
            
            # Group by card for efficient processing
            for card_id, group_idx in X.groupby('card1').groups.items():
                if pd.isna(card_id):
                    continue
                    
                group_times = X.loc[group_idx, time_col].values
                group_velocities = np.ones(len(group_times), dtype=np.int32)
                
                for i, current_time in enumerate(group_times):
                    if pd.isna(current_time):
                        continue
                        
                    # Count transactions in window using vectorized operations
                    time_window_start = current_time - window_seconds
                    in_window = (group_times <= current_time) & (group_times > time_window_start)
                    group_velocities[i] = np.sum(in_window)
                
                velocities[group_idx] = group_velocities
            
            return pd.Series(velocities, index=X.index)
            
        except Exception as e:
            logger.warning(f"Error calculating velocities: {e}")
            return pd.Series(np.ones(len(X)), index=X.index)
    
    def _create_cyclical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create cyclical encodings for temporal features."""
        try:
            for feature in self.config.cyclical_features:
                if feature == 'hour':
                    col = f'{self.time_col}_hour'
                    period = 24
                elif feature == 'day':
                    col = f'{self.time_col}_day'
                    period = 7
                else:
                    continue
                    
                if col in X.columns:
                    X[f'{feature}_sin'] = np.sin(2 * np.pi * X[col] / period)
                    X[f'{feature}_cos'] = np.cos(2 * np.pi * X[col] / period)
        except Exception as e:
            logger.warning(f"Error creating cyclical features: {e}")
        
        return X
    
    def _create_temporal_statistics(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create temporal statistics and patterns."""
        try:
            if 'card1' in X.columns:
                # User temporal consistency
                for col in [f'{self.time_col}_hour', 'is_weekend']:
                    if col in X.columns:
                        user_std = X.groupby('card1')[col].std().reset_index()
                        user_std.columns = ['card1', f'{col}_user_std']
                        X = X.merge(user_std, on='card1', how='left')
                        X[f'{col}_consistency'] = 1 / (X[f'{col}_user_std'] + 1e-6)
        except Exception as e:
            logger.warning(f"Error creating temporal statistics: {e}")
        
        return X


class AmountFeatureTransformer(BaseFeatureTransformer):
    """Sophisticated amount-based feature engineering."""
    
    def __init__(self, config: FeatureConfig, amount_col: str = 'TransactionAmt'):
        super().__init__(config)
        self.amount_col = amount_col
        self.card_stats_ = {}
        self.global_stats_ = {}
        
    def _fit_transform_train(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform training data."""
        X_transformed = X.copy()
        
        # Check if amount column exists
        if self.amount_col not in X_transformed.columns:
            logger.warning(f"Amount column '{self.amount_col}' not found, skipping amount features")
            return X_transformed
        
        # Ensure amount column is numeric
        X_transformed[self.amount_col] = pd.to_numeric(X_transformed[self.amount_col], errors='coerce').fillna(0)
        
        # Basic transformations
        X_transformed = self._create_basic_amount_features(X_transformed)
        
        # Card-level statistics
        X_transformed = self._create_card_level_features(X_transformed, is_training=True)
        
        # Amount patterns and deviations
        X_transformed = self._create_amount_patterns(X_transformed)
        
        # Percentile-based features
        X_transformed = self._create_percentile_features(X_transformed)
        
        self.feature_names_out_ = [col for col in X_transformed.columns if col not in X.columns]
        logger.info(f"Created {len(self.feature_names_out_)} amount features")
        
        return X_transformed
    
    def _transform_test(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform test data using fitted parameters."""
        X_transformed = X.copy()
        
        # Check if amount column exists
        if self.amount_col not in X_transformed.columns:
            logger.warning(f"Amount column '{self.amount_col}' not found, skipping amount features")
            return X_transformed
        
        # Ensure amount column is numeric
        X_transformed[self.amount_col] = pd.to_numeric(X_transformed[self.amount_col], errors='coerce').fillna(0)
        
        # Apply same transformations
        X_transformed = self._create_basic_amount_features(X_transformed)
        X_transformed = self._create_card_level_features(X_transformed, is_training=False)
        X_transformed = self._create_amount_patterns(X_transformed)
        X_transformed = self._create_percentile_features(X_transformed)
        
        return X_transformed
    
    def _create_basic_amount_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create basic amount transformations."""
        try:
            # Mathematical transformations
            X[f'{self.amount_col}_log'] = np.log1p(X[self.amount_col])
            X[f'{self.amount_col}_sqrt'] = np.sqrt(X[self.amount_col])
            X[f'{self.amount_col}_cbrt'] = np.cbrt(X[self.amount_col])
            
            # Decimal analysis
            X[f'{self.amount_col}_decimal'] = X[self.amount_col] - X[self.amount_col].astype(int)
            X['is_round_amount'] = (X[f'{self.amount_col}_decimal'] == 0).astype(np.int8)
            
            # Pattern recognition
            X['amount_digits'] = X[self.amount_col].astype(str).str.len()
            X['is_even_amount'] = (X[self.amount_col] % 2 == 0).astype(np.int8)
            X['ends_with_00'] = (X[self.amount_col] % 100 == 0).astype(np.int8)
            X['ends_with_99'] = X[self.amount_col].astype(str).str.endswith('99').astype(np.int8)
            X['ends_with_95'] = X[self.amount_col].astype(str).str.endswith('95').astype(np.int8)
            
        except Exception as e:
            logger.warning(f"Error creating basic amount features: {e}")
        
        return X
    
    def _create_card_level_features(self, X: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Create card-level amount statistics."""
        if 'card1' not in X.columns:
            logger.warning("card1 column not found, skipping card-level features")
            return X
        
        try:
            if is_training:
                # Calculate card-level statistics
                card_stats = X.groupby('card1')[self.amount_col].agg([
                    'mean', 'std', 'min', 'max', 'count', 'median'
                ]).reset_index()
                card_stats.columns = ['card1', 'card_amt_mean', 'card_amt_std', 
                                    'card_amt_min', 'card_amt_max', 'card_amt_count', 'card_amt_median']
                
                # Add percentiles
                for percentile in self.config.amount_percentiles:
                    pct_name = f'card_amt_p{int(percentile*100)}'
                    card_pct = X.groupby('card1')[self.amount_col].quantile(percentile).reset_index()
                    card_pct.columns = ['card1', pct_name]
                    card_stats = card_stats.merge(card_pct, on='card1', how='left')
                
                # Store for test set
                self.card_stats_ = card_stats
                
                # Merge with original data
                X = X.merge(card_stats, on='card1', how='left')
            else:
                # Use pre-calculated statistics
                if hasattr(self, 'card_stats_') and not self.card_stats_.empty:
                    X = X.merge(self.card_stats_, on='card1', how='left')
                    
                    # Fill missing with global statistics
                    for col in self.card_stats_.columns:
                        if col != 'card1' and col in X.columns:
                            global_mean = self.card_stats_[col].mean()
                            X[col].fillna(global_mean, inplace=True)
            
            # Deviation features
            if 'card_amt_mean' in X.columns and 'card_amt_std' in X.columns:
                X['amt_deviation_from_mean'] = (
                    np.abs(X[self.amount_col] - X['card_amt_mean']) / 
                    (X['card_amt_std'] + 1e-6)
                )
                X['is_amt_outlier'] = (X['amt_deviation_from_mean'] > self.config.outlier_threshold).astype(np.int8)
                
        except Exception as e:
            logger.warning(f"Error creating card-level features: {e}")
        
        return X
    
    def _create_amount_patterns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create amount pattern features."""
        try:
            # Ranking within card
            if 'card1' in X.columns:
                X['amt_rank_within_card'] = X.groupby('card1')[self.amount_col].rank(pct=True)
            
            # Time-amount interactions (if temporal features exist)
            for time_feature in ['TransactionDT_hour', 'is_weekend', 'is_night', 'is_business_hours']:
                if time_feature in X.columns:
                    X[f'amt_{time_feature}_interaction'] = X[self.amount_col] * X[time_feature]
                    
        except Exception as e:
            logger.warning(f"Error creating amount patterns: {e}")
        
        return X
    
    def _create_percentile_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create percentile-based features."""
        try:
            # Global percentile ranking
            X[f'{self.amount_col}_global_rank'] = X[self.amount_col].rank(pct=True)
            
            # Binning based on percentiles
            percentile_bins = [0] + [np.percentile(X[self.amount_col], p * 100) 
                                   for p in self.config.amount_percentiles] + [np.inf]
            X[f'{self.amount_col}_bin'] = pd.cut(X[self.amount_col], bins=percentile_bins, 
                                               labels=False, include_lowest=True)
                                               
        except Exception as e:
            logger.warning(f"Error creating percentile features: {e}")
        
        return X


class FrequencyEncoder(BaseFeatureTransformer):
    """Frequency encoding with Bayesian smoothing."""
    
    def __init__(self, config: FeatureConfig, categorical_columns: List[str], 
                 target_col: str = 'isFraud'):
        super().__init__(config)
        self.categorical_columns = categorical_columns
        self.target_col = target_col
        self.encoding_maps_ = {}
        self.global_fraud_rate_ = 0.0
        
    def _fit_transform_train(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform training data."""
        if y is None and self.target_col not in X.columns:
            logger.warning(f"Target column '{self.target_col}' not found, skipping frequency encoding")
            return X
        
        X_transformed = X.copy()
        target = y if y is not None else X[self.target_col]
        self.global_fraud_rate_ = target.mean()
        
        # Filter categorical columns to only include valid ones
        valid_categorical_columns = []
        for col in self.categorical_columns:
            if col in X_transformed.columns:
                # Check if column has valid categorical data
                unique_count = X_transformed[col].nunique()
                total_count = len(X_transformed[col])
                
                # Skip if too many unique values (likely not categorical)
                if unique_count > total_count * 0.5 and unique_count > 50:
                    logger.info(f"Skipping {col}: too many unique values ({unique_count})")
                    continue
                    
                # Skip if all values are null
                if X_transformed[col].isna().all():
                    logger.info(f"Skipping {col}: all null values")
                    continue
                
                valid_categorical_columns.append(col)
    
        logger.info(f"Processing {len(valid_categorical_columns)} valid categorical columns")
        
        for col in valid_categorical_columns:
            try:
                X_transformed, encoding_map = self._encode_column(
                    X_transformed, col, target, is_training=True
                )
                if encoding_map is not None:
                    self.encoding_maps_[col] = encoding_map
            except Exception as e:
                logger.warning(f"Error encoding column {col}: {e}")

        # FIX: Move this inside the method
        self.feature_names_out_ = [col for col in X_transformed.columns 
                             if col.endswith(('_fraud_rate_smooth', '_frequency'))]
        logger.info(f"Created {len(self.feature_names_out_)} frequency features")
    
        return X_transformed
    
    def _transform_test(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform test data using fitted encodings."""
        X_transformed = X.copy()
        
        for col in self.categorical_columns:
            if col in X_transformed.columns and col in self.encoding_maps_:
                try:
                    X_transformed, _ = self._encode_column(
                        X_transformed, col, None, is_training=False
                    )
                except Exception as e:
                    logger.warning(f"Error transforming column {col}: {e}")
        
        return X_transformed
    
    def _encode_column(self, X: pd.DataFrame, col: str, target: Optional[pd.Series], 
                      is_training: bool = True) -> Tuple[pd.DataFrame, Optional[Dict]]:
        """Encode a single categorical column."""
        
        # Add input validation first
        if col not in X.columns:
            logger.warning(f"Column '{col}' not found in DataFrame, skipping")
            return X, None
        
        # Check if column has valid data
        if X[col].isna().all():
            logger.warning(f"Column '{col}' contains only null values, skipping")
            return X, None
        
        try:
            if is_training:
                # Ensure target is properly aligned
                if target is None:
                    logger.warning(f"No target provided for training encoding of {col}")
                    return X, None
                
                # Reset index to ensure proper alignment
                X_working = X.reset_index(drop=True)
                target_working = target.reset_index(drop=True)
                
                # Calculate category statistics - FIX THE GROUPBY OPERATION
                stats_df = pd.DataFrame({
                    col: X_working[col],
                    'target': target_working
                }).dropna()  # Remove rows with NaN values
                
                if stats_df.empty:
                    logger.warning(f"No valid data for encoding {col}")
                    return X, None
                
                # Group by the categorical column properly
                stats = stats_df.groupby(col, as_index=False)['target'].agg(['count', 'sum'])
                stats = stats.reset_index()  # Ensure proper column structure
                
                # Rename columns explicitly
                stats.columns = [col, 'count', 'fraud_count']
                
                # Filter out low-frequency categories
                stats = stats[stats['count'] >= self.config.min_category_frequency]
                
                if stats.empty:
                    logger.warning(f"No categories meet minimum frequency for {col}")
                    return X, None
                
                # Bayesian smoothing
                stats[f'{col}_fraud_rate_smooth'] = (
                    (stats['fraud_count'] + self.config.bayesian_alpha * self.global_fraud_rate_) / 
                    (stats['count'] + self.config.bayesian_alpha)
                )
                
                # Frequency encoding
                stats[f'{col}_frequency'] = stats['count']
                
                # Create encoding maps
                encoding_map = {
                    'fraud_rate': dict(zip(stats[col], stats[f'{col}_fraud_rate_smooth'])),
                    'frequency': dict(zip(stats[col], stats[f'{col}_frequency']))
                }
                
                # Apply encoding with proper error handling
                X[f'{col}_fraud_rate_smooth'] = X[col].map(encoding_map['fraud_rate']).fillna(self.global_fraud_rate_)
                X[f'{col}_frequency'] = X[col].map(encoding_map['frequency']).fillna(1)
                
                return X, encoding_map
                
            else:
                # Use pre-calculated encodings
                if col not in self.encoding_maps_:
                    logger.warning(f"No encoding map found for {col}")
                    return X, None
                    
                encoding_map = self.encoding_maps_[col]
                X[f'{col}_fraud_rate_smooth'] = X[col].map(encoding_map['fraud_rate']).fillna(self.global_fraud_rate_)
                X[f'{col}_frequency'] = X[col].map(encoding_map['frequency']).fillna(1)
                
                return X, None
                
        except Exception as e:
            logger.warning(f"Error encoding column {col}: {str(e)}")
            logger.debug(f"Column dtype: {X[col].dtype}, unique values: {X[col].nunique()}")
            return X, None


class AnomalyDetector(BaseFeatureTransformer):
    """Advanced anomaly detection features."""
    
    def __init__(self, config: FeatureConfig):
        super().__init__(config)
        self.isolation_forest_ = None
        self.scaler_ = None
        
    def _fit_transform_train(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform training data."""
        X_transformed = X.copy()
        
        try:
            # Select numeric features for anomaly detection
            numeric_cols = X_transformed.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if not col.endswith('ID')]
            
            if len(numeric_cols) < 2:
                logger.warning("Insufficient numeric features for anomaly detection")
                return X_transformed
            
            # Prepare data for isolation forest
            X_numeric = X_transformed[numeric_cols].fillna(0)
            
            # Fit isolation forest
            self.isolation_forest_ = IsolationForest(
                contamination=self.config.isolation_contamination,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs
            )
            
            anomaly_scores = self.isolation_forest_.fit_predict(X_numeric)
            X_transformed['isolation_anomaly'] = (anomaly_scores == -1).astype(np.int8)
            
            # Statistical anomalies
            X_transformed = self._create_statistical_anomalies(X_transformed)
            
            self.feature_names_out_ = ['isolation_anomaly'] + [col for col in X_transformed.columns 
                                                             if col.endswith(('_zscore', '_outlier', '_extreme'))]
            logger.info(f"Created {len(self.feature_names_out_)} anomaly features")
            
        except Exception as e:
            logger.warning(f"Error creating anomaly features: {e}")
        
        return X_transformed
    
    def _transform_test(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform test data using fitted models."""
        X_transformed = X.copy()
        
        try:
            if self.isolation_forest_ is not None:
                # Select same numeric features as training
                numeric_cols = X_transformed.select_dtypes(include=[np.number]).columns.tolist()
                numeric_cols = [col for col in numeric_cols if not col.endswith('ID')]
                
                if len(numeric_cols) >= 2:
                    X_numeric = X_transformed[numeric_cols].fillna(0)
                    anomaly_scores = self.isolation_forest_.predict(X_numeric)
                    X_transformed['isolation_anomaly'] = (anomaly_scores == -1).astype(np.int8)
            
            # Statistical anomalies
            X_transformed = self._create_statistical_anomalies(X_transformed)
            
        except Exception as e:
            logger.warning(f"Error transforming anomaly features: {e}")
        
        return X_transformed
    
    def _create_statistical_anomalies(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create statistical anomaly features."""
        try:
            # Amount-based anomalies
            if 'TransactionAmt' in X.columns:
                X['amt_zscore'] = np.abs(stats.zscore(X['TransactionAmt'].fillna(0)))
                X['is_amt_extreme'] = (X['amt_zscore'] > self.config.zscore_threshold).astype(np.int8)
            
            # Velocity anomalies
            for hours in self.config.velocity_windows:
                col = f'velocity_{hours}h'
                if col in X.columns:
                    X[f'{col}_zscore'] = np.abs(stats.zscore(X[col].fillna(1)))
                    X[f'is_{col}_extreme'] = (X[f'{col}_zscore'] > self.config.zscore_threshold).astype(np.int8)
                    
        except Exception as e:
            logger.warning(f"Error creating statistical anomalies: {e}")
        
        return X


class InteractionFeatureCreator(BaseFeatureTransformer):
    """Create meaningful feature interactions."""
    
    def __init__(self, config: FeatureConfig):
        super().__init__(config)
        
    def _fit_transform_train(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform training data."""
        return self._create_interactions(X)
    
    def _transform_test(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform test data."""
        return self._create_interactions(X)
    
    def _create_interactions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features."""
        X_transformed = X.copy()
        
        try:
            # Amount × Time interactions
            amount_col = 'TransactionAmt'
            if amount_col in X_transformed.columns:
                for time_feature in ['TransactionDT_hour', 'is_weekend', 'is_night', 'is_business_hours']:
                    if time_feature in X_transformed.columns:
                        X_transformed[f'amt_{time_feature}_interaction'] = (
                            X_transformed[amount_col] * X_transformed[time_feature]
                        )
            
            # Velocity × Amount interactions
            if amount_col in X_transformed.columns:
                for hours in self.config.velocity_windows:
                    velocity_col = f'velocity_{hours}h'
                    if velocity_col in X_transformed.columns:
                        X_transformed[f'velocity_amt_{hours}h_interaction'] = (
                            X_transformed[velocity_col] * np.log1p(X_transformed[amount_col])
                        )
            
            # Card family interactions
            if 'card1' in X_transformed.columns:
                X_transformed['card_family'] = X_transformed['card1'] // 1000
                
                if amount_col in X_transformed.columns:
                    X_transformed['card_family_amt_interaction'] = (
                        X_transformed['card_family'] * X_transformed[amount_col]
                    )
            
            # Categorical combinations
            categorical_pairs = [
                ('ProductCD', 'card4'),
                ('P_emaildomain', 'R_emaildomain'),
                ('addr1', 'addr2')
            ]
            
            for col1, col2 in categorical_pairs:
                if col1 in X_transformed.columns and col2 in X_transformed.columns:
                    X_transformed[f'{col1}_{col2}_combo'] = (
                        X_transformed[col1].astype(str) + '_' + X_transformed[col2].astype(str)
                    )
            
            self.feature_names_out_ = [col for col in X_transformed.columns if col not in X.columns]
            logger.info(f"Created {len(self.feature_names_out_)} interaction features")
            
        except Exception as e:
            logger.warning(f"Error creating interaction features: {e}")
        
        return X_transformed


class FeatureEngineeringPipeline:
    """Complete feature engineering pipeline with column consistency handling."""
    
    def __init__(self, config: Optional[FeatureConfig] = None, 
                 enable_feature_selection: bool = True):
        self.config = config or FeatureConfig()
        self.transformers = {}
        self.feature_selector = None
        self.enable_feature_selection = enable_feature_selection
        self.is_fitted_ = False
        self.original_columns_ = None
        self.expected_features_ = set()  # Add this line
        
    def fit(self, train_df: pd.DataFrame, target_col: str = 'isFraud') -> 'FeatureEngineeringPipeline':
        """Fit the pipeline including feature selection."""
        logger.info("Starting feature engineering pipeline fit")
        
        # Use the imported function from column_utils
        if any('-' in col for col in train_df.columns):
            logger.warning("Detected hyphens in column names, applying standardization...")
            current_df = standardize_column_names(train_df.copy())
        else:
            current_df = train_df.copy()
        
        target = current_df[target_col]
        
        # STEP 2: Apply all transformers
        for transformer_name in ['temporal', 'amount', 'frequency', 'anomaly', 'interaction']:
            try:
                if transformer_name == 'temporal':
                    self.transformers[transformer_name] = TemporalFeatureTransformer(self.config)
                elif transformer_name == 'amount':
                    self.transformers[transformer_name] = AmountFeatureTransformer(self.config)
                elif transformer_name == 'frequency':
                    categorical_cols = current_df.select_dtypes(include=['object']).columns.tolist()
                    if target_col in categorical_cols:
                        categorical_cols.remove(target_col)
                    self.transformers[transformer_name] = FrequencyEncoder(self.config, categorical_cols)
                elif transformer_name == 'anomaly':
                    self.transformers[transformer_name] = AnomalyDetector(self.config)
                elif transformer_name == 'interaction':
                    self.transformers[transformer_name] = InteractionFeatureCreator(self.config)
                
                logger.info(f"Fitting {transformer_name} transformer")
                current_df = self.transformers[transformer_name].fit_transform(current_df, target)
                logger.info(f"After {transformer_name}: {current_df.shape}")
                
            except Exception as e:
                logger.error(f"Error in {transformer_name} transformer: {e}")
                # Continue with other transformers
                continue
        
        # STEP 3: Feature selection
        if self.enable_feature_selection:
            logger.info("Starting feature selection process")
            try:
                from feature_selection import AdvancedFeatureSelector
                self.feature_selector = AdvancedFeatureSelector(self.config)
                
                # Separate features and target
                feature_df = current_df.drop(columns=[target_col])
                
                # Apply feature selection
                selected_df = self.feature_selector.fit_transform(feature_df, target)
                
                # Recombine with target
                current_df = pd.concat([selected_df, target], axis=1)
                
                # Store expected features for consistency validation
                self.expected_features_ = set(selected_df.columns)
                
                # Log feature selection results
                report = self.feature_selector.get_feature_selection_report()
                logger.info(f"Feature selection complete: {report['original_features']} -> {report['final_features']} features")
                logger.info(f"Reduction ratio: {report['reduction_ratio']:.2%}")
                
            except Exception as e:
                logger.error(f"Feature selection failed: {e}")
                logger.info("Continuing without feature selection")
                self.enable_feature_selection = False
                # Store all feature columns (excluding target)
                self.expected_features_ = set(current_df.columns) - {target_col}
        else:
            # Store all feature columns (excluding target)
            self.expected_features_ = set(current_df.columns) - {target_col}
        
        self.is_fitted_ = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data with robust error handling."""
        if not self.is_fitted_:
            raise ValueError("Pipeline must be fitted before transform")
        
        # STEP 1: Standardize column names using same mapping as training
        current_df = standardize_column_names(df.copy())
        
        # STEP 2: Apply all transformers
        for transformer_name, transformer in self.transformers.items():
            try:
                logger.info(f"Applying {transformer_name} transformer")
                current_df = transformer.transform(current_df)
                logger.info(f"After {transformer_name}: {current_df.shape}")
            except Exception as e:
                logger.error(f"Error transforming with {transformer_name}: {e}")
                # Continue with other transformers
                continue
        
        # STEP 3: Apply feature selection with error handling
        if self.enable_feature_selection and self.feature_selector:
            try:
                logger.info("Applying feature selection")
                current_df = self.feature_selector.transform(current_df)
                logger.info(f"After feature selection: {current_df.shape}")
            except Exception as e:
                logger.error(f"Feature selection transform failed: {e}")
                # Get available features and proceed
                available_features = self.feature_selector.get_available_features(current_df)
                if available_features:
                    logger.info(f"Using {len(available_features)} available features")
                    current_df = current_df[available_features]
                else:
                    logger.warning("No selected features available, returning all features")
        
        return current_df
    
    def get_feature_names(self) -> List[str]:
        """Get list of final feature names after all transformations."""
        if not self.is_fitted_:
            raise ValueError("Pipeline must be fitted to get feature names")
        
        if self.enable_feature_selection and self.feature_selector:
            return self.feature_selector.get_selected_features()
        else:
            # Return all feature names from transformers
            all_features = set()
            for transformer_name, transformer in self.transformers.items():
                if hasattr(transformer, 'feature_names_out_'):
                    all_features.update(transformer.feature_names_out_)
            return list(all_features)

    def get_feature_importance_mapping(self) -> Dict[str, List[str]]:
        """Get mapping of feature categories to feature names."""
        feature_mapping = {}
        
        for transformer_name, transformer in self.transformers.items():
            if hasattr(transformer, 'feature_names_out_'):
                feature_mapping[transformer_name] = transformer.feature_names_out_
        
        return feature_mapping

    def get_selected_features(self) -> List[str]:
        """Get list of selected features (alias for get_feature_names for compatibility)."""
        return self.get_feature_names()

    def validate_consistency(self, test_df: pd.DataFrame, target_col: str = 'isFraud') -> Dict:
        """Validate consistency between training and test data."""
        if not self.is_fitted_:
            raise ValueError("Pipeline must be fitted before validation")
        
        # Get expected features (excluding target)
        if hasattr(self, 'expected_features_') and self.expected_features_:
            expected_features = self.expected_features_
        else:
            # Fallback: use selected features if available
            if self.enable_feature_selection and self.feature_selector:
                expected_features = set(self.feature_selector.get_selected_features())
            else:
                expected_features = set()
        
        # Get test features (excluding target if present)
        test_features = set(test_df.columns)
        if target_col in test_features:
            test_features.remove(target_col)
        
        # Calculate differences
        missing_features = expected_features - test_features
        extra_features = test_features - expected_features
        
        return {
            'is_consistent': len(missing_features) == 0 and len(extra_features) == 0,
            'missing_features': list(missing_features),
            'extra_features': list(extra_features),
            'missing_count': len(missing_features),
            'extra_count': len(extra_features),
            'expected_count': len(expected_features),
            'actual_count': len(test_features)
        }
    
class AdvancedFeatureSelector(BaseFeatureTransformer):
    """Advanced feature selection with multiple methods."""
    
    def __init__(self, config: FeatureConfig, max_features: int = 300):
        super().__init__(config)
        self.max_features = max_features
        self.selected_features_ = []
        self.feature_importances_ = {}
        self.selection_report_ = {}
        
    def _fit_transform_train(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform training data with feature selection."""
        if y is None:
            logger.warning("No target provided for feature selection, returning all features")
            self.selected_features_ = X.columns.tolist()
            return X
        
        logger.info(f"Starting feature selection with {X.shape[1]} features")
        
        # Step 1: Remove constant and quasi-constant features
        X_filtered = self._remove_constant_features(X)
        logger.info(f"After constant removal: {X_filtered.shape[1]} features")
        
        # Step 2: Remove highly correlated features
        X_filtered = self._remove_correlated_features(X_filtered)
        logger.info(f"After correlation removal: {X_filtered.shape[1]} features")
        
        # Step 3: Statistical feature selection
        X_filtered = self._statistical_selection(X_filtered, y)
        logger.info(f"After statistical selection: {X_filtered.shape[1]} features")
        
        # Step 4: Tree-based feature importance (if still too many features)
        if X_filtered.shape[1] > self.max_features:
            X_filtered = self._tree_based_selection(X_filtered, y)
            logger.info(f"After tree-based selection: {X_filtered.shape[1]} features")
        
        self.selected_features_ = X_filtered.columns.tolist()
        
        # Create selection report
        self.selection_report_ = {
            'original_features': X.shape[1],
            'final_features': len(self.selected_features_),
            'reduction_ratio': 1 - (len(self.selected_features_) / X.shape[1])
        }
        
        logger.info(f"Feature selection complete: {self.selection_report_}")
        
        return X_filtered
    
    def _transform_test(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform test data using selected features."""
        # Only keep selected features that exist in the test set
        available_features = [col for col in self.selected_features_ if col in X.columns]
        
        if len(available_features) < len(self.selected_features_):
            missing_features = set(self.selected_features_) - set(available_features)
            logger.warning(f"Missing {len(missing_features)} features in test set: {list(missing_features)[:5]}")
        
        return X[available_features]
    
    def _remove_constant_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove constant and quasi-constant features."""
        try:
            # Remove constant features
            constant_features = []
            for col in X.columns:
                if X[col].nunique() <= 1:
                    constant_features.append(col)
            
            # Remove quasi-constant features (>95% same value)
            quasi_constant_features = []
            for col in X.columns:
                if col not in constant_features:
                    mode_freq = X[col].value_counts().iloc[0] / len(X) if len(X) > 0 else 0
                    if mode_freq > 0.95:
                        quasi_constant_features.append(col)
            
            features_to_remove = constant_features + quasi_constant_features
            
            if features_to_remove:
                logger.info(f"Removing {len(features_to_remove)} constant/quasi-constant features")
                X = X.drop(columns=features_to_remove)
            
            return X
            
        except Exception as e:
            logger.warning(f"Error removing constant features: {e}")
            return X
    
    def _remove_correlated_features(self, X: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        """Remove highly correlated features."""
        try:
            # Only use numeric features for correlation
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) < 2:
                return X
            
            # Calculate correlation matrix
            corr_matrix = X[numeric_cols].corr().abs()
            
            # Find highly correlated pairs
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            # Find features to drop
            to_drop = [column for column in upper_triangle.columns 
                      if any(upper_triangle[column] > threshold)]
            
            if to_drop:
                logger.info(f"Removing {len(to_drop)} highly correlated features")
                X = X.drop(columns=to_drop)
            
            return X
            
        except Exception as e:
            logger.warning(f"Error removing correlated features: {e}")
            return X
    
    def _statistical_selection(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Statistical feature selection using mutual information."""
        try:
            # Separate numeric and categorical features
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            categorical_cols = X.select_dtypes(include=['object']).columns
            
            selected_features = []
            
            # Handle numeric features
            if len(numeric_cols) > 0:
                X_numeric = X[numeric_cols].fillna(0)
                
                # Use mutual information for numeric features
                k_best_numeric = min(len(numeric_cols), self.max_features // 2)
                selector_numeric = SelectKBest(score_func=mutual_info_classif, k=k_best_numeric)
                selector_numeric.fit(X_numeric, y)
                
                selected_numeric = numeric_cols[selector_numeric.get_support()].tolist()
                selected_features.extend(selected_numeric)
            
            # Handle categorical features (keep top frequency encoded ones)
            if len(categorical_cols) > 0:
                # Prioritize frequency encoded features
                freq_encoded_cols = [col for col in categorical_cols 
                                   if any(suffix in col for suffix in ['_frequency', '_fraud_rate_smooth'])]
                
                # Keep up to 50 categorical features, prioritizing encoded ones
                k_categorical = min(len(categorical_cols), 50)
                if len(freq_encoded_cols) > 0:
                    selected_categorical = freq_encoded_cols[:k_categorical]
                else:
                    selected_categorical = categorical_cols[:k_categorical].tolist()
                
                selected_features.extend(selected_categorical)
            
            # Ensure we don't exceed max_features
            if len(selected_features) > self.max_features:
                selected_features = selected_features[:self.max_features]
            
            return X[selected_features]
            
        except Exception as e:
            logger.warning(f"Error in statistical selection: {e}")
            return X
    
    def _tree_based_selection(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Tree-based feature importance selection."""
        try:
            # Prepare data for random forest
            X_prepared = X.copy()
            
            # Handle categorical variables
            for col in X_prepared.select_dtypes(include=['object']).columns:
                X_prepared[col] = pd.Categorical(X_prepared[col]).codes
            
            # Fill missing values
            X_prepared = X_prepared.fillna(0)
            
            # Fit random forest
            rf = RandomForestClassifier(
                n_estimators=100,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                max_depth=10  # Limit depth for speed
            )
            
            rf.fit(X_prepared, y)
            
            # Get feature importances
            importances = rf.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Select top features
            top_features = feature_importance_df.head(self.max_features)['feature'].tolist()
            
            # Store importance information
            self.feature_importances_ = dict(zip(feature_importance_df['feature'], 
                                               feature_importance_df['importance']))
            
            return X[top_features]
            
        except Exception as e:
            logger.warning(f"Error in tree-based selection: {e}")
            return X
    
    def get_selected_features(self) -> List[str]:
        """Get list of selected features."""
        return self.selected_features_
    
    def get_feature_selection_report(self) -> Dict:
        """Get feature selection report."""
        return self.selection_report_
    
    def get_available_features(self, X: pd.DataFrame) -> List[str]:
        """Get available selected features in the given dataframe."""
        return [col for col in self.selected_features_ if col in X.columns]


# Now update your existing AnomalyDetector class with the production version
class ProductionAnomalyDetector(AnomalyDetector):
    """Production-ready anomaly detector with consistent feature output."""
    
    def __init__(self, config: FeatureConfig):
        super().__init__(config)
        self.training_features_ = []
        self.fitted_stats_ = {}
    
    def _fit_transform_train(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform training data, storing feature metadata."""
        
        # Store training features for consistency
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if not col.endswith('ID')]
        self.training_features_ = numeric_cols
        
        # Call parent method
        result = super()._fit_transform_train(X, y)
        
        # Store feature statistics for test set defaults
        for col in result.columns:
            if col not in X.columns:  # New feature
                try:
                    self.fitted_stats_[col] = {
                        'mean': result[col].mean(),
                        'median': result[col].median(),
                        'std': result[col].std()
                    }
                except Exception:
                    pass
        
        return result
    
    def _transform_test(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform test data ensuring all training features are present."""
        X_transformed = X.copy()
        
        try:
            # Always create isolation_anomaly feature
            if self.isolation_forest_ is not None:
                numeric_cols = X_transformed.select_dtypes(include=[np.number]).columns.tolist()
                numeric_cols = [col for col in numeric_cols if not col.endswith('ID')]
                
                if len(numeric_cols) >= 2:
                    X_numeric = X_transformed[numeric_cols].fillna(0)
                    
                    # Ensure we have the same features as training
                    training_features = getattr(self, 'training_features_', numeric_cols)
                    
                    # Align features
                    for feat in training_features:
                        if feat not in X_numeric.columns:
                            X_numeric[feat] = 0.0
                    
                    # Use only training features
                    X_numeric = X_numeric[training_features]
                    
                    anomaly_scores = self.isolation_forest_.predict(X_numeric)
                    X_transformed['isolation_anomaly'] = (anomaly_scores == -1).astype(np.int8)
                else:
                    # Fallback: no anomaly detected
                    X_transformed['isolation_anomaly'] = 0
            else:
                # Model not fitted, assume no anomalies
                X_transformed['isolation_anomaly'] = 0
            
            # Statistical anomalies with guaranteed feature creation
            X_transformed = self._create_statistical_anomalies_robust(X_transformed)
            
        except Exception as e:
            logger.warning(f"Error in anomaly detection: {e}")
            # Ensure critical features exist even if calculation fails
            if 'isolation_anomaly' not in X_transformed.columns:
                X_transformed['isolation_anomaly'] = 0
        
        return X_transformed
    
    def _create_statistical_anomalies_robust(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create statistical anomalies with guaranteed feature output."""
        
        # Amount-based anomalies
        if 'TransactionAmt' in X.columns:
            try:
                X['amt_zscore'] = np.abs(stats.zscore(X['TransactionAmt'].fillna(0)))
                X['is_amt_extreme'] = (X['amt_zscore'] > self.config.zscore_threshold).astype(np.int8)
            except Exception:
                X['amt_zscore'] = 0.0
                X['is_amt_extreme'] = 0
        
        # Velocity anomalies - ALWAYS create these features
        for hours in self.config.velocity_windows:
            zscore_col = f'velocity_{hours}h_zscore'
            extreme_col = f'is_velocity_{hours}h_extreme'
            velocity_col = f'velocity_{hours}h'
            
            if velocity_col in X.columns:
                try:
                    X[zscore_col] = np.abs(stats.zscore(X[velocity_col].fillna(1)))
                    X[extreme_col] = (X[zscore_col] > self.config.zscore_threshold).astype(np.int8)
                except Exception:
                    X[zscore_col] = 0.0
                    X[extreme_col] = 0
            else:
                # Create features even if velocity column doesn't exist
                X[zscore_col] = 0.0
                X[extreme_col] = 0
        
        return X


# Add the alignment function
def align_datasets(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                  target_col: str = 'isFraud') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Align train and test datasets to have consistent columns."""
    
    # Standardize column names first
    train_aligned = standardize_column_names(train_df.copy())
    test_aligned = standardize_column_names(test_df.copy())
    
    # Get column sets (excluding target from train)
    train_cols = set(train_aligned.columns)
    if target_col in train_cols:
        train_feature_cols = train_cols - {target_col}
    else:
        train_feature_cols = train_cols
    
    test_cols = set(test_aligned.columns)
    
    # Find missing and extra columns
    missing_in_test = train_feature_cols - test_cols
    extra_in_test = test_cols - train_feature_cols
    
    # Add missing columns to test with appropriate default values
    for col in missing_in_test:
        if 'velocity' in col.lower():
            test_aligned[col] = 1  # Default velocity
        elif 'anomaly' in col.lower() or 'outlier' in col.lower():
            test_aligned[col] = 0  # Default no anomaly
        elif 'zscore' in col.lower():
            test_aligned[col] = 0.0  # Default z-score
        elif col.endswith(('_frequency', '_count')):
            test_aligned[col] = 1  # Default frequency
        elif col.endswith('_fraud_rate_smooth'):
            test_aligned[col] = 0.1  # Default fraud rate
        else:
            test_aligned[col] = 0  # General default
        
        logger.info(f"Added missing column to test: {col}")
    
    # Remove extra columns from test
    if extra_in_test:
        test_aligned = test_aligned.drop(columns=list(extra_in_test))
        logger.info(f"Removed {len(extra_in_test)} extra columns from test")
    
    # Verify alignment
    train_feature_cols_final = set(train_aligned.columns) - {target_col} if target_col in train_aligned.columns else set(train_aligned.columns)
    test_cols_final = set(test_aligned.columns)
    
    if train_feature_cols_final == test_cols_final:
        logger.info("✅ Datasets successfully aligned")
    else:
        remaining_diff = train_feature_cols_final.symmetric_difference(test_cols_final)
        logger.warning(f"⚠ Still {len(remaining_diff)} column differences: {list(remaining_diff)[:5]}")
    
    return train_aligned, test_aligned