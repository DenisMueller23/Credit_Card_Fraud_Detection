import pandas as pd
import numpy as np
import warnings
import logging
from typing import List, Dict, Optional, Tuple
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif, RFE, RFECV,
    SelectFromModel, VarianceThreshold
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AdvancedFeatureSelector:
    """
    Multi-stage feature selection pipeline for fraud detection.
    """
    
    def __init__(self, config):
        self.config = config
        self.selected_features_ = {}
        self.selection_scores_ = {}
        self.correlation_matrix_ = None
        self.is_fitted_ = False
        self.final_features_ = []
        
    def fit_transform(self, X, y, feature_names=None):
        """Complete feature selection pipeline."""
        logger.info(f"Starting feature selection with {X.shape[1]} features")
        
        # Stage 1: Remove obvious problematic features
        X_stage1 = self._stage1_basic_filtering(X, y)
        logger.info(f"Removing {X.shape[1] - X_stage1.shape[1]} constant/quasi-constant features")
        logger.info(f"After constant removal: {X_stage1.shape[1]} features")
        
        # Stage 2: Correlation-based filtering
        X_stage2 = self._stage3_correlation_filtering(X_stage1, y)
        logger.info(f"Removing {X_stage1.shape[1] - X_stage2.shape[1]} highly correlated features")
        logger.info(f"After correlation removal: {X_stage2.shape[1]} features")
        
        # Stage 3: Statistical significance filtering
        X_final = self._stage2_statistical_filtering(X_stage2, y)
        logger.info(f"After statistical selection: {X_final.shape[1]} features")
        
        # Store final features
        self.final_features_ = X_final.columns.tolist()
        self.is_fitted_ = True
        
        # Create selection report
        report = {
            'original_features': X.shape[1],
            'final_features': len(self.final_features_),
            'reduction_ratio': 1 - (len(self.final_features_) / X.shape[1])
        }
        
        logger.info(f"Feature selection complete: {report}")
        
        return X_final
    
    def _stage1_basic_filtering(self, X, y):
        """Remove features with obvious issues."""
        X_filtered = X.copy()
        features_to_remove = []
        
        # Remove features with too many missing values
        missing_threshold = 0.95
        missing_ratios = X_filtered.isnull().mean()
        high_missing = missing_ratios[missing_ratios > missing_threshold].index
        features_to_remove.extend(high_missing)
        
        # For numeric columns, apply variance threshold
        numeric_cols = X_filtered.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            variance_threshold = VarianceThreshold(threshold=0.01)
            try:
                variance_threshold.fit(X_filtered[numeric_cols].fillna(0))
                low_variance = numeric_cols[~variance_threshold.get_support()]
                features_to_remove.extend(low_variance)
            except:
                pass
        
        # Remove duplicate features
        duplicated_features = self._find_duplicate_features(X_filtered)
        features_to_remove.extend(duplicated_features)
        
        # Apply filtering
        features_to_remove = list(set(features_to_remove))
        if features_to_remove:
            X_filtered = X_filtered.drop(columns=features_to_remove)
        
        return X_filtered
    
    def _stage2_statistical_filtering(self, X, y):
        """Filter based on statistical significance."""
        try:
            # Handle mixed data types
            X_processed = X.copy()
            
            # Simple label encoding for categorical variables
            categorical_cols = X.select_dtypes(exclude=[np.number]).columns
            for col in categorical_cols:
                if X_processed[col].nunique() < 50:  # Only encode if reasonable number of categories
                    le = LabelEncoder()
                    X_processed[col] = le.fit_transform(X_processed[col].astype(str))
                else:
                    X_processed = X_processed.drop(columns=[col])
            
            # Fill missing values
            X_filled = X_processed.fillna(X_processed.median())
            
            # Mutual information selection
            k_best = min(300, X_filled.shape[1])  # Reasonable limit
            mi_selector = SelectKBest(score_func=mutual_info_classif, k=k_best)
            mi_selector.fit(X_filled, y)
            
            selected_features = X.columns[mi_selector.get_support()].tolist()
            return X[selected_features]
            
        except Exception as e:
            logger.warning(f"Statistical filtering failed: {e}")
            return X
    
    def _stage3_correlation_filtering(self, X, y):
        """Remove highly correlated features."""
        try:
            # Only use numeric columns for correlation
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) < 2:
                return X
            
            # Calculate correlation matrix
            X_numeric = X[numeric_cols].fillna(0)
            corr_matrix = X_numeric.corr().abs()
            
            # Find highly correlated pairs
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            # Features to remove
            to_drop = [column for column in upper_triangle.columns 
                      if any(upper_triangle[column] > 0.95)]
            
            return X.drop(columns=to_drop)
            
        except Exception as e:
            logger.warning(f"Correlation filtering failed: {e}")
            return X
    
    def _find_duplicate_features(self, X):
        """Find duplicate columns."""
        try:
            duplicates = []
            for i, col1 in enumerate(X.columns):
                for col2 in X.columns[i+1:]:
                    if X[col1].equals(X[col2]):
                        duplicates.append(col2)
            return duplicates
        except:
            return []
    
    def transform(self, X):
        """Transform using selected features."""
        if not self.is_fitted_:
            raise ValueError("Selector must be fitted before transform")
        
        available_features = [f for f in self.final_features_ if f in X.columns]
        return X[available_features]
    
    def get_available_features(self, X: pd.DataFrame) -> List[str]:
        """Get available features in DataFrame."""
        if not self.is_fitted_:
            return list(X.columns)
        
        return [f for f in self.final_features_ if f in X.columns]
    
    def get_feature_selection_report(self):
        """Get selection report."""
        if hasattr(self, 'selection_report_'):
            return self.selection_report_
        
        return {
            'original_features': 0,
            'final_features': len(self.final_features_) if self.final_features_ else 0,
            'reduction_ratio': 0.0
        }
    
    def get_selected_features(self):
        """Get selected features."""
        return self.final_features_