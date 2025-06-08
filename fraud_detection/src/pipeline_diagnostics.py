import pandas as pd
import numpy as np
from typing import Dict, Set, List, Tuple
import logging

logger = logging.getLogger(__name__)

class PipelineDiagnostics:
    """Elite-level pipeline diagnostics for production-ready ML systems."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def diagnose_column_mismatch(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                                target_col: str = 'isFraud') -> Dict:
        """Comprehensive column mismatch analysis."""
        
        # Get feature columns (exclude target)
        train_features = set(train_df.columns)
        if target_col in train_features:
            train_features.remove(target_col)
        test_features = set(test_df.columns)
        
        # Analyze mismatches
        missing_in_test = train_features - test_features
        extra_in_test = test_features - train_features
        
        # Categorize missing features by type
        missing_analysis = self._categorize_missing_features(missing_in_test)
        
        # Calculate impact severity
        impact_score = self._calculate_impact_severity(missing_in_test, train_features)
        
        return {
            'missing_in_test': list(missing_in_test),
            'extra_in_test': list(extra_in_test),
            'missing_count': len(missing_in_test),
            'total_features': len(train_features),
            'missing_percentage': len(missing_in_test) / len(train_features) * 100 if train_features else 0,
            'impact_score': impact_score,
            'missing_analysis': missing_analysis,
            'is_critical': impact_score > 0.7
        }
    
    def _categorize_missing_features(self, missing_features: Set[str]) -> Dict[str, List[str]]:
        """Categorize missing features by engineering type."""
        categories = {
            'velocity': [],
            'anomaly': [],
            'statistical': [],
            'temporal': [],
            'interaction': [],
            'other': []
        }
        
        for feature in missing_features:
            feature_lower = feature.lower()
            if 'velocity' in feature_lower:
                categories['velocity'].append(feature)
            elif any(x in feature_lower for x in ['anomaly', 'outlier', 'isolation']):
                categories['anomaly'].append(feature)
            elif any(x in feature_lower for x in ['zscore', 'std', 'mean']):
                categories['statistical'].append(feature)
            elif any(x in feature_lower for x in ['time', 'hour', 'day', 'temporal']):
                categories['temporal'].append(feature)
            elif 'interaction' in feature_lower or '_x_' in feature_lower:
                categories['interaction'].append(feature)
            else:
                categories['other'].append(feature)
        
        return {k: v for k, v in categories.items() if v}
    
    def _calculate_impact_severity(self, missing_features: Set[str], 
                                 total_features: Set[str]) -> float:
        """Calculate business impact severity score (0-1)."""
        if not missing_features:
            return 0.0
        
        # Base severity from percentage
        percentage_impact = len(missing_features) / len(total_features)
        
        # Weight by feature importance (heuristic)
        importance_weights = {
            'velocity': 0.9,  # Critical for fraud detection
            'anomaly': 0.8,   # High importance
            'statistical': 0.6,
            'temporal': 0.7,
            'interaction': 0.5
        }
        
        weighted_impact = 0
        for feature in missing_features:
            feature_lower = feature.lower()
            if 'velocity' in feature_lower:
                weighted_impact += importance_weights['velocity']
            elif any(x in feature_lower for x in ['anomaly', 'outlier']):
                weighted_impact += importance_weights['anomaly']
            elif any(x in feature_lower for x in ['zscore', 'std']):
                weighted_impact += importance_weights['statistical']
            else:
                weighted_impact += 0.3  # Default weight
        
        # Normalize and combine
        normalized_weighted = min(weighted_impact / len(missing_features), 1.0)
        
        return min(percentage_impact * 0.3 + normalized_weighted * 0.7, 1.0)
    
    def print_diagnosis_report(self, diagnosis: Dict):
        """Print a comprehensive diagnosis report."""
        print("\n" + "="*60)
        print("🔍 PIPELINE DIAGNOSIS REPORT")
        print("="*60)
        
        if diagnosis['is_critical']:
            print("🚨 CRITICAL ISSUE DETECTED")
        else:
            print("✅ Issue severity: LOW-MEDIUM")
        
        print(f"📊 Missing features: {diagnosis['missing_count']}/{diagnosis['total_features']} ({diagnosis['missing_percentage']:.1f}%)")
        print(f"🎯 Impact score: {diagnosis['impact_score']:.2f}/1.0")
        
        if diagnosis['missing_analysis']:
            print(f"\n📋 Missing features by category:")
            for category, features in diagnosis['missing_analysis'].items():
                print(f"   • {category.title()}: {len(features)} features")
                if len(features) <= 5:
                    for feature in features:
                        print(f"     - {feature}")
                else:
                    for feature in features[:3]:
                        print(f"     - {feature}")
                    print(f"     ... and {len(features)-3} more")
        
        print("="*60)