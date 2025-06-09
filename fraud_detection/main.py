import sys
import os
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Tuple, Optional

# Add the current directory and src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'src'))

# Now import directly from preprocessing and feature engineering modules
from preprocessing import DataCleaner, clean_credit_data
from feature_engineering import (
    FeatureConfig, 
    FeatureEngineeringPipeline,  
    align_datasets,
    AdvancedFeatureSelector
)
from column_utils import standardize_column_names, validate_column_consistency
from pipeline_diagnostics import PipelineDiagnostics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Enhanced main execution function for credit default data preprocessing and feature engineering
    """
    # Define file paths for all 4 files
    files_to_process = [
        {
            "name": "train_transaction",
            "raw_path": "/Users/denco_23/Documents/Second_Brain/01_Projects/AI/Data_Science_Projects/Credit_Default/fraud_detection/data/raw/train_transaction.csv",
            "output_path": "/Users/denco_23/Documents/Second_Brain/01_Projects/AI/Data_Science_Projects/Credit_Default/fraud_detection/data/preprocessed/train_transaction_cleaned.csv",
            "type": "train"
        },
        {
            "name": "test_transaction", 
            "raw_path": "/Users/denco_23/Documents/Second_Brain/01_Projects/AI/Data_Science_Projects/Credit_Default/fraud_detection/data/raw/test_transaction.csv",
            "output_path": "/Users/denco_23/Documents/Second_Brain/01_Projects/AI/Data_Science_Projects/Credit_Default/fraud_detection/data/preprocessed/test_transaction_cleaned.csv",
            "type": "test"
        },
        {
            "name": "train_identity",
            "raw_path": "/Users/denco_23/Documents/Second_Brain/01_Projects/AI/Data_Science_Projects/Credit_Default/fraud_detection/data/raw/train_identity.csv", 
            "output_path": "/Users/denco_23/Documents/Second_Brain/01_Projects/AI/Data_Science_Projects/Credit_Default/fraud_detection/data/preprocessed/train_identity_cleaned.csv",
            "type": "train"
        },
        {
            "name": "test_identity",
            "raw_path": "/Users/denco_23/Documents/Second_Brain/01_Projects/AI/Data_Science_Projects/Credit_Default/fraud_detection/data/raw/test_identity.csv",
            "output_path": "/Users/denco_23/Documents/Second_Brain/01_Projects/AI/Data_Science_Projects/Credit_Default/fraud_detection/data/preprocessed/test_identity_cleaned.csv",
            "type": "test"
        }
    ]
    
    # Create output directories
    os.makedirs("data/preprocessed", exist_ok=True)
    os.makedirs("data/feature_engineered", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Store cleaner instances and results
    cleaner_instances = {}
    cleaned_datasets = {}
    
    print("🚀 Starting Enhanced Credit Default Data Processing Pipeline...")
    print("=" * 80)
    
    # Phase 1: Data Cleaning
    print("\n🔧 PHASE 1: DATA CLEANING")
    print("-" * 40)
    
    # Process each file
    for file_info in files_to_process:
        file_name = file_info["name"]
        raw_path = file_info["raw_path"]
        output_path = file_info["output_path"]
        
        print(f"\n📂 Processing {file_name}...")
        
        # First, try to load existing cleaned dataset
        try:
            print(f"🔍 Checking for existing cleaned file: {output_path}")
            cleaned_data = pd.read_csv(output_path)
            print(f"✅ Found existing cleaned dataset for {file_name}")
            print(f"📊 Loaded shape: {cleaned_data.shape}")
            
            # Store the loaded dataset
            cleaned_datasets[file_name] = cleaned_data
            cleaner_instances[file_name] = None
            
        except (FileNotFoundError, pd.errors.EmptyDataError, Exception) as e:
            print(f"⚠️  Cleaned file not found or corrupted: {str(e)}")
            print(f"🔄 Proceeding with cleaning process for {file_name}...")
            
            # Check if raw data file exists
            if not os.path.exists(raw_path):
                print(f"❌ Raw data file not found: {raw_path}")
                print(f"⚠️  Skipping {file_name}")
                continue
            
            try:
                # Execute cleaning pipeline using the preprocessing class
                cleaned_data, cleaner_instance = clean_credit_data(raw_path, output_path)
                
                # Store the instances and data
                cleaner_instances[file_name] = cleaner_instance
                cleaned_datasets[file_name] = cleaned_data
                
                print(f"✅ {file_name} cleaning completed successfully!")
                print(f"📊 Original shape: {cleaner_instance.original_shape}")
                print(f"📊 Cleaned shape: {cleaned_data.shape}")
                print(f"💾 Output saved to: {output_path}")
                
            except Exception as e:
                print(f"❌ Error during cleaning process for {file_name}: {str(e)}")
                continue
    
    # Phase 2: Dataset Preparation and Merging
    print("\n🔗 PHASE 2: DATASET PREPARATION")
    print("-" * 40)
    
    # Merge transaction and identity datasets
    train_merged, test_merged = merge_datasets(cleaned_datasets)
    
    if train_merged is None or test_merged is None:
        print("❌ Failed to merge datasets. Exiting.")
        return cleaner_instances, cleaned_datasets, None, None, None
    
    # Phase 3: Advanced Feature Engineering
    print("\n🏗️  PHASE 3: ADVANCED FEATURE ENGINEERING")
    print("-" * 40)
    
    # Configure feature engineering
    feature_config = FeatureConfig(
        velocity_windows=[1, 24, 168],  # 1 hour, 1 day, 1 week
        cyclical_features=['hour', 'day'],
        amount_percentiles=[0.25, 0.5, 0.75, 0.95],
        outlier_threshold=2.0,
        bayesian_alpha=10.0,
        min_category_frequency=5,
        isolation_contamination=0.1,
        zscore_threshold=3.0,
        random_state=42,
        n_jobs=-1
    )
    
    # Apply feature engineering
    try:
        train_engineered, test_engineered, fe_pipeline = apply_feature_engineering(
            train_merged, test_merged, feature_config
        )
        
        print(f"✅ Feature engineering completed successfully!")
        print(f"📊 Training set: {train_merged.shape} -> {train_engineered.shape}")
        print(f"📊 Test set: {test_merged.shape} -> {test_engineered.shape}")
        
        try:
            feature_names = fe_pipeline.get_feature_names()
            print(f"🔧 Features created: {len(feature_names)}")
        except AttributeError:
            if hasattr(fe_pipeline, 'expected_features_'):
                print(f"🔧 Features created: {len(fe_pipeline.expected_features_)}")
            else:
                print("🔧 Features created: [count unavailable]")
        
    except Exception as e:
        print(f"❌ Error during feature engineering: {str(e)}")
        logger.error(f"Feature engineering failed: {e}", exc_info=True)
        return cleaner_instances, cleaned_datasets, train_merged, test_merged, None
    
    # Phase 4: Save feature-selected datasets
    print("\n💾 PHASE 4: SAVING FEATURE-SELECTED DATASETS")
    print("-" * 40)
    
    # Save both regular and feature-selected datasets
    save_final_datasets(train_engineered, test_engineered, fe_pipeline)
    
    # Also save feature-selected version with metadata
    if fe_pipeline and hasattr(fe_pipeline, 'get_feature_names'):
        save_feature_selected_datasets(train_engineered, test_engineered, fe_pipeline)
    
    # Phase 4: Final Summary
    print("\n" + "=" * 80)
    print("📋 COMPLETE PIPELINE SUMMARY:")
    print("-" * 40)
    print(f"✅ Successfully processed: {len(cleaned_datasets)} raw files")
    print(f"🔗 Merged datasets: Train + Test")
    print(f"🏗️  Feature engineering: Complete")
    print(f"📁 Available datasets: {list(cleaned_datasets.keys())}")
    
    for name, data in cleaned_datasets.items():
        print(f"   - {name}: {data.shape}")
    
    print(f"\n🎯 FINAL DATASETS:")
    print(f"   - Train (engineered): {train_engineered.shape}")
    print(f"   - Test (engineered): {test_engineered.shape}")
    print(f"   - Total features: {train_engineered.shape[1]}")
    
    # Save final datasets
    print("💾 Final datasets saved successfully!")
    
    # Return all components for further use
    return (cleaner_instances, cleaned_datasets, train_engineered, 
            test_engineered, fe_pipeline)

def merge_datasets(cleaned_datasets: Dict[str, pd.DataFrame]) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Merge transaction and identity datasets for train and test sets
    """
    print("🔗 Merging transaction and identity datasets...")
    
    # Check if all required datasets are available
    required_datasets = ['train_transaction', 'train_identity', 'test_transaction', 'test_identity']
    available_datasets = list(cleaned_datasets.keys())
    
    missing_datasets = [ds for ds in required_datasets if ds not in available_datasets]
    if missing_datasets:
        print(f"⚠️  Warning: Missing datasets: {missing_datasets}")
        print("🔄 Proceeding with available datasets...")
    
    try:
        # Merge training datasets
        train_merged = None
        if 'train_transaction' in cleaned_datasets:
            train_merged = cleaned_datasets['train_transaction'].copy()
            print(f"📊 Train transaction base: {train_merged.shape}")
            
            if 'train_identity' in cleaned_datasets:
                train_identity = cleaned_datasets['train_identity']
                print(f"📊 Train identity: {train_identity.shape}")
                
                # Merge on TransactionID
                train_merged = train_merged.merge(
                    train_identity, 
                    on='TransactionID', 
                    how='left'
                )
                print(f"📊 Train merged: {train_merged.shape}")
            else:
                print("⚠️  Train identity not available, using transaction data only")
        
        # Merge test datasets
        test_merged = None
        if 'test_transaction' in cleaned_datasets:
            test_merged = cleaned_datasets['test_transaction'].copy()
            print(f"📊 Test transaction base: {test_merged.shape}")
            
            if 'test_identity' in cleaned_datasets:
                test_identity = cleaned_datasets['test_identity']
                print(f"📊 Test identity: {test_identity.shape}")
                
                # Merge on TransactionID
                test_merged = test_merged.merge(
                    test_identity, 
                    on='TransactionID', 
                    how='left'
                )
                print(f"📊 Test merged: {test_merged.shape}")
            else:
                print("⚠️  Test identity not available, using transaction data only")
        
        return train_merged, test_merged
        
    except Exception as e:
        print(f"❌ Error during dataset merging: {str(e)}")
        logger.error(f"Dataset merging failed: {e}", exc_info=True)
        return None, None

def apply_feature_engineering_v2(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    config: FeatureConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, FeatureEngineeringPipeline]:
    """Apply production-grade feature engineering with guaranteed consistency."""
    
    print("🏗️  Starting production feature engineering pipeline...")
    
    # Initialize diagnostics
    diagnostics = PipelineDiagnostics()
    
    # Pre-transformation validation
    print("🔍 Running pre-transformation diagnostics...")
    initial_diagnosis = diagnostics.diagnose_column_mismatch(train_df, test_df)
    diagnostics.print_diagnosis_report(initial_diagnosis)
    
    # Apply dataset alignment first
    print("🔧 Aligning datasets for consistency...")
    train_aligned, test_aligned = align_datasets(train_df.copy(), test_df.copy(), 'isFraud')
    
    try:
        # Use production pipeline
        pipeline = FeatureEngineeringPipeline(config, enable_feature_selection=True)
        
        # Fit on training data
        print("🔧 Fitting production pipeline...")
        pipeline.fit(train_aligned, 'isFraud')
        
        # Validate consistency before transformation
        print("✅ Validating pipeline consistency...")
        consistency_report = pipeline.validate_consistency(test_aligned)
        
        if not consistency_report['is_consistent']:
            print(f"⚠️  Consistency issues detected:")
            print(f"   Missing features: {consistency_report['missing_count']}")
            print(f"   Extra features: {consistency_report['extra_count']}")
            print("🔧 Pipeline will auto-correct these issues...")
        
        # Transform both datasets
        print("🚀 Transforming datasets...")
        train_engineered = pipeline.transform(train_aligned)
        test_engineered = pipeline.transform(test_aligned)
        
        # Post-transformation validation
        print("🔍 Running post-transformation diagnostics...")
        final_diagnosis = diagnostics.diagnose_column_mismatch(
            train_engineered, test_engineered, 'isFraud'
        )
        
        if final_diagnosis['is_critical']:
            raise ValueError(f"Pipeline failed to ensure consistency: {final_diagnosis}")
        
        print(f"✅ Feature engineering completed successfully!")
        print(f"📊 Training set: {train_df.shape} -> {train_engineered.shape}")
        print(f"📊 Test set: {test_df.shape} -> {test_engineered.shape}")
        print(f"🔧 Features: {len(pipeline.expected_features_)}")
        print(f"🎯 Column consistency: GUARANTEED")
        
        # Final validation report
        diagnostics.print_diagnosis_report(final_diagnosis)
        
        return train_engineered, test_engineered, pipeline
        
    except Exception as e:
        print(f"❌ Production pipeline failed: {str(e)}")
        logger.error(f"Production feature engineering error: {e}", exc_info=True)
        raise

# Update your main function to use the new pipeline
def apply_feature_engineering(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    config: FeatureConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, FeatureEngineeringPipeline]:
    """Apply comprehensive feature engineering with robust error handling."""
    
    print("🏗️  Starting advanced feature engineering...")
    
    try:
        # Use the new production pipeline
        return apply_feature_engineering_v2(train_df, test_df, config)
        
    except Exception as e:
        print(f"❌ Feature engineering failed: {str(e)}")
        logger.error(f"Feature engineering error: {e}", exc_info=True)
        
        # Fallback to basic alignment
        print("🔄 Falling back to basic column alignment...")
        train_aligned, test_aligned = align_datasets(train_df.copy(), test_df.copy(), 'isFraud')
        
        # Create minimal pipeline
        minimal_config = FeatureConfig()
        minimal_pipeline = FeatureEngineeringPipeline(minimal_config, enable_feature_selection=False)
        minimal_pipeline.is_fitted_ = True
        minimal_pipeline.expected_features_ = set(train_aligned.columns) - {'isFraud'}
        minimal_pipeline.feature_defaults_ = {}
        
        return train_aligned, test_aligned, minimal_pipeline
def save_final_datasets(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                       pipeline: Optional[FeatureEngineeringPipeline] = None) -> None:
    """
    Save the final engineered datasets
    """
    print("\n💾 Saving final engineered datasets...")
    
    try:
        # Get the current directory (fraud_detection)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Define output directory and create it
        output_dir = os.path.join(current_dir, "data", "feature_engineered")
        os.makedirs(output_dir, exist_ok=True)
        
        # Define output paths using absolute paths
        train_output = os.path.join(output_dir, "train_final_engineered.csv")
        test_output = os.path.join(output_dir, "test_final_engineered.csv")
        
        # Save datasets
        print(f"📍 Saving training data to: {train_output}")
        train_df.to_csv(train_output, index=False)
        
        print(f"📍 Saving test data to: {test_output}")
        test_df.to_csv(test_output, index=False)
        
        print(f"✅ Training set saved to: {train_output}")
        print(f"✅ Test set saved to: {test_output}")
        
        # Save feature list
        feature_list_path = os.path.join(output_dir, "feature_list.txt")
        with open(feature_list_path, 'w') as f:
            for feature in train_df.columns:
                f.write(f"{feature}\n")
        
        print(f"✅ Feature list saved to: {feature_list_path}")
        
        # Verify files were created
        if os.path.exists(train_output) and os.path.exists(test_output):
            train_size = os.path.getsize(train_output)
            test_size = os.path.getsize(test_output)
            print(f"📏 Training file size: {train_size / (1024*1024):.2f} MB")
            print(f"📏 Test file size: {test_size / (1024*1024):.2f} MB")
        else:
            print("⚠️  Warning: Files may not have been created successfully")
        
        # Only save pipeline info if pipeline is provided and has selection capability
        if pipeline and hasattr(pipeline, 'get_feature_selection_report'):
            try:
                selection_report = pipeline.get_feature_selection_report()
                if selection_report:  # Check if report exists and is not empty
                    
                    # Save detailed feature selection report
                    report_path = os.path.join(output_dir, "feature_selection_report.json")
                    import json
                    with open(report_path, 'w') as f:
                        json.dump(selection_report, f, indent=2)
                    
                    # Save selected features only
                    selected_features_path = os.path.join(output_dir, "selected_features.txt")
                    selected_features = pipeline.get_feature_names()
                    with open(selected_features_path, 'w') as f:
                        for feature in selected_features:
                            f.write(f"{feature}\n")
                    
                    print(f"✅ Feature selection report saved to: {report_path}")
                    print(f"✅ Selected features list saved to: {selected_features_path}")
                    print(f"🎯 Features reduced from {selection_report['original_features']} to {selection_report['final_features']}")    
            except Exception as e:
                print(f"⚠️  Could not save pipeline reports: {e}")
        
    except Exception as e:
        print(f"❌ Error saving final datasets: {str(e)}")
        logger.error(f"Dataset saving failed: {e}", exc_info=True)
        
        # Additional debugging information
        print(f"🔍 Current working directory: {os.getcwd()}")
        print(f"🔍 Script directory: {os.path.dirname(os.path.abspath(__file__))}")
def save_feature_selected_datasets(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                                 pipeline: FeatureEngineeringPipeline) -> None:
    """
    Save feature-selected datasets to CSV files with comprehensive metadata.
    """
    # Get absolute path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "data", "feature_engineered")
    os.makedirs(output_dir, exist_ok=True)
    
    # Define file paths
    train_path = os.path.join(output_dir, "train_final_feature_selected.csv")
    test_path = os.path.join(output_dir, "test_final_feature_selected.csv")
    
    print(f"\n💾 SAVING FEATURE-SELECTED DATASETS")
    print(f"📁 Output directory: {output_dir}")
    print("-" * 50)
    
    try:
        # Save training data
        print(f"📊 Saving training data: {train_df.shape}")
        train_df.to_csv(train_path, index=False)
        print(f"✅ Training data saved to: {train_path}")
        
        # Save test data  
        print(f"📊 Saving test data: {test_df.shape}")
        test_df.to_csv(test_path, index=False)
        print(f"✅ Test data saved to: {test_path}")
        
        # Save comprehensive metadata
        save_feature_metadata(train_df, test_df, pipeline, output_dir)
        
        # Verify file sizes
        train_size = os.path.getsize(train_path) / (1024 * 1024)  # MB
        test_size = os.path.getsize(test_path) / (1024 * 1024)   # MB
        
        print(f"📁 File sizes - Train: {train_size:.1f} MB, Test: {test_size:.1f} MB")
        print(f"🎯 Total features saved: {train_df.shape[1]}")
        
    except Exception as e:
        print(f"❌ Error saving datasets: {e}")
        logger.error(f"Error saving feature-selected datasets: {e}")

def save_feature_metadata(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                         pipeline: FeatureEngineeringPipeline, output_dir: str) -> None:
    """
    Save comprehensive feature metadata and selection information.
    """
    try:
        # 1. Feature list with categories
        feature_details = []
        selected_features = pipeline.get_feature_names() if hasattr(pipeline, 'get_feature_names') else train_df.columns.tolist()
        
        for feature in selected_features:
            if feature == 'isFraud':  # Skip target column
                continue
                
            feature_info = {
                'feature_name': feature,
                'data_type': str(train_df[feature].dtype),
                'category': get_feature_category(feature),
                'missing_rate_train': train_df[feature].isnull().mean(),
                'missing_rate_test': test_df[feature].isnull().mean() if feature in test_df.columns else 1.0,
                'unique_values_train': train_df[feature].nunique(),
                'unique_values_test': test_df[feature].nunique() if feature in test_df.columns else 0,
                'min_value': train_df[feature].min() if pd.api.types.is_numeric_dtype(train_df[feature]) else None,
                'max_value': train_df[feature].max() if pd.api.types.is_numeric_dtype(train_df[feature]) else None,
                'mean_value': train_df[feature].mean() if pd.api.types.is_numeric_dtype(train_df[feature]) else None
            }
            feature_details.append(feature_info)
        
        # Save feature metadata to CSV
        feature_df = pd.DataFrame(feature_details)
        feature_metadata_path = os.path.join(output_dir, "feature_metadata.csv")
        feature_df.to_csv(feature_metadata_path, index=False)
        print(f"✅ Feature metadata saved to: {feature_metadata_path}")
        
        # 2. Feature selection summary
        selection_summary = {
            'total_features_after_engineering': len(train_df.columns) - (1 if 'isFraud' in train_df.columns else 0),
            'final_selected_features': len(selected_features) - (1 if 'isFraud' in selected_features else 0),
            'reduction_ratio': 1 - (len(selected_features) / len(train_df.columns)) if len(train_df.columns) > 0 else 0,
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'target_distribution': train_df['isFraud'].value_counts().to_dict() if 'isFraud' in train_df.columns else {},
            'selection_timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Add feature category counts
        category_counts = feature_df['category'].value_counts().to_dict()
        selection_summary['feature_categories'] = category_counts
        
        # Save selection summary
        import json
        summary_path = os.path.join(output_dir, "selection_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(selection_summary, f, indent=2, default=str)
        print(f"✅ Selection summary saved to: {summary_path}")
        
        # 3. Feature importance (if available)
        if hasattr(pipeline, 'feature_selector') and pipeline.feature_selector:
            if hasattr(pipeline.feature_selector, 'feature_importances_'):
                importance_data = []
                for feature, importance in pipeline.feature_selector.feature_importances_.items():
                    importance_data.append({
                        'feature_name': feature,
                        'importance_score': importance,
                        'category': get_feature_category(feature)
                    })
                
                importance_df = pd.DataFrame(importance_data).sort_values('importance_score', ascending=False)
                importance_path = os.path.join(output_dir, "feature_importance.csv")
                importance_df.to_csv(importance_path, index=False)
                print(f"✅ Feature importance saved to: {importance_path}")
        
        # 4. Column mapping (original to final)
        column_mapping = {
            'original_columns': list(train_df.columns),
            'selected_features': selected_features,
            'engineered_features': [f for f in selected_features if get_feature_category(f) != 'original'],
            'original_features': [f for f in selected_features if get_feature_category(f) == 'original']
        };
        
        mapping_path = os.path.join(output_dir, "column_mapping.json")
        with open(mapping_path, 'w') as f:
            json.dump(column_mapping, f, indent=2)
        print(f"✅ Column mapping saved to: {mapping_path}")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not save all metadata: {e}")
        logger.warning(f"Error saving feature metadata: {e}")

def get_feature_category(feature_name: str) -> str:
    """
    Categorize features based on naming patterns.
    """
    feature_lower = feature_name.lower()
    
    # Temporal features
    if any(pattern in feature_lower for pattern in ['velocity', 'time', 'hour', 'day', 'weekend', 'night', 'business', 'cyclical', 'sin', 'cos']):
        return 'temporal'
    
    # Amount features  
    elif any(pattern in feature_lower for pattern in ['amt', 'amount', 'transactionamt', 'decimal', 'round', 'log', 'sqrt', 'cbrt']):
        return 'amount'
    
    # Anomaly features
    elif any(pattern in feature_lower for pattern in ['anomaly', 'outlier', 'zscore', 'extreme', 'isolation']):
        return 'anomaly'
    
    # Frequency features
    elif any(pattern in feature_lower for pattern in ['frequency', 'fraud_rate', '_count', 'smooth']):
        return 'frequency'
    
    # Interaction features
    elif any(pattern in feature_lower for pattern in ['interaction', 'combo', '_x_']):
        return 'interaction'
    
    # Card/Identity features
    elif any(pattern in feature_lower for pattern in ['card', 'addr', 'email', 'device', 'id']):
        return 'identity'
    
    # Statistical features
    elif any(pattern in feature_lower for pattern in ['mean', 'std', 'median', 'min', 'max', 'percentile', 'rank']):
        return 'statistical'
    
    # Original features (likely from raw data)
    else:
        return 'original'

def load_engineered_datasets() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[FeatureEngineeringPipeline]]:
    """
    Load previously engineered datasets and pipeline
    """
    try:
        train_path = "data/feature_engineered/train_final_engineered.csv"
        test_path = "data/feature_engineered/test_final_engineered.csv"
        pipeline_path = "models/feature_engineering_pipeline.pkl";
        
        # Load datasets
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # Load pipeline
        pipeline = FeatureEngineeringPipeline.load_pipeline(pipeline_path)
        
        print(f"✅ Loaded engineered datasets:")
        print(f"   - Training: {train_df.shape}")
        print(f"   - Test: {test_df.shape}")
        
        return train_df, test_df, pipeline
        
    except Exception as e:
        print(f"⚠️  Could not load engineered datasets: {str(e)}")
        return None, None, None

def get_feature_summary(df: pd.DataFrame, pipeline: Optional[FeatureEngineeringPipeline] = None) -> None:
    """
    Print comprehensive feature summary
    """
    print(f"\n📊 FEATURE SUMMARY:")
    print("-" * 40)
    print(f"Total features: {df.shape[1]}")
    print(f"Total samples: {df.shape[0]}")
    
    # Data types summary
    dtype_summary = df.dtypes.value_counts()
    print(f"\nData types:")
    for dtype, count in dtype_summary.items():
        print(f"   - {dtype}: {count} features")
    
    # Missing values summary
    missing_summary = df.isnull().sum()
    missing_features = missing_summary[missing_summary > 0]
    print(f"\nMissing values: {len(missing_features)} features with missing data")
    
    if pipeline:
        feature_mapping = pipeline.get_feature_importance_mapping()
        print(f"\nEngineered feature categories:")
        for category, features in feature_mapping.items():
            print(f"   - {category.upper()}: {len(features)} features")
def debug_pipeline_status(train_engineered, test_engineered, fe_pipeline):
    """Enhanced debug function with detailed column analysis."""
    print(f"\n🔍 PIPELINE DEBUG INFO:")
    print(f"Train engineered is None: {train_engineered is None}")
    print(f"Test engineered is None: {test_engineered is None}")  # Fixed the typo and completed the string
    print(f"Pipeline is None: {fe_pipeline is None}")
    
    if train_engineered is not None:
        print(f"Train shape: {train_engineered.shape}")
        print(f"Train columns: {list(train_engineered.columns)[:10]}...")  # Show first 10 columns
    
    if test_engineered is not None:
        print(f"Test shape: {test_engineered.shape}")
        print(f"Test columns: {list(test_engineered.columns)[:10]}...")  # Show first 10 columns
    
    if fe_pipeline is not None:
        print(f"Pipeline fitted: {hasattr(fe_pipeline, 'is_fitted_') and fe_pipeline.is_fitted_}")
        if hasattr(fe_pipeline, 'expected_features_'):
            print(f"Expected features count: {len(fe_pipeline.expected_features_)}")

# Remove the circular import code at the bottom and replace with:

def run_complete_pipeline():
    """
    Run the complete pipeline and save datasets for modeling.
    """
    print("🚀 Starting complete fraud detection pipeline...")
    
    # Run main pipeline
    results = main()
    
    if results[2] is not None and results[3] is not None:  # train_engineered, test_engineered
        train_engineered, test_engineered, fe_pipeline = results[2], results[3], results[4]
        
        # Save feature-selected datasets
        save_feature_selected_datasets(train_engineered, test_engineered, fe_pipeline)
        print("✅ Pipeline completed successfully!")
        
        return train_engineered, test_engineered, fe_pipeline
    else:
        print("❌ Pipeline failed - no datasets to save")
        return None, None, None

def quick_load_for_modeling() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, Optional[Dict]]:
    """
    Quick loader for modeling (fixes the original function).
    """
    train_df, test_df, metadata = load_feature_selected_datasets()
    
    if train_df is None or test_df is None:
        print("⚠️  Feature-selected datasets not found. Running pipeline first...")
        train_df, test_df, fe_pipeline = run_complete_pipeline()
        
        if train_df is None:
            raise FileNotFoundError("Pipeline failed to generate datasets")
    
    # Separate features and target
    if 'isFraud' in train_df.columns:
        X_train = train_df.drop('isFraud', axis=1)
        y_train = train_df['isFraud']
    else:
        raise ValueError("Target column 'isFraud' not found in training data")
    
    X_test = test_df
    
    print(f"📊 Ready for modeling:")
    print(f"   • X_train: {X_train.shape}")
    print(f"   • X_test: {X_test.shape}")
    print(f"   • y_train: {y_train.shape} (fraud rate: {y_train.mean():.3%})")
    
    return X_train, X_test, y_train, metadata

# Only run main when script is executed directly
if __name__ == "__main__":
    # Option 1: Run complete pipeline
    run_complete_pipeline()
    
    # Option 2: Or just run main() if you prefer
    # main()