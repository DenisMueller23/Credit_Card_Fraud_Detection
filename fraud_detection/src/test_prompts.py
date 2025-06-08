# filepath: /Users/denco_23/Documents/Second_Brain/01_Projects/AI/Data_Science_Projects/Credit_Default/fraud_detection/test_prompts.py
import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'src'))

from feature_engineering import FeatureEngineeringPipeline, FeatureConfig
from feature_selection import AdvancedFeatureSelector


class TestColumnConsistency(unittest.TestCase):
    """Test column naming consistency across train/test datasets"""
    
    def setUp(self):
        """Set up test data with consistent column names"""
        # Create sample data with consistent naming
        self.train_data = pd.DataFrame({
            'TransactionDT': np.random.randint(1000, 10000, 100),
            'TransactionAmt': np.random.uniform(1, 1000, 100),
            'id_01': np.random.uniform(0, 1, 100),
            'id_02': np.random.uniform(0, 1, 100),
            'id_03': np.random.uniform(0, 1, 100),
            'id_04': np.random.uniform(0, 1, 100),
            'id_05': np.random.uniform(0, 1, 100),
            'C1': np.random.randint(0, 10, 100),
            'C2': np.random.randint(0, 10, 100),
            'isFraud': np.random.randint(0, 2, 100)
        })
        
        self.test_data = pd.DataFrame({
            'TransactionDT': np.random.randint(1000, 10000, 80),
            'TransactionAmt': np.random.uniform(1, 1000, 80),
            'id_01': np.random.uniform(0, 1, 80),
            'id_02': np.random.uniform(0, 1, 80),
            'id_03': np.random.uniform(0, 1, 80),
            'id_04': np.random.uniform(0, 1, 80),
            'id_05': np.random.uniform(0, 1, 80),
            'C1': np.random.randint(0, 10, 80),
            'C2': np.random.randint(0, 10, 80),
        })
    
    def test_column_name_consistency(self):
        """Test that train and test datasets have consistent column naming"""
        train_cols = set(self.train_data.columns) - {'isFraud'}
        test_cols = set(self.test_data.columns)
        
        # Check for missing columns in test data
        missing_in_test = train_cols - test_cols
        self.assertEqual(len(missing_in_test), 0, 
                        f"Missing columns in test data: {missing_in_test}")
        
        # Check for extra columns in test data
        extra_in_test = test_cols - train_cols
        self.assertEqual(len(extra_in_test), 0,
                        f"Extra columns in test data: {extra_in_test}")
    
    def test_column_name_format_validation(self):
        """Test that column names follow expected format"""
        # Check for hyphen vs underscore consistency
        train_id_cols = [col for col in self.train_data.columns if col.startswith('id_')]
        test_id_cols = [col for col in self.test_data.columns if col.startswith('id_')]
        
        # Ensure all id columns use underscores, not hyphens
        hyphen_cols_train = [col for col in self.train_data.columns if '-' in col]
        hyphen_cols_test = [col for col in self.test_data.columns if '-' in col]
        
        self.assertEqual(len(hyphen_cols_train), 0, 
                        f"Found hyphen columns in train: {hyphen_cols_train}")
        self.assertEqual(len(hyphen_cols_test), 0,
                        f"Found hyphen columns in test: {hyphen_cols_test}")
    
    def test_fix_column_naming_inconsistency(self):
        """Test function to fix column naming inconsistencies"""
        # Create test data with inconsistent naming
        inconsistent_test = self.test_data.copy()
        inconsistent_test.columns = [col.replace('_', '-') if col.startswith('id_') 
                                   else col for col in inconsistent_test.columns]
        
        # Function to standardize column names
        def standardize_column_names(df):
            """Standardize column names to use underscores"""
            df = df.copy()
            df.columns = [col.replace('-', '_') for col in df.columns]
            return df
        
        # Apply standardization
        fixed_test = standardize_column_names(inconsistent_test)
        
        # Verify fix worked
        train_cols = set(self.train_data.columns) - {'isFraud'}
        fixed_test_cols = set(fixed_test.columns)
        
        self.assertEqual(train_cols, fixed_test_cols,
                        "Column standardization failed")


class TestFeatureEngineeringPipeline(unittest.TestCase):
    """Test feature engineering pipeline robustness"""
    
    def setUp(self):
        """Set up test data and configuration"""
        self.config = FeatureConfig(
            velocity_windows=[1, 24],
            cyclical_features=['hour'],
            amount_percentiles=[0.25, 0.75],
            outlier_threshold=2.0,
            random_state=42
        )
        
        # Create consistent test data
        self.train_df = pd.DataFrame({
            'TransactionDT': np.random.randint(1000, 10000, 100),
            'TransactionAmt': np.random.uniform(1, 1000, 100),
            'id_01': np.random.uniform(0, 1, 100),
            'id_02': np.random.uniform(0, 1, 100),
            'C1': np.random.randint(0, 10, 100),
            'isFraud': np.random.randint(0, 2, 100)
        })
        
        self.test_df = pd.DataFrame({
            'TransactionDT': np.random.randint(1000, 10000, 80),
            'TransactionAmt': np.random.uniform(1, 1000, 80),
            'id_01': np.random.uniform(0, 1, 80),
            'id_02': np.random.uniform(0, 1, 80),
            'C1': np.random.randint(0, 10, 80),
        })
    
    def test_pipeline_column_validation(self):
        """Test that pipeline validates column consistency"""
        pipeline = FeatureEngineeringPipeline(self.config)
        
        # Fit pipeline
        pipeline.fit(self.train_df)
        
        # Test with consistent columns
        try:
            result = pipeline.transform(self.test_df)
            self.assertIsNotNone(result, "Pipeline should handle consistent columns")
        except Exception as e:
            self.fail(f"Pipeline failed with consistent columns: {e}")
    
    def test_pipeline_handles_missing_features(self):
        """Test pipeline behavior with missing features"""
        pipeline = FeatureEngineeringPipeline(self.config)
        pipeline.fit(self.train_df)
        
        # Create test data missing some features
        incomplete_test = self.test_df.drop(['id_01'], axis=1)
        
        # Should handle missing features gracefully
        with self.assertRaises(KeyError):
            pipeline.transform(incomplete_test)
    
    @patch('feature_engineering.logger')
    def test_pipeline_error_logging(self, mock_logger):
        """Test that pipeline logs errors appropriately"""
        pipeline = FeatureEngineeringPipeline(self.config)
        pipeline.fit(self.train_df)
        
        # Create problematic test data
        bad_test = pd.DataFrame({'wrong_col': [1, 2, 3]})
        
        try:
            pipeline.transform(bad_test)
        except:
            # Verify error was logged
            mock_logger.error.assert_called()


class TestFeatureSelectionConsistency(unittest.TestCase):
    """Test feature selection consistency between train/test"""
    
    def setUp(self):
        """Set up test data for feature selection"""
        np.random.seed(42)
        n_samples = 1000
        n_features = 50
        
        # Create synthetic data
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)
        
        # Create feature names
        feature_names = [f'feature_{i:02d}' for i in range(n_features)]
        
        self.train_df = pd.DataFrame(X, columns=feature_names)
        self.train_df['target'] = y
        
        # Create test data with same features
        X_test = np.random.randn(800, n_features)
        self.test_df = pd.DataFrame(X_test, columns=feature_names)
    
    def test_feature_selector_consistency(self):
        """Test that feature selector maintains consistency"""
        selector = AdvancedFeatureSelector(
            target_column='target',
            max_features=20,
            correlation_threshold=0.95
        )
        
        # Fit selector
        selector.fit(self.train_df)
        
        # Get selected features
        selected_features = selector.get_selected_features()
        
        # Verify test data has all selected features
        missing_features = set(selected_features) - set(self.test_df.columns)
        self.assertEqual(len(missing_features), 0,
                        f"Test data missing selected features: {missing_features}")
    
    def test_feature_selector_handles_missing_columns(self):
        """Test feature selector with missing columns in test data"""
        selector = AdvancedFeatureSelector(
            target_column='target',
            max_features=20
        )
        
        selector.fit(self.train_df)
        
        # Remove some columns from test data
        incomplete_test = self.test_df.drop(['feature_01', 'feature_02'], axis=1)
        
        # Should raise KeyError for missing features
        with self.assertRaises(KeyError):
            selector.transform(incomplete_test)


class TestErrorHandlingAndRecovery(unittest.TestCase):
    """Test error handling and recovery mechanisms"""
    
    def test_column_name_standardization_function(self):
        """Test utility function for column name standardization"""
        
        def standardize_feature_names(df):
            """Standardize feature names to use underscores"""
            df = df.copy()
            # Replace hyphens with underscores
            df.columns = [col.replace('-', '_') for col in df.columns]
            return df
        
        # Test data with mixed naming
        test_df = pd.DataFrame({
            'id-01': [1, 2, 3],
            'id_02': [4, 5, 6],
            'C-1': [7, 8, 9],
            'C_2': [10, 11, 12]
        })
        
        standardized = standardize_feature_names(test_df)
        
        # Check all columns use underscores
        hyphen_cols = [col for col in standardized.columns if '-' in col]
        self.assertEqual(len(hyphen_cols), 0, "Standardization failed")
        
        # Check expected columns exist
        expected_cols = ['id_01', 'id_02', 'C_1', 'C_2']
        self.assertEqual(set(standardized.columns), set(expected_cols))
    
    def test_feature_validation_utility(self):
        """Test utility function for feature validation"""
        
        def validate_feature_consistency(train_df, test_df, exclude_cols=None):
            """Validate feature consistency between train and test"""
            if exclude_cols is None:
                exclude_cols = []
            
            train_cols = set(train_df.columns) - set(exclude_cols)
            test_cols = set(test_df.columns)
            
            missing_in_test = train_cols - test_cols
            extra_in_test = test_cols - train_cols
            
            return {
                'missing_in_test': list(missing_in_test),
                'extra_in_test': list(extra_in_test),
                'is_consistent': len(missing_in_test) == 0 and len(extra_in_test) == 0
            }
        
        # Test with consistent data
        train_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'target': [0, 1]})
        test_df = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
        
        result = validate_feature_consistency(train_df, test_df, ['target'])
        self.assertTrue(result['is_consistent'])
        
        # Test with inconsistent data
        inconsistent_test = pd.DataFrame({'A': [5, 6], 'C': [7, 8]})
        result = validate_feature_consistency(train_df, inconsistent_test, ['target'])
        self.assertFalse(result['is_consistent'])
        self.assertIn('B', result['missing_in_test'])
        self.assertIn('C', result['extra_in_test'])


class TestMainPipelineIntegration(unittest.TestCase):
    """Integration tests for main pipeline functions"""
    
    @patch('main.clean_credit_data')
    @patch('main.create_feature_engineering_pipeline')
    def test_main_pipeline_error_handling(self, mock_fe_pipeline, mock_clean_data):
        """Test main pipeline handles errors gracefully"""
        
        # Mock successful data cleaning
        mock_clean_data.return_value = (
            pd.DataFrame({'A': [1, 2, 3]}),  # cleaned data
            Mock()  # cleaner instance
        )
        
        # Mock feature engineering failure
        mock_fe_pipeline.side_effect = KeyError("Features not in index")
        
        # Import and test (would need actual main module)
        # This is a template for testing main pipeline
        from main import apply_feature_engineering, FeatureConfig
        
        config = FeatureConfig()
        train_df = pd.DataFrame({'TransactionDT': [1, 2], 'TransactionAmt': [10, 20]})
        test_df = pd.DataFrame({'TransactionDT': [3, 4], 'TransactionAmt': [30, 40]})
        
        # Should handle the error and return None values
        result = apply_feature_engineering(train_df, test_df, config)
        self.assertIsNone(result[0])  # train_engineered should be None
        self.assertIsNone(result[1])  # test_engineered should be None


if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestColumnConsistency,
        TestFeatureEngineeringPipeline,
        TestFeatureSelectionConsistency,
        TestErrorHandlingAndRecovery,
        TestMainPipelineIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError: ')[-1].split('\n')[0]}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('Error: ')[-1].split('\n')[0]}")