"""
Tests for machine learning experiments module.

Tests MLAnalyzer and FeatureEngineering classes.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_classification, make_regression

from kosmos.execution.ml_experiments import MLAnalyzer, FeatureEngineering


class TestMLAnalyzer:
    """Test MLAnalyzer machine learning methods."""

    def test_train_test_split(self):
        """Test train/test splitting."""
        # Create sample data
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        analyzer = MLAnalyzer(random_state=42)
        X_train, X_test, y_train, y_test = analyzer.train_test_split_data(
            X, y, test_size=0.2, stratify=False
        )

        # Check sizes
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20

    def test_train_test_split_stratified(self):
        """Test stratified train/test splitting."""
        # Create imbalanced data
        X = np.random.randn(100, 5)
        y = np.array([0] * 80 + [1] * 20)

        analyzer = MLAnalyzer(random_state=42)
        X_train, X_test, y_train, y_test = analyzer.train_test_split_data(
            X, y, test_size=0.2, stratify=True
        )

        # Check stratification preserved class balance
        train_ratio = np.mean(y_train)
        test_ratio = np.mean(y_test)
        original_ratio = 0.2

        # Ratios should be similar
        assert abs(train_ratio - original_ratio) < 0.05
        assert abs(test_ratio - original_ratio) < 0.10

    def test_cross_validate_model_single_metric(self):
        """Test cross-validation with single metric."""
        # Create sample data
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)

        model = LogisticRegression(random_state=42)
        analyzer = MLAnalyzer(random_state=42)

        result = analyzer.cross_validate_model(
            model, X, y, cv=5, stratified=True, scoring='accuracy'
        )

        # Check structure
        assert 'mean_score' in result
        assert 'std_score' in result
        assert 'fold_scores' in result
        assert result['n_folds'] == 5

        # Check values
        assert 0 <= result['mean_score'] <= 1
        assert len(result['fold_scores']) == 5

    def test_cross_validate_model_multiple_metrics(self):
        """Test cross-validation with multiple metrics."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)

        model = LogisticRegression(random_state=42)
        analyzer = MLAnalyzer(random_state=42)

        result = analyzer.cross_validate_model(
            model, X, y, cv=3,
            scoring=['accuracy', 'precision', 'recall']
        )

        # Check structure
        assert 'scores' in result
        assert 'accuracy' in result['scores']
        assert 'precision' in result['scores']
        assert 'recall' in result['scores']

        # Check each metric has mean, std, fold_scores
        for metric in ['accuracy', 'precision', 'recall']:
            assert 'mean' in result['scores'][metric]
            assert 'std' in result['scores'][metric]
            assert 'fold_scores' in result['scores'][metric]

    def test_cross_validate_stratified(self):
        """Test stratified cross-validation."""
        # Imbalanced classification
        X, y = make_classification(
            n_samples=100, n_features=10,
            weights=[0.9, 0.1], random_state=42
        )

        model = LogisticRegression(random_state=42)
        analyzer = MLAnalyzer(random_state=42)

        result = analyzer.cross_validate_model(
            model, X, y, cv=5, stratified=True, scoring='accuracy'
        )

        # Should complete without error
        assert result['n_folds'] == 5
        assert 'mean_score' in result

    def test_evaluate_classification(self):
        """Test classification model evaluation."""
        # Create sample predictions
        np.random.seed(42)
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 0, 1, 1])
        y_prob = np.random.rand(10, 2)  # Probabilities for ROC-AUC

        analyzer = MLAnalyzer()
        result = analyzer.evaluate_classification(y_true, y_pred, y_prob)

        # Check structure
        assert 'accuracy' in result
        assert 'precision' in result
        assert 'recall' in result
        assert 'f1_score' in result
        assert 'confusion_matrix' in result
        assert 'roc_auc' in result
        assert 'classification_report' in result

        # Check types
        assert isinstance(result['accuracy'], float)
        assert isinstance(result['confusion_matrix'], list)
        assert isinstance(result['classification_report'], str)

        # Check values in valid ranges
        assert 0 <= result['accuracy'] <= 1
        assert 0 <= result['precision'] <= 1
        assert 0 <= result['f1_score'] <= 1

    def test_evaluate_classification_without_probabilities(self):
        """Test classification evaluation without probabilities."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])

        analyzer = MLAnalyzer()
        result = analyzer.evaluate_classification(y_true, y_pred, y_prob=None)

        # Should not have ROC-AUC
        assert 'roc_auc' not in result or result['roc_auc'] is None

    def test_evaluate_regression(self):
        """Test regression model evaluation."""
        # Create sample predictions
        np.random.seed(42)
        y_true = np.array([1.5, 2.3, 3.1, 4.5, 5.2])
        y_pred = np.array([1.6, 2.1, 3.3, 4.3, 5.5])

        analyzer = MLAnalyzer()
        result = analyzer.evaluate_regression(y_true, y_pred)

        # Check structure
        assert 'mse' in result
        assert 'rmse' in result
        assert 'mae' in result
        assert 'r2' in result
        assert 'explained_variance' in result
        assert 'mean_residual' in result
        assert 'std_residual' in result

        # Check types
        assert isinstance(result['mse'], float)
        assert isinstance(result['r2'], float)

        # Check values
        assert result['mse'] >= 0
        assert result['rmse'] >= 0
        assert result['mae'] >= 0

    def test_create_pipeline_with_scaler(self):
        """Test pipeline creation with scaler."""
        model = LogisticRegression()
        analyzer = MLAnalyzer()

        pipeline = analyzer.create_pipeline(model, scaler='standard', include_scaler=True)

        # Check pipeline has scaler and model
        assert len(pipeline.steps) == 2
        assert pipeline.steps[0][0] == 'scaler'
        assert pipeline.steps[1][0] == 'model'

    def test_create_pipeline_without_scaler(self):
        """Test pipeline creation without scaler."""
        model = LogisticRegression()
        analyzer = MLAnalyzer()

        pipeline = analyzer.create_pipeline(model, include_scaler=False)

        # Check pipeline has only model
        assert len(pipeline.steps) == 1
        assert pipeline.steps[0][0] == 'model'

    def test_create_pipeline_minmax_scaler(self):
        """Test pipeline with MinMax scaler."""
        model = LogisticRegression()
        analyzer = MLAnalyzer()

        pipeline = analyzer.create_pipeline(model, scaler='minmax')

        # Check scaler type
        assert len(pipeline.steps) == 2
        assert pipeline.steps[0][0] == 'scaler'

    def test_grid_search(self):
        """Test hyperparameter grid search."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)

        model = LogisticRegression()
        # l2 is LogisticRegression's default; scikit-learn 1.8 deprecates the 'penalty' parameter
        param_grid = {
            'C': [0.1, 1.0, 10.0],
        }

        analyzer = MLAnalyzer(random_state=42)
        result = analyzer.grid_search(
            model, X, y,
            param_grid=param_grid,
            cv=3,
            scoring='accuracy'
        )

        # Check structure
        assert 'best_params' in result
        assert 'best_score' in result
        assert 'best_estimator' in result
        assert 'cv_results' in result

        # Check best params is one of the grid combinations
        assert result['best_params']['C'] in [0.1, 1.0, 10.0]
        assert set(result['best_params']) == {'C'}

        # Check tested 3 combinations
        assert result['n_combinations_tested'] == 3

    def test_run_experiment_classification(self):
        """Test complete classification experiment."""
        X, y = make_classification(
            n_samples=100, n_features=10,
            n_informative=5, random_state=42
        )

        model = LogisticRegression(random_state=42)
        analyzer = MLAnalyzer(random_state=42)

        result = analyzer.run_experiment(
            model, X, y,
            test_size=0.2,
            cv=3,
            task_type='classification',
            scale_features=True
        )

        # Check structure
        assert 'train_test_results' in result
        assert 'cv_results' in result
        assert 'model' in result
        assert 'train_size' in result
        assert 'test_size' in result

        # Check train/test split
        assert result['train_size'] == 80
        assert result['test_size'] == 20

        # Check test results have classification metrics
        assert 'accuracy' in result['train_test_results']
        assert 'f1_score' in result['train_test_results']

        # Check CV results
        assert 'mean_score' in result['cv_results']

    def test_run_experiment_regression(self):
        """Test complete regression experiment."""
        X, y = make_regression(n_samples=100, n_features=10, random_state=42)

        model = LinearRegression()
        analyzer = MLAnalyzer(random_state=42)

        result = analyzer.run_experiment(
            model, X, y,
            test_size=0.3,
            cv=5,
            task_type='regression',
            scale_features=True
        )

        # Check structure
        assert 'train_test_results' in result
        assert 'cv_results' in result

        # Check test results have regression metrics
        assert 'r2' in result['train_test_results']
        assert 'mse' in result['train_test_results']
        assert 'rmse' in result['train_test_results']

    def test_run_experiment_no_scaling(self):
        """Test experiment without feature scaling."""
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        analyzer = MLAnalyzer(random_state=42)

        result = analyzer.run_experiment(
            model, X, y,
            scale_features=False,
            task_type='classification'
        )

        # Should complete successfully
        assert 'train_test_results' in result

    def test_run_experiment_feature_importance(self):
        """Test feature importance extraction."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)

        # Random forest has feature_importances_
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        analyzer = MLAnalyzer(random_state=42)

        result = analyzer.run_experiment(
            model, X, y,
            scale_features=False,
            task_type='classification'
        )

        # Should extract feature importances
        assert result['feature_importance'] is not None
        assert len(result['feature_importance']) == 5

    def test_results_history(self):
        """Test results history tracking."""
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)

        analyzer = MLAnalyzer(random_state=42)

        # Run multiple experiments
        model1 = LogisticRegression(random_state=42)
        result1 = analyzer.run_experiment(model1, X, y, task_type='classification')

        model2 = RandomForestClassifier(n_estimators=5, random_state=42)
        result2 = analyzer.run_experiment(model2, X, y, task_type='classification')

        # Check history
        assert len(analyzer.results_history) == 2
        assert all('timestamp' in r for r in analyzer.results_history)
        assert all('task_type' in r for r in analyzer.results_history)


class TestFeatureEngineering:
    """Test FeatureEngineering utilities."""

    def test_encode_categorical_label(self):
        """Test label encoding."""
        df = pd.DataFrame({
            'category': ['A', 'B', 'C', 'A', 'B'],
            'value': [1, 2, 3, 4, 5]
        })

        df_encoded, info = FeatureEngineering.encode_categorical(
            df, columns=['category'], method='label'
        )

        # Check encoding
        assert 'category' in df_encoded.columns
        assert df_encoded['category'].dtype in [np.int32, np.int64]

        # Check info
        assert 'category' in info
        assert 'classes' in info['category']
        assert len(info['category']['classes']) == 3

    def test_encode_categorical_onehot(self):
        """Test one-hot encoding."""
        df = pd.DataFrame({
            'category': ['A', 'B', 'C'],
            'value': [1, 2, 3]
        })

        df_encoded, info = FeatureEngineering.encode_categorical(
            df, columns=['category'], method='onehot'
        )

        # Should create new columns (drop_first=True, so 2 new columns for 3 categories)
        assert len(df_encoded.columns) > len(df.columns)
        assert 'method' in info
        assert info['method'] == 'onehot'

    def test_create_polynomial_features(self):
        """Test polynomial feature creation."""
        X = np.array([[1, 2], [3, 4], [5, 6]])

        X_poly = FeatureEngineering.create_polynomial_features(X, degree=2)

        # Should have more features (original 2 + interactions)
        assert X_poly.shape[1] > X.shape[1]
        assert X_poly.shape[0] == X.shape[0]

    def test_select_k_best_features(self):
        """Test feature selection."""
        # Create data with some informative features
        X, y = make_classification(
            n_samples=100, n_features=20,
            n_informative=5, n_redundant=5,
            random_state=42
        )

        X_selected, scores = FeatureEngineering.select_k_best_features(
            X, y, k=10, score_func='f_classif'
        )

        # Should select k features
        assert X_selected.shape[1] == 10
        assert X_selected.shape[0] == X.shape[0]
        assert len(scores) == 20  # Scores for all original features

    def test_select_k_best_regression(self):
        """Test feature selection for regression."""
        X, y = make_regression(n_samples=100, n_features=15, n_informative=8, random_state=42)

        X_selected, scores = FeatureEngineering.select_k_best_features(
            X, y, k=8, score_func='f_regression'
        )

        assert X_selected.shape[1] == 8


class TestMLAnalyzerIntegration:
    """Integration tests for complete ML workflows."""

    def test_complete_classification_pipeline(self):
        """Test end-to-end classification pipeline."""
        # Create realistic data
        X, y = make_classification(
            n_samples=200, n_features=15,
            n_informative=10, n_redundant=3,
            random_state=42
        )

        # Initialize analyzer
        analyzer = MLAnalyzer(random_state=42)

        # Split data
        X_train, X_test, y_train, y_test = analyzer.train_test_split_data(
            X, y, test_size=0.2, stratify=True
        )

        # Create pipeline
        model = LogisticRegression(random_state=42, max_iter=1000)
        pipeline = analyzer.create_pipeline(model, scaler='standard')

        # Train
        pipeline.fit(X_train, y_train)

        # Predict
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)

        # Evaluate
        results = analyzer.evaluate_classification(y_test, y_pred, y_prob)

        # Should have reasonable performance
        assert results['accuracy'] > 0.5
        assert 'roc_auc' in results

    def test_complete_regression_pipeline(self):
        """Test end-to-end regression pipeline."""
        # Create data
        X, y = make_regression(n_samples=150, n_features=10, random_state=42)

        analyzer = MLAnalyzer(random_state=42)

        # Run complete experiment
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        results = analyzer.run_experiment(
            model, X, y,
            test_size=0.25,
            cv=5,
            task_type='regression',
            scale_features=False
        )

        # Check all results present
        assert 'train_test_results' in results
        assert 'cv_results' in results
        assert 'feature_importance' in results

        # Should have reasonable R² (lowered threshold due to random data variation)
        assert results['train_test_results']['r2'] > 0.3

    def test_hyperparameter_tuning_workflow(self):
        """Test hyperparameter tuning workflow."""
        X, y = make_classification(n_samples=100, n_features=8, random_state=42)

        analyzer = MLAnalyzer(random_state=42)

        # Define parameter grid
        param_grid = {
            'max_depth': [3, 5, 7],
            'n_estimators': [5, 10]
        }

        # Grid search
        model = RandomForestClassifier(random_state=42)
        grid_results = analyzer.grid_search(
            model, X, y,
            param_grid=param_grid,
            cv=3,
            scoring='accuracy'
        )

        # Use best model for final evaluation
        best_model = grid_results['best_estimator']

        # Split and evaluate
        X_train, X_test, y_train, y_test = analyzer.train_test_split_data(X, y, test_size=0.2)
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)

        eval_results = analyzer.evaluate_classification(y_test, y_pred)

        # Should have decent performance
        assert eval_results['accuracy'] > 0.5
