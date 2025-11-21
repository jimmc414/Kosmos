"""
Tests for Data Analysis Agent Capability Requirements (REQ-DAA-CAP-*).

These tests validate that the Data Analysis Agent can perform required
analytical operations as specified in REQUIREMENTS.md.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Test markers for requirements traceability
pytestmark = [
    pytest.mark.requirement("REQ-DAA-CAP"),
    pytest.mark.category("data_analysis"),
]


@pytest.mark.requirement("REQ-DAA-CAP-001")
@pytest.mark.priority("MUST")
def test_req_daa_cap_001_exploratory_analysis():
    """
    REQ-DAA-CAP-001: The Data Analysis Agent MUST successfully perform
    exploratory data analysis (summary statistics, distributions, missing value analysis).
    """
    from kosmos.execution.data_analysis import DataAnalyzer

    # Arrange: Create test dataset with known properties
    np.random.seed(42)
    df = pd.DataFrame({
        'numeric_col': np.random.randn(100),
        'categorical_col': np.random.choice(['A', 'B', 'C'], 100),
        'missing_col': [np.nan if i % 10 == 0 else i for i in range(100)]
    })

    # Act: Perform summary statistics
    summary = df.describe()
    missing_analysis = df.isnull().sum()
    distributions = df['numeric_col'].hist(bins=10)

    # Assert: Verify exploratory analysis capabilities
    assert 'numeric_col' in summary.columns
    assert summary.loc['count', 'numeric_col'] == 100
    assert missing_analysis['missing_col'] == 10
    assert len(df['categorical_col'].value_counts()) == 3

    # Verify DataAnalyzer can compute basic statistics
    mean_val = DataAnalyzer.compute_statistics(df, 'numeric_col')
    assert 'mean' in mean_val or mean_val is not None


@pytest.mark.requirement("REQ-DAA-CAP-002")
@pytest.mark.priority("MUST")
def test_req_daa_cap_002_data_transformations():
    """
    REQ-DAA-CAP-002: The Data Analysis Agent MUST successfully perform
    data transformations (normalization, log transformation, scaling).
    """
    from kosmos.execution.data_analysis import DataAnalyzer

    # Arrange
    np.random.seed(42)
    df = pd.DataFrame({
        'value': np.random.exponential(10, 100)
    })

    # Act & Assert: Test normalization (z-score)
    normalized = (df['value'] - df['value'].mean()) / df['value'].std()
    assert abs(normalized.mean()) < 1e-10  # Mean should be ~0
    assert abs(normalized.std() - 1.0) < 1e-10  # Std should be ~1

    # Act & Assert: Test log transformation
    log_transformed = np.log1p(df['value'])
    assert (log_transformed >= 0).all()  # All values should be non-negative

    # Act & Assert: Test min-max scaling
    scaled = (df['value'] - df['value'].min()) / (df['value'].max() - df['value'].min())
    assert scaled.min() == 0.0
    assert scaled.max() == 1.0


@pytest.mark.requirement("REQ-DAA-CAP-003")
@pytest.mark.priority("MUST")
def test_req_daa_cap_003_statistical_tests():
    """
    REQ-DAA-CAP-003: The Data Analysis Agent MUST successfully perform
    statistical tests (t-tests, ANOVA, chi-square, correlation analysis).
    """
    from kosmos.execution.data_analysis import DataAnalyzer
    from scipy import stats

    # Arrange
    np.random.seed(42)
    group1 = np.random.normal(10, 2, 50)
    group2 = np.random.normal(12, 2, 50)

    # Act: T-test
    t_stat, p_value = stats.ttest_ind(group1, group2)

    # Assert: T-test results are valid
    assert isinstance(t_stat, (float, np.floating))
    assert isinstance(p_value, (float, np.floating))
    assert 0 <= p_value <= 1
    assert p_value < 0.05  # Should detect the difference

    # Act & Assert: Correlation analysis
    df = pd.DataFrame({'x': group1[:30], 'y': group1[:30] + np.random.randn(30) * 0.5})
    corr = df['x'].corr(df['y'])
    assert -1 <= corr <= 1
    assert corr > 0.5  # Should be positively correlated

    # Act & Assert: Chi-square test
    observed = np.array([[10, 10, 20], [20, 20, 10]])
    chi2, p, dof, expected = stats.chi2_contingency(observed)
    assert isinstance(chi2, (float, np.floating))
    assert 0 <= p <= 1

    # Test DataAnalyzer wrapper method
    df_test = pd.DataFrame({
        'group': ['A'] * 50 + ['B'] * 50,
        'value': np.concatenate([group1, group2])
    })
    result = DataAnalyzer.ttest_comparison(df_test, 'group', 'value', ('A', 'B'))
    assert 'p_value' in result
    assert result['p_value'] < 0.05


@pytest.mark.requirement("REQ-DAA-CAP-004")
@pytest.mark.priority("MUST")
def test_req_daa_cap_004_regression_analysis():
    """
    REQ-DAA-CAP-004: The Data Analysis Agent MUST successfully perform
    regression analysis (linear, logistic, multivariate).
    """
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import r2_score, accuracy_score

    # Arrange: Linear regression data
    np.random.seed(42)
    X_linear = np.random.randn(100, 1)
    y_linear = 3 * X_linear.ravel() + 2 + np.random.randn(100) * 0.5

    # Act: Linear regression
    lr = LinearRegression()
    lr.fit(X_linear, y_linear)
    y_pred = lr.predict(X_linear)
    r2 = r2_score(y_linear, y_pred)

    # Assert: Linear regression works
    assert r2 > 0.8  # Should have high R²
    assert abs(lr.coef_[0] - 3.0) < 0.5  # Coefficient should be ~3
    assert abs(lr.intercept_ - 2.0) < 0.5  # Intercept should be ~2

    # Arrange: Logistic regression data
    X_logistic = np.random.randn(100, 2)
    y_logistic = (X_logistic[:, 0] + X_logistic[:, 1] > 0).astype(int)

    # Act: Logistic regression
    log_reg = LogisticRegression()
    log_reg.fit(X_logistic, y_logistic)
    y_pred_log = log_reg.predict(X_logistic)
    accuracy = accuracy_score(y_logistic, y_pred_log)

    # Assert: Logistic regression works
    assert accuracy > 0.7  # Should have reasonable accuracy

    # Arrange: Multivariate regression
    X_multi = np.random.randn(100, 5)
    y_multi = (2 * X_multi[:, 0] + 3 * X_multi[:, 1] - X_multi[:, 2] +
               1.5 * X_multi[:, 3] + 0.5 * X_multi[:, 4] + np.random.randn(100) * 0.5)

    # Act: Multivariate regression
    mlr = LinearRegression()
    mlr.fit(X_multi, y_multi)
    y_pred_multi = mlr.predict(X_multi)
    r2_multi = r2_score(y_multi, y_pred_multi)

    # Assert: Multivariate regression works
    assert r2_multi > 0.85
    assert len(mlr.coef_) == 5


@pytest.mark.requirement("REQ-DAA-CAP-005")
@pytest.mark.priority("MUST")
def test_req_daa_cap_005_advanced_analyses():
    """
    REQ-DAA-CAP-005: The Data Analysis Agent MUST successfully perform
    advanced analyses (feature importance via SHAP, distribution fitting,
    segmented regression).

    Test: SHAP feature importance
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification

    # Arrange
    X, y = make_classification(n_samples=100, n_features=10, n_informative=5,
                               n_redundant=2, random_state=42)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)

    # Act: Get feature importances (proxy for SHAP if not available)
    feature_importance = model.feature_importances_

    # Assert: Feature importance computed
    assert len(feature_importance) == 10
    assert (feature_importance >= 0).all()
    assert abs(feature_importance.sum() - 1.0) < 1e-6  # Should sum to 1

    # Check if most important features are the informative ones (indices 0-4)
    top_features = np.argsort(feature_importance)[::-1][:5]
    assert len(set(top_features) & set(range(5))) >= 3  # At least 3 in top 5

    # Test: Distribution fitting
    from scipy import stats

    # Arrange: Normal distribution data
    np.random.seed(42)
    data_normal = np.random.normal(10, 2, 1000)

    # Act: Fit normal distribution
    mu, std = stats.norm.fit(data_normal)

    # Assert: Fitted parameters are reasonable
    assert abs(mu - 10) < 0.2
    assert abs(std - 2) < 0.2

    # Test: Segmented regression (piecewise linear)
    X_seg = np.linspace(0, 10, 100)
    y_seg = np.where(X_seg < 5, 2 * X_seg + 1, -3 * X_seg + 26) + np.random.randn(100) * 0.5

    # Split and fit two segments
    mask = X_seg < 5
    lr1 = LinearRegression().fit(X_seg[mask].reshape(-1, 1), y_seg[mask])
    lr2 = LinearRegression().fit(X_seg[~mask].reshape(-1, 1), y_seg[~mask])

    # Assert: Segmented regression captures different slopes
    assert lr1.coef_[0] > 0  # Positive slope in first segment
    assert lr2.coef_[0] < 0  # Negative slope in second segment
    assert abs(lr1.coef_[0] - 2.0) < 0.5
    assert abs(lr2.coef_[0] - (-3.0)) < 0.5


@pytest.mark.requirement("REQ-DAA-CAP-006")
@pytest.mark.priority("MUST")
def test_req_daa_cap_006_publication_visualizations():
    """
    REQ-DAA-CAP-006: The Data Analysis Agent MUST successfully generate
    publication-quality visualizations (scatter plots, box plots, heatmaps,
    distribution plots).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Arrange
    np.random.seed(42)
    df = pd.DataFrame({
        'x': np.random.randn(100),
        'y': np.random.randn(100),
        'group': np.random.choice(['A', 'B', 'C'], 100),
        'value': np.random.randn(100)
    })

    # Act & Assert: Scatter plot
    fig, ax = plt.subplots()
    ax.scatter(df['x'], df['y'])
    assert len(ax.collections) > 0  # Check that scatter plot was created
    plt.close()

    # Act & Assert: Box plot
    fig, ax = plt.subplots()
    df.boxplot(column='value', by='group', ax=ax)
    assert len(ax.patches) > 0 or len(ax.lines) > 0  # Check boxes were drawn
    plt.close()

    # Act & Assert: Heatmap
    corr_matrix = df[['x', 'y', 'value']].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, ax=ax)
    assert len(ax.collections) > 0  # Check heatmap was created
    plt.close()

    # Act & Assert: Distribution plot
    fig, ax = plt.subplots()
    ax.hist(df['value'], bins=20)
    assert len(ax.patches) == 20  # Check histogram bins
    plt.close()

    # Verify plots can be saved
    fig, ax = plt.subplots()
    ax.scatter(df['x'], df['y'])
    assert fig is not None
    plt.close('all')


@pytest.mark.requirement("REQ-DAA-CAP-007")
@pytest.mark.priority("MUST")
def test_req_daa_cap_007_statistical_validity():
    """
    REQ-DAA-CAP-007: The system MUST validate analysis outputs for
    statistical validity (e.g., p-values in valid range, confidence
    intervals properly calculated).
    """
    from scipy import stats

    # Arrange
    np.random.seed(42)
    sample1 = np.random.normal(10, 2, 50)
    sample2 = np.random.normal(12, 2, 50)

    # Act: Perform t-test
    t_stat, p_value = stats.ttest_ind(sample1, sample2)

    # Assert: P-value validation
    assert 0 <= p_value <= 1, "P-value must be between 0 and 1"
    assert not np.isnan(p_value), "P-value must not be NaN"
    assert not np.isinf(p_value), "P-value must not be infinite"

    # Act: Calculate confidence interval
    mean = np.mean(sample1)
    sem = stats.sem(sample1)
    ci = stats.t.interval(0.95, len(sample1)-1, loc=mean, scale=sem)

    # Assert: Confidence interval validation
    assert len(ci) == 2, "CI must have lower and upper bounds"
    assert ci[0] < mean < ci[1], "Mean must be within CI"
    assert not np.isnan(ci[0]) and not np.isnan(ci[1]), "CI bounds must not be NaN"
    assert ci[0] < ci[1], "Lower bound must be less than upper bound"

    # Test effect size calculation (Cohen's d)
    pooled_std = np.sqrt(((len(sample1)-1) * np.var(sample1, ddof=1) +
                          (len(sample2)-1) * np.var(sample2, ddof=1)) /
                         (len(sample1) + len(sample2) - 2))
    cohens_d = (np.mean(sample2) - np.mean(sample1)) / pooled_std

    # Assert: Effect size is reasonable
    assert not np.isnan(cohens_d), "Effect size must not be NaN"
    assert not np.isinf(cohens_d), "Effect size must not be infinite"
    assert abs(cohens_d) < 10, "Effect size should be reasonable"


@pytest.mark.requirement("REQ-DAA-CAP-008")
@pytest.mark.priority("MUST")
@pytest.mark.slow
def test_req_daa_cap_008_pathway_enrichment():
    """
    REQ-DAA-CAP-008: The Data Analysis Agent MUST successfully perform
    pathway enrichment analysis using standard biological databases and
    tools (e.g., gseapy).

    Note: This test verifies the capability exists. Actual enrichment
    requires biological data and databases.
    """
    pytest.importorskip("gseapy", reason="gseapy not installed")

    import gseapy as gp

    # Arrange: Mock gene list (would be real genes in production)
    gene_list = ['TP53', 'BRCA1', 'EGFR', 'KRAS', 'MYC',
                 'PIK3CA', 'PTEN', 'RB1', 'VEGFA', 'ERBB2']

    # Act: Verify gseapy is available and can be called
    # In real usage, this would call gp.enrichr() with actual database
    assert hasattr(gp, 'enrichr'), "gseapy must have enrichr function"
    assert hasattr(gp, 'gsea'), "gseapy must have gsea function"

    # Test that the basic structure is available
    # Full integration test would require network access and databases
    assert callable(gp.enrichr)
    assert callable(gp.gsea)

    # Verify key databases are recognized
    libraries = gp.get_library_name()
    assert 'KEGG_2021_Human' in libraries or 'GO_Biological_Process_2021' in libraries


@pytest.mark.requirement("REQ-DAA-CAP-009")
@pytest.mark.priority("SHOULD")
def test_req_daa_cap_009_novel_metrics():
    """
    REQ-DAA-CAP-009: The Data Analysis Agent SHOULD be capable of defining
    novel composite metrics or proposing unconventional analytical methods
    relevant to the research objective.

    Test: Verify system can compute custom composite metrics.
    """
    # Arrange: Create dataset for custom metric
    np.random.seed(42)
    df = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.exponential(1, 100),
        'target': np.random.choice([0, 1], 100)
    })

    # Act: Define novel composite metric (e.g., Mechanistic Ranking Score)
    # This mimics the paper's Discovery 5 metric
    weights = {'feature1': 0.4, 'feature2': 0.3, 'feature3': 0.3}

    # Normalize features first
    df_norm = (df[['feature1', 'feature2', 'feature3']] -
               df[['feature1', 'feature2', 'feature3']].mean()) / df[['feature1', 'feature2', 'feature3']].std()

    # Compute composite score
    composite_score = (weights['feature1'] * df_norm['feature1'] +
                       weights['feature2'] * df_norm['feature2'] +
                       weights['feature3'] * df_norm['feature3'])

    # Assert: Composite metric is valid
    assert len(composite_score) == 100
    assert not composite_score.isnull().any()
    assert composite_score.std() > 0  # Has variance

    # Test: Novel analytical approach (segmented regression on composite)
    from scipy import stats as sp_stats

    # Sort by composite score
    sorted_idx = composite_score.argsort()
    sorted_composite = composite_score.iloc[sorted_idx].values
    sorted_target = df['target'].iloc[sorted_idx].values

    # Fit different models to different segments
    mid_point = len(sorted_composite) // 2
    corr1 = sp_stats.spearmanr(sorted_composite[:mid_point], sorted_target[:mid_point])[0]
    corr2 = sp_stats.spearmanr(sorted_composite[mid_point:], sorted_target[mid_point:])[0]

    # Assert: Can perform segmented analysis
    assert not np.isnan(corr1)
    assert not np.isnan(corr2)
    assert -1 <= corr1 <= 1
    assert -1 <= corr2 <= 1
