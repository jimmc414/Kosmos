"""
Tests for data analysis pipeline.

Tests DataAnalyzer, DataLoader, and DataCleaner classes with methods
extracted from kosmos-figures analysis patterns.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from kosmos.execution.data_analysis import (
    DataAnalyzer, DataLoader, DataCleaner
)


class TestDataAnalyzer:
    """Test DataAnalyzer statistical methods."""

    def test_ttest_comparison_basic(self):
        """Test basic t-test comparison."""
        # Create test data
        np.random.seed(42)
        df = pd.DataFrame({
            'group': ['control'] * 50 + ['treatment'] * 50,
            'score': np.concatenate([
                np.random.normal(10, 2, 50),  # control: mean=10
                np.random.normal(12, 2, 50)   # treatment: mean=12
            ])
        })

        # Perform t-test
        result = DataAnalyzer.ttest_comparison(
            df, 'group', 'score', ('treatment', 'control')
        )

        # Check result structure
        assert 't_statistic' in result
        assert 'p_value' in result
        assert 'group1_mean' in result
        assert 'group2_mean' in result
        assert 'mean_difference' in result
        assert 'log2_fold_change' in result
        assert 'significance_label' in result

        # Check types
        assert isinstance(result['t_statistic'], float)
        assert isinstance(result['p_value'], float)
        assert isinstance(result['significant_0.05'], bool)

        # Check sample sizes
        assert result['n_group1'] == 50
        assert result['n_group2'] == 50

        # Check means are reasonable (within expected range)
        assert 11 < result['group1_mean'] < 13  # treatment ~ 12
        assert 9 < result['group2_mean'] < 11   # control ~ 10

    def test_ttest_comparison_with_log_transform(self):
        """Test t-test with log transformation."""
        np.random.seed(42)
        df = pd.DataFrame({
            'group': ['A'] * 30 + ['B'] * 30,
            'value': np.concatenate([
                np.random.exponential(10, 30),  # skewed data
                np.random.exponential(15, 30)
            ])
        })

        result = DataAnalyzer.ttest_comparison(
            df, 'group', 'value', ('B', 'A'), log_transform=True
        )

        assert 'log2_fold_change' in result
        assert isinstance(result['log2_fold_change'], (float, type(None)))

    def test_ttest_comparison_empty_group(self):
        """Test t-test with empty group raises error."""
        df = pd.DataFrame({
            'group': ['A'] * 10,
            'value': np.random.randn(10)
        })

        with pytest.raises(ValueError, match="One or both groups have no data"):
            DataAnalyzer.ttest_comparison(df, 'group', 'value', ('A', 'B'))

    def test_ttest_comparison_significance_labels(self):
        """Test significance label generation."""
        # Create data with known significant difference
        np.random.seed(42)
        df = pd.DataFrame({
            'group': ['A'] * 100 + ['B'] * 100,
            'score': np.concatenate([
                np.random.normal(10, 1, 100),
                np.random.normal(15, 1, 100)  # Large difference
            ])
        })

        result = DataAnalyzer.ttest_comparison(df, 'group', 'score', ('B', 'A'))

        # Should be highly significant
        assert result['p_value'] < 0.001
        assert result['significant_0.001'] is True
        assert result['significance_label'] == '***'

    def test_correlation_analysis_pearson(self):
        """Test Pearson correlation analysis."""
        # Create correlated data
        np.random.seed(42)
        x = np.random.randn(100)
        y = 0.7 * x + np.random.randn(100) * 0.3  # Strong positive correlation

        df = pd.DataFrame({'x': x, 'y': y})

        result = DataAnalyzer.correlation_analysis(df, 'x', 'y', method='pearson')

        # Check structure
        assert 'correlation' in result
        assert 'p_value' in result
        assert 'r_squared' in result
        assert 'slope' in result
        assert 'intercept' in result
        assert 'equation' in result

        # Check correlation is positive and strong
        assert result['correlation'] > 0.5
        assert result['p_value'] < 0.05
        assert result['n_samples'] == 100

        # Check equation format
        assert 'y = ' in result['equation']
        assert 'x' in result['equation']

    def test_correlation_analysis_spearman(self):
        """Test Spearman correlation analysis."""
        # Create monotonic but non-linear relationship
        np.random.seed(42)
        x = np.arange(100)
        y = x ** 2 + np.random.randn(100) * 10

        df = pd.DataFrame({'x': x, 'y': y})

        result = DataAnalyzer.correlation_analysis(df, 'x', 'y', method='spearman')

        assert result['method'] == 'spearman'
        # Spearman should detect monotonic relationship
        assert result['correlation'] > 0.9

    def test_correlation_analysis_with_nan(self):
        """Test correlation handles NaN values."""
        df = pd.DataFrame({
            'x': [1, 2, np.nan, 4, 5, 6],
            'y': [2, 4, 6, np.nan, 10, 12]
        })

        result = DataAnalyzer.correlation_analysis(df, 'x', 'y')

        # Should use only complete cases
        assert result['n_samples'] < 6

    def test_correlation_analysis_insufficient_data(self):
        """Test correlation with insufficient data raises error."""
        df = pd.DataFrame({'x': [1, 2], 'y': [2, 4]})

        with pytest.raises(ValueError, match="Insufficient data"):
            DataAnalyzer.correlation_analysis(df, 'x', 'y')

    def test_log_log_scaling_analysis(self):
        """Test log-log scaling analysis (power law fitting)."""
        # Create power law data: y = 2 * x^1.5
        np.random.seed(42)
        x = np.random.uniform(1, 100, 100)
        y = 2 * (x ** 1.5) * np.exp(np.random.randn(100) * 0.1)  # Add noise

        df = pd.DataFrame({'x': x, 'y': y})

        result = DataAnalyzer.log_log_scaling_analysis(df, 'x', 'y')

        # Check structure
        assert 'spearman_rho' in result
        assert 'p_value' in result
        assert 'power_law_exponent' in result
        assert 'power_law_coefficient' in result
        assert 'equation' in result

        # Check power law parameters are close to true values
        assert 1.3 < result['power_law_exponent'] < 1.7  # Should be ~1.5
        assert result['power_law_coefficient'] > 0  # Should be ~2

        # Check correlation is strong
        assert result['spearman_rho'] > 0.9

        # Check equation format
        assert 'y = ' in result['equation']
        assert 'x^' in result['equation']

    def test_log_log_scaling_removes_non_positive(self):
        """Test log-log analysis removes non-positive values."""
        df = pd.DataFrame({
            'x': [-1, 0, 1, 2, 3, 4, 5],
            'y': [10, 0, 2, 4, 6, 8, 10]
        })

        result = DataAnalyzer.log_log_scaling_analysis(df, 'x', 'y')

        # Should only use positive values (last 5 points)
        assert result['n_samples'] < 7

    def test_log_log_scaling_insufficient_data(self):
        """Test log-log analysis with insufficient positive data."""
        df = pd.DataFrame({
            'x': [0, -1],
            'y': [0, -1]
        })

        with pytest.raises(ValueError, match="Insufficient positive data"):
            DataAnalyzer.log_log_scaling_analysis(df, 'x', 'y')

    def test_anova_comparison(self):
        """Test one-way ANOVA."""
        # Create data with 3 groups
        np.random.seed(42)
        df = pd.DataFrame({
            'group': ['A'] * 30 + ['B'] * 30 + ['C'] * 30,
            'value': np.concatenate([
                np.random.normal(10, 2, 30),
                np.random.normal(12, 2, 30),
                np.random.normal(14, 2, 30)
            ])
        })

        result = DataAnalyzer.anova_comparison(df, 'group', 'value')

        # Check structure
        assert 'f_statistic' in result
        assert 'p_value' in result
        assert 'group_means' in result
        assert 'group_stds' in result
        assert 'eta_squared' in result

        # Check we have 3 groups
        assert result['n_groups'] == 3
        assert len(result['group_means']) == 3

        # Check means are in expected ranges
        assert 9 < result['group_means']['A'] < 11
        assert 11 < result['group_means']['B'] < 13
        assert 13 < result['group_means']['C'] < 15

        # Should be significant (groups have different means)
        assert result['p_value'] < 0.05

    def test_anova_insufficient_groups(self):
        """Test ANOVA with < 2 groups raises error."""
        df = pd.DataFrame({
            'group': ['A'] * 10,
            'value': np.random.randn(10)
        })

        with pytest.raises(ValueError, match="Need at least 2 groups"):
            DataAnalyzer.anova_comparison(df, 'group', 'value')


class TestDataLoader:
    """Test DataLoader file loading utilities."""

    def test_load_csv(self):
        """Test CSV loading."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('a,b,c\n')
            f.write('1,2,3\n')
            f.write('4,5,6\n')
            temp_path = f.name

        try:
            df = DataLoader.load_csv(temp_path)
            assert len(df) == 2
            assert list(df.columns) == ['a', 'b', 'c']
            assert df['a'].tolist() == [1, 4]
        finally:
            os.unlink(temp_path)

    def test_load_excel(self):
        """Test Excel loading."""
        pytest.importorskip("openpyxl")  # pandas' Excel engine; not a declared dependency
        # Create temporary Excel file
        df_original = pd.DataFrame({
            'x': [1, 2, 3],
            'y': [4, 5, 6]
        })

        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
            temp_path = f.name

        try:
            df_original.to_excel(temp_path, index=False)
            df_loaded = DataLoader.load_excel(temp_path)
            assert len(df_loaded) == 3
            assert list(df_loaded.columns) == ['x', 'y']
        finally:
            os.unlink(temp_path)

    def test_load_json(self):
        """Test JSON loading."""
        # Create temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('[{"a": 1, "b": 2}, {"a": 3, "b": 4}]')
            temp_path = f.name

        try:
            df = DataLoader.load_json(temp_path)
            assert len(df) == 2
            assert list(df.columns) == ['a', 'b']
        finally:
            os.unlink(temp_path)

    def test_load_data_autodetect_csv(self):
        """Test auto-detection for CSV."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('col1,col2\n1,2\n')
            temp_path = f.name

        try:
            df = DataLoader.load_data(temp_path)
            assert len(df) == 1
        finally:
            os.unlink(temp_path)

    def test_load_data_unsupported_format(self):
        """Test unsupported file format raises error."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported file type"):
                DataLoader.load_data(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_data_file_not_found(self):
        """Test loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            DataLoader.load_data('/nonexistent/file.csv')


class TestDataCleaner:
    """Test DataCleaner preprocessing utilities."""

    def test_remove_missing(self):
        """Test removing missing values."""
        df = pd.DataFrame({
            'a': [1, 2, np.nan, 4],
            'b': [5, np.nan, 7, 8]
        })

        df_clean = DataCleaner.remove_missing(df)

        # Should keep only rows without NaN
        assert len(df_clean) == 2

    def test_remove_missing_subset(self):
        """Test removing missing values in subset of columns."""
        df = pd.DataFrame({
            'a': [1, 2, np.nan, 4],
            'b': [5, 6, 7, 8]
        })

        df_clean = DataCleaner.remove_missing(df, subset=['b'])

        # Should keep all rows (no NaN in column 'b')
        assert len(df_clean) == 4

    def test_filter_positive(self):
        """Test filtering for positive values."""
        df = pd.DataFrame({
            'x': [-1, 0, 1, 2, 3],
            'y': [1, 2, 3, 4, 5]
        })

        df_clean = DataCleaner.filter_positive(df, ['x', 'y'])

        # Should keep only rows where both x and y are positive
        assert len(df_clean) == 3
        assert df_clean['x'].min() > 0

    def test_remove_outliers_iqr(self):
        """Test outlier removal using IQR method."""
        # Create data with outliers
        np.random.seed(42)
        values = list(np.random.normal(10, 2, 100)) + [100, 200]  # Add outliers

        df = pd.DataFrame({'value': values})

        df_clean = DataCleaner.remove_outliers(df, 'value', method='iqr', threshold=1.5)

        # Should remove extreme outliers
        assert len(df_clean) < len(df)
        assert df_clean['value'].max() < 100

    def test_remove_outliers_zscore(self):
        """Test outlier removal using Z-score method."""
        np.random.seed(42)
        values = list(np.random.normal(10, 2, 100)) + [50]  # Add outlier

        df = pd.DataFrame({'value': values})

        df_clean = DataCleaner.remove_outliers(df, 'value', method='zscore', threshold=3)

        assert len(df_clean) < len(df)

    def test_normalize_zscore(self):
        """Test Z-score normalization."""
        df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })

        df_norm = DataCleaner.normalize(df, ['a', 'b'], method='zscore')

        # Check mean ~ 0 and std ~ 1
        assert abs(df_norm['a'].mean()) < 0.01
        assert abs(df_norm['a'].std() - 1.0) < 0.01
        assert abs(df_norm['b'].mean()) < 0.01

    def test_normalize_minmax(self):
        """Test min-max normalization."""
        df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5]
        })

        df_norm = DataCleaner.normalize(df, ['a'], method='minmax')

        # Check range is [0, 1]
        assert df_norm['a'].min() == 0.0
        assert df_norm['a'].max() == 1.0

    def test_normalize_constant_column(self):
        """Test normalization of constant column (should warn and skip)."""
        df = pd.DataFrame({
            'a': [5, 5, 5, 5]
        })

        # Should not raise error, just warn
        df_norm = DataCleaner.normalize(df, ['a'], method='zscore')

        # Values should remain unchanged
        assert (df_norm['a'] == df['a']).all()


class TestDataLoaderH5ad:
    """Test DataLoader h5ad file loading."""

    @pytest.fixture
    def sample_h5ad_file(self, tmp_path):
        """Create a sample h5ad file for testing."""
        try:
            import anndata
            import scipy.sparse as sp
        except ImportError:
            pytest.skip("anndata not installed")

        # Create sample single-cell data
        np.random.seed(42)
        n_cells = 100
        n_genes = 50

        # Expression matrix (sparse)
        X = sp.random(n_cells, n_genes, density=0.3, format='csr')

        # Cell metadata
        obs = pd.DataFrame({
            'cell_type': np.random.choice(['T-cell', 'B-cell', 'Monocyte'], n_cells),
            'sample_id': np.random.choice(['sample_1', 'sample_2'], n_cells),
            'n_counts': np.random.randint(1000, 10000, n_cells)
        }, index=[f'cell_{i}' for i in range(n_cells)])

        # Gene metadata
        var = pd.DataFrame({
            'gene_symbol': [f'Gene{i}' for i in range(n_genes)],
            'highly_variable': np.random.choice([True, False], n_genes)
        }, index=[f'gene_{i}' for i in range(n_genes)])

        # Create AnnData object
        adata = anndata.AnnData(X=X, obs=obs, var=var)

        # Save to file
        h5ad_path = tmp_path / 'sample.h5ad'
        adata.write_h5ad(h5ad_path)

        return h5ad_path

    def test_load_h5ad_to_dataframe(self, sample_h5ad_file):
        """Test loading h5ad file as DataFrame."""
        df = DataLoader.load_h5ad(sample_h5ad_file, to_dataframe=True)

        # Check dimensions
        assert len(df) == 100  # n_cells
        assert 'gene_0' in df.columns  # Gene columns
        assert 'obs_cell_type' in df.columns  # Metadata columns

    def test_load_h5ad_as_anndata(self, sample_h5ad_file):
        """Test loading h5ad file as AnnData object."""
        try:
            import anndata
        except ImportError:
            pytest.skip("anndata not installed")

        adata = DataLoader.load_h5ad(sample_h5ad_file, to_dataframe=False)

        assert isinstance(adata, anndata.AnnData)
        assert adata.n_obs == 100
        assert adata.n_vars == 50

    def test_load_h5ad_with_obs_columns(self, sample_h5ad_file):
        """Test loading h5ad with specific obs columns."""
        df = DataLoader.load_h5ad(
            sample_h5ad_file,
            to_dataframe=True,
            obs_columns=['cell_type']
        )

        assert 'obs_cell_type' in df.columns
        assert 'obs_sample_id' not in df.columns

    def test_load_h5ad_file_not_found(self):
        """Test h5ad loading with non-existent file."""
        with pytest.raises(FileNotFoundError):
            DataLoader.load_h5ad('/nonexistent/file.h5ad')

    def test_load_h5ad_import_error(self, monkeypatch):
        """Test h5ad loading when anndata not installed."""
        import sys

        # Mock anndata not being installed
        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def mock_import(name, *args, **kwargs):
            if name == 'anndata':
                raise ImportError("No module named 'anndata'")
            return original_import(name, *args, **kwargs)

        with pytest.raises(ImportError, match="anndata package required"):
            # Temporarily hide anndata
            anndata_module = sys.modules.get('anndata')
            sys.modules['anndata'] = None
            try:
                # Force reimport
                import importlib
                from kosmos.execution import data_analysis
                importlib.reload(data_analysis)
                data_analysis.DataLoader.load_h5ad('/fake/path.h5ad')
            finally:
                if anndata_module:
                    sys.modules['anndata'] = anndata_module
                else:
                    sys.modules.pop('anndata', None)


class TestDataLoaderParquet:
    """Test DataLoader parquet file loading."""

    @pytest.fixture
    def sample_parquet_file(self, tmp_path):
        """Create a sample parquet file for testing."""
        try:
            import pyarrow.parquet as pq
            import pyarrow as pa
        except ImportError:
            pytest.skip("pyarrow not installed")

        # Create sample data
        np.random.seed(42)
        df = pd.DataFrame({
            'id': range(1000),
            'category': np.random.choice(['A', 'B', 'C'], 1000),
            'value': np.random.randn(1000),
            'timestamp': pd.date_range('2024-01-01', periods=1000, freq='h')
        })

        # Save to parquet
        parquet_path = tmp_path / 'sample.parquet'
        df.to_parquet(parquet_path)

        return parquet_path

    def test_load_parquet_basic(self, sample_parquet_file):
        """Test basic parquet loading."""
        df = DataLoader.load_parquet(sample_parquet_file)

        assert len(df) == 1000
        assert list(df.columns) == ['id', 'category', 'value', 'timestamp']

    def test_load_parquet_with_columns(self, sample_parquet_file):
        """Test parquet loading with specific columns."""
        df = DataLoader.load_parquet(sample_parquet_file, columns=['id', 'value'])

        assert len(df) == 1000
        assert list(df.columns) == ['id', 'value']
        assert 'category' not in df.columns

    def test_load_parquet_file_not_found(self):
        """Test parquet loading with non-existent file."""
        # The loader checks pyarrow first; CI may resolve a pyarrow that needs NumPy 2
        pytest.importorskip("pyarrow.parquet")
        with pytest.raises(FileNotFoundError):
            DataLoader.load_parquet('/nonexistent/file.parquet')

    def test_load_parquet_via_autodetect(self, sample_parquet_file):
        """Test parquet loading through load_data auto-detection."""
        df = DataLoader.load_data(sample_parquet_file)

        assert len(df) == 1000
        assert 'id' in df.columns


class TestDataLoaderAutoDetect:
    """Test DataLoader.load_data auto-detection for new formats."""

    def test_autodetect_h5ad(self, tmp_path):
        """Test auto-detection of h5ad files."""
        try:
            import anndata
        except ImportError:
            pytest.skip("anndata not installed")

        # Create minimal h5ad file
        np.random.seed(42)
        adata = anndata.AnnData(X=np.random.randn(10, 5))
        h5ad_path = tmp_path / 'test.h5ad'
        adata.write_h5ad(h5ad_path)

        # Load via auto-detection
        df = DataLoader.load_data(h5ad_path)

        assert len(df) == 10

    def test_autodetect_parquet(self, tmp_path):
        """Test auto-detection of parquet files."""
        try:
            import pyarrow
        except ImportError:
            pytest.skip("pyarrow not installed")

        # Create minimal parquet file
        df_orig = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        parquet_path = tmp_path / 'test.parquet'
        df_orig.to_parquet(parquet_path)

        # Load via auto-detection
        df = DataLoader.load_data(parquet_path)

        assert len(df) == 3
        assert list(df.columns) == ['a', 'b']


class TestDataAnalyzerIntegration:
    """Integration tests for complete analysis workflows."""

    def test_complete_ttest_workflow(self):
        """Test complete t-test workflow with data loading and cleaning."""
        # Create test data
        np.random.seed(42)
        df = pd.DataFrame({
            'treatment': ['control'] * 50 + ['drug'] * 50,
            'response': np.concatenate([
                np.random.normal(100, 15, 50),
                np.random.normal(120, 15, 50)
            ])
        })

        # Add some NaN values
        df.loc[0, 'response'] = np.nan
        df.loc[50, 'response'] = np.nan

        # Clean data
        df_clean = DataCleaner.remove_missing(df)

        # Analyze
        result = DataAnalyzer.ttest_comparison(
            df_clean, 'treatment', 'response', ('drug', 'control')
        )

        # Should detect significant difference
        assert result['p_value'] < 0.05
        assert result['group1_mean'] > result['group2_mean']

    def test_complete_correlation_workflow(self):
        """Test complete correlation workflow."""
        # Create test data
        np.random.seed(42)
        df = pd.DataFrame({
            'hours_studied': np.random.uniform(0, 10, 100),
            'exam_score': np.random.uniform(50, 100, 100)
        })

        # Add correlation
        df['exam_score'] = 50 + 5 * df['hours_studied'] + np.random.randn(100) * 5

        # Remove outliers
        df_clean = DataCleaner.remove_outliers(df, 'exam_score', method='zscore', threshold=3)

        # Analyze correlation
        result = DataAnalyzer.correlation_analysis(
            df_clean, 'hours_studied', 'exam_score'
        )

        # Should detect positive correlation
        assert result['correlation'] > 0.5
        assert result['p_value'] < 0.05

    def test_complete_power_law_workflow(self):
        """Test complete power law analysis workflow (Figure 4 pattern)."""
        # Simulate connectome data (neuron length vs synapse count)
        np.random.seed(42)
        neuron_length = np.random.uniform(10, 1000, 200)
        synapse_count = 0.5 * (neuron_length ** 1.2) * np.exp(np.random.randn(200) * 0.15)

        df = pd.DataFrame({
            'Length': neuron_length,
            'Synapses': synapse_count
        })

        # Clean: remove non-positive values (kosmos-figures Figure 4 pattern)
        df_clean = DataCleaner.filter_positive(df, ['Length', 'Synapses'])

        # Perform log-log scaling analysis
        result = DataAnalyzer.log_log_scaling_analysis(df_clean, 'Length', 'Synapses')

        # Should detect power law with exponent ~ 1.2
        assert 1.0 < result['power_law_exponent'] < 1.4
        assert result['spearman_rho'] > 0.8
        assert result['p_value'] < 0.05
