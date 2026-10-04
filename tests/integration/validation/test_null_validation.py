"""
Integration tests for null model validation (Issue #70).

Tests real-world validation scenarios and integration with ScholarEval.
"""

import pytest
import numpy as np
import pandas as pd

from kosmos.validation import (
    NullModelValidator,
    NullModelResult,
    ScholarEvalValidator,
    ScholarEvalScore
)
from kosmos.world_model.artifacts import Finding


class TestRealCorrelationValidation:
    """Test validation of real correlation findings."""

    def test_validate_strong_correlation(self):
        """Test that a strong correlation passes validation."""
        # Create finding with strong correlation statistics
        # Use 'statistic' key for the Fisher z-transformed value (more appropriate)
        # For r=0.85, n=100, Fisher z = 0.5 * ln((1+r)/(1-r)) * sqrt(n-3) ~ 12.4
        finding = {
            'summary': 'X and Y are strongly positively correlated',
            'statistics': {
                'test_type': 'correlation',
                'statistic': 12.4,  # Fisher z-transformed statistic
                'correlation': 0.85,
                'r_squared': 0.72,
                'p_value': 0.0001,
                'sample_size': 100
            },
            'methods': 'Pearson correlation analysis',
            'interpretation': 'Strong positive relationship between X and Y'
        }

        validator = NullModelValidator(n_permutations=500, random_seed=42)
        result = validator.validate_finding(finding)

        # Strong correlation should pass null test
        assert result.passes_null_test == True
        assert result.null_percentile > 90

    def test_detect_spurious_correlation(self):
        """Test that a spurious correlation is detected."""
        # Create finding with weak/spurious statistics
        finding = {
            'summary': 'A and B might be correlated',
            'statistics': {
                'test_type': 'correlation',
                'correlation': 0.15,  # Weak
                'p_value': 0.3,  # Not significant
                'sample_size': 50
            }
        }

        validator = NullModelValidator(n_permutations=500, random_seed=42)
        result = validator.validate_finding(finding)

        # Spurious correlation should fail or persist in noise
        assert result.permutation_p_value > 0.05 or result.persists_in_noise == True


class TestRealTTestValidation:
    """Test validation of t-test findings."""

    def test_validate_significant_group_difference(self):
        """Test that significant group difference passes."""
        finding = {
            'summary': 'Treatment group shows significantly higher values',
            'statistics': {
                'test_type': 't_test',
                't_statistic': 4.5,
                'p_value': 0.0001,
                'effect_size': 0.9,  # Large effect
                'degrees_of_freedom': 98,
                'sample_size': 100
            },
            'methods': 'Independent samples t-test'
        }

        validator = NullModelValidator(n_permutations=500, random_seed=42)
        result = validator.validate_finding(finding)

        assert result.passes_null_test == True
        assert result.is_valid == True

    def test_detect_false_positive_ttest(self):
        """Test that likely false positive t-test is flagged."""
        finding = {
            'summary': 'Groups might differ slightly',
            'statistics': {
                'test_type': 't_test',
                't_statistic': 1.2,  # Small
                'p_value': 0.2,  # Not significant
                'effect_size': 0.1,  # Small effect
                'degrees_of_freedom': 48
            }
        }

        validator = NullModelValidator(n_permutations=500, random_seed=42)
        result = validator.validate_finding(finding)

        # Should fail validation
        assert result.is_valid == False


class TestScholarEvalIntegration:
    """Test integration with ScholarEval validator."""

    def test_scholar_eval_includes_null_model(self):
        """Test that ScholarEval includes null model results."""
        finding = {
            'summary': 'Gene expression differs between groups',
            'statistics': {
                'test_type': 't_test',
                'statistic': 3.5,
                'p_value': 0.001,
                'effect_size': 0.7,
                'degrees_of_freedom': 50
            },
            'methods': 'Differential expression analysis',
            'interpretation': 'Treatment significantly affects gene expression'
        }

        validator = ScholarEvalValidator(allow_mock=True)  # No LLM client, uses mock
        score = validator.evaluate_finding(finding)

        # Should include null model result
        assert score.null_model_result is not None
        assert 'permutation_p_value' in score.null_model_result
        assert 'persists_in_noise' in score.null_model_result

    def test_scholar_eval_statistical_validity_score(self):
        """Test that statistical_validity score is set."""
        finding = {
            'summary': 'Significant finding',
            'statistics': {
                'test_type': 't_test',
                'statistic': 4.0,
                'p_value': 0.0001,
                'degrees_of_freedom': 100
            }
        }

        validator = ScholarEvalValidator(allow_mock=True)
        score = validator.evaluate_finding(finding)

        # Should have statistical_validity
        assert score.statistical_validity is not None
        assert 0 <= score.statistical_validity <= 1

    def test_scholar_eval_penalizes_noise_persistence(self):
        """Test that findings persisting in noise are penalized."""
        # Create finding with weak statistics that will persist in noise
        weak_finding = {
            'summary': 'Weak finding',
            'statistics': {
                'test_type': 't_test',
                'statistic': 0.3,  # Very weak
                'p_value': 0.7,
                'degrees_of_freedom': 50
            }
        }

        # Create finding with strong statistics
        strong_finding = {
            'summary': 'Strong finding',
            'statistics': {
                'test_type': 't_test',
                'statistic': 5.0,  # Strong
                'p_value': 0.00001,
                'degrees_of_freedom': 50
            }
        }

        validator = ScholarEvalValidator(allow_mock=True)
        weak_score = validator.evaluate_finding(weak_finding)
        strong_score = validator.evaluate_finding(strong_finding)

        # Weak finding should have lower or equal overall score
        # (due to null model penalty if it persists in noise)
        if weak_score.null_model_result and weak_score.null_model_result.get('persists_in_noise'):
            assert weak_score.overall_score <= strong_score.overall_score


class TestFindingIntegration:
    """Test integration with Finding dataclass."""

    def test_finding_stores_null_model_result(self):
        """Test that Finding can store null model results."""
        null_result = {
            'observed_statistic': 3.5,
            'permutation_p_value': 0.01,
            'null_percentile': 99.0,
            'passes_null_test': True,
            'persists_in_noise': False,
            'n_permutations': 1000,
            'shuffle_method': 'label',
            'alpha': 0.05
        }

        finding = Finding(
            finding_id='f001',
            cycle=1,
            task_id=1,
            summary='Test finding',
            statistics={'p_value': 0.001},
            null_model_result=null_result
        )

        assert finding.null_model_result is not None
        assert finding.null_model_result['permutation_p_value'] == 0.01
        assert finding.null_model_result['passes_null_test'] == True

    def test_finding_serialization_with_null_model(self):
        """Test that Finding serializes null model results correctly."""
        null_result = {
            'observed_statistic': 2.5,
            'permutation_p_value': 0.03,
            'passes_null_test': True,
            'persists_in_noise': False
        }

        finding = Finding(
            finding_id='f002',
            cycle=1,
            task_id=2,
            summary='Serializable finding',
            statistics={'effect_size': 0.8},
            null_model_result=null_result
        )

        # Serialize and deserialize
        data = finding.to_dict()
        restored = Finding.from_dict(data)

        assert restored.null_model_result is not None
        assert restored.null_model_result['permutation_p_value'] == 0.03


class TestFullPermutationValidation:
    """Test full permutation testing with real data and analysis function."""

    def test_full_permutation_with_correlation(self):
        """Test full permutation test for correlation analysis."""
        # Create correlated data
        np.random.seed(42)
        n = 100
        x = np.random.normal(0, 1, n)
        y = 0.8 * x + np.random.normal(0, 0.3, n)  # Strong correlation
        data = pd.DataFrame({'x': x, 'y': y})

        def correlation_analysis(df):
            from scipy import stats
            r, p = stats.pearsonr(df['x'], df['y'])
            return {'correlation': r, 'p_value': p}

        # Run the original analysis
        original_result = correlation_analysis(data)

        finding = {
            'statistics': {
                'test_type': 'correlation',
                'correlation': original_result['correlation'],
                'p_value': original_result['p_value']
            }
        }

        validator = NullModelValidator(n_permutations=200, random_seed=42)
        result = validator.validate_finding(
            finding,
            data=data,
            analysis_func=correlation_analysis
        )

        # Real correlation should pass null test
        assert result.passes_null_test == True
        assert result.n_permutations >= 100  # Some might fail

    def test_full_permutation_detects_noise(self):
        """Test that full permutation correctly identifies noise."""
        # Create uncorrelated data
        np.random.seed(42)
        n = 100
        x = np.random.normal(0, 1, n)
        y = np.random.normal(0, 1, n)  # Independent
        data = pd.DataFrame({'x': x, 'y': y})

        def correlation_analysis(df):
            from scipy import stats
            r, p = stats.pearsonr(df['x'], df['y'])
            return {'correlation': abs(r), 'p_value': p}

        original_result = correlation_analysis(data)

        finding = {
            'statistics': {
                'test_type': 'correlation',
                'correlation': original_result['correlation'],
                'p_value': original_result['p_value']
            }
        }

        validator = NullModelValidator(n_permutations=200, random_seed=42)
        result = validator.validate_finding(
            finding,
            data=data,
            analysis_func=correlation_analysis
        )

        # Noise should either fail null test or persist in noise
        assert result.passes_null_test == False or result.persists_in_noise == True


class TestBatchValidation:
    """Test batch validation of multiple findings."""

    def test_batch_validation_statistics(self):
        """Test aggregate statistics from batch validation."""
        findings = [
            # Strong findings
            {'statistics': {'test_type': 't_test', 'statistic': 5.0, 'p_value': 0.0001, 'degrees_of_freedom': 50}},
            {'statistics': {'test_type': 't_test', 'statistic': 4.5, 'p_value': 0.0001, 'degrees_of_freedom': 50}},
            {'statistics': {'test_type': 't_test', 'statistic': 4.0, 'p_value': 0.001, 'degrees_of_freedom': 50}},
            # Weak findings
            {'statistics': {'test_type': 't_test', 'statistic': 1.0, 'p_value': 0.3, 'degrees_of_freedom': 50}},
            {'statistics': {'test_type': 't_test', 'statistic': 0.5, 'p_value': 0.5, 'degrees_of_freedom': 50}},
        ]

        validator = NullModelValidator(n_permutations=100, random_seed=42)
        results = validator.batch_validate(findings)
        stats = validator.get_validation_statistics(results)

        assert stats['count'] == 5
        assert 0 < stats['valid_rate'] < 1  # Some valid, some not
        assert stats['mean_p_value'] > 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_statistics(self):
        """Test handling of finding with empty statistics."""
        finding = {'summary': 'No stats', 'statistics': {}}

        validator = NullModelValidator()
        result = validator.validate_finding(finding)

        assert result.is_valid == False
        assert len(result.warnings) > 0

    def test_missing_statistics_field(self):
        """Test handling of finding without statistics field."""
        finding = {'summary': 'No statistics field at all'}

        validator = NullModelValidator()
        result = validator.validate_finding(finding)

        assert result.is_valid == False

    def test_unusual_test_type(self):
        """Test handling of unusual test type."""
        finding = {
            'statistics': {
                'test_type': 'exotic_bayesian_nonparametric_test',
                'statistic': 2.5,
                'p_value': 0.01
            }
        }

        validator = NullModelValidator(n_permutations=100)
        result = validator.validate_finding(finding)

        # Should still work with default handling
        assert result.n_permutations > 0
        assert result.shuffle_method == 'column'  # Default

    def test_very_small_sample_size(self):
        """Test handling of very small sample sizes."""
        finding = {
            'statistics': {
                'test_type': 't_test',
                'statistic': 2.0,
                'degrees_of_freedom': 3  # Very small
            }
        }

        validator = NullModelValidator(n_permutations=100)
        result = validator.validate_finding(finding)

        # Should still produce a result
        assert result.n_permutations > 0


class TestValidationResultQuality:
    """Test quality and correctness of validation results."""

    def test_permutation_pvalue_in_valid_range(self):
        """Test that permutation p-value is always in valid range."""
        findings = [
            {'statistics': {'test_type': 't_test', 'statistic': 5.0, 'degrees_of_freedom': 50}},
            {'statistics': {'test_type': 't_test', 'statistic': 0.1, 'degrees_of_freedom': 50}},
            {'statistics': {'correlation': 0.9}},
            {'statistics': {'correlation': 0.05}},
        ]

        validator = NullModelValidator(n_permutations=100)
        for finding in findings:
            result = validator.validate_finding(finding)
            assert 0 <= result.permutation_p_value <= 1

    def test_percentile_in_valid_range(self):
        """Test that percentile is always in valid range."""
        findings = [
            {'statistics': {'test_type': 't_test', 'statistic': 10.0, 'degrees_of_freedom': 50}},
            {'statistics': {'test_type': 't_test', 'statistic': 0.01, 'degrees_of_freedom': 50}},
        ]

        validator = NullModelValidator(n_permutations=100)
        for finding in findings:
            result = validator.validate_finding(finding)
            assert 0 <= result.null_percentile <= 100

    def test_consistency_of_results(self):
        """Test that results are consistent with same seed."""
        finding = {
            'statistics': {
                'test_type': 't_test',
                'statistic': 3.0,
                'degrees_of_freedom': 50
            }
        }

        validator1 = NullModelValidator(n_permutations=100, random_seed=42)
        validator2 = NullModelValidator(n_permutations=100, random_seed=42)

        result1 = validator1.validate_finding(finding)
        result2 = validator2.validate_finding(finding)

        assert result1.permutation_p_value == result2.permutation_p_value
        assert result1.null_percentile == result2.null_percentile
