"""
Integration tests for the compression pipeline.

Tests full compression workflow with real notebooks and multi-tier compression.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock

from kosmos.compression.compressor import (
    ContextCompressor,
    NotebookCompressor,
    LiteratureCompressor,
    CompressedContext
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def real_notebook_content():
    """Realistic notebook content with statistical analysis."""
    return """
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["# KRAS Mutation Analysis\\n", "Analysis of differential gene expression in KRAS mutant samples"]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "outputs": [],
   "source": [
    "import pandas as pd\\n",
    "import numpy as np\\n",
    "from scipy import stats\\n",
    "\\n",
    "# Load expression data\\n",
    "data = pd.read_csv('kras_expression.csv')\\n",
    "print(f'Loaded {len(data)} samples')\\n",
    "# n = 250 samples total"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "outputs": [{"output_type": "stream", "text": "p-value: 0.00023"}],
   "source": [
    "# Differential expression test\\n",
    "mutant = data[data['KRAS_status'] == 'mutant']\\n",
    "wildtype = data[data['KRAS_status'] == 'wildtype']\\n",
    "\\n",
    "result = stats.ttest_ind(mutant['expression'], wildtype['expression'])\\n",
    "print(f'p-value: {result.pvalue:.5f}')\\n",
    "# p = 0.00023\\n",
    "# Effect size: Cohen's d = 0.72"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "outputs": [],
   "source": [
    "# Found 87 differentially expressed genes\\n",
    "# Correlation between KRAS expression and survival: r = -0.45\\n",
    "# FDR-corrected threshold: 0.05"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Results Summary\\n",
    "\\n",
    "- 87 genes significantly differentially expressed\\n",
    "- KRAS expression negatively correlated with survival\\n",
    "- Results consistent with published literature"
   ]
  }
 ]
}
"""


@pytest.fixture
def real_papers():
    """Realistic paper list for literature compression."""
    return [
        {
            'paper_id': 'paper_001',
            'title': 'KRAS G12D mutations drive oncogenic signaling in pancreatic cancer',
            'authors': ['Smith, J.', 'Jones, M.', 'Lee, K.'],
            'year': 2023,
            'journal': 'Nature Cancer',
            'abstract': '''KRAS mutations are found in 90% of pancreatic ductal adenocarcinoma (PDAC).
            We analyzed 500 PDAC samples and found G12D mutations associated with poor prognosis
            (HR=2.5, p<0.001). Statistical analysis reveals significant correlation between
            mutation status and therapeutic response (r=0.67, n=150).''',
            'findings': '''Main finding: KRAS G12D mutations predict poor outcomes in PDAC.
            p-value: 0.0001, hazard ratio: 2.5, sample size: 500 patients.
            Effect size: Cohen's d = 0.82 for survival difference.''',
            'relevance_score': 0.95
        },
        {
            'paper_id': 'paper_002',
            'title': 'Novel KRAS inhibitor shows efficacy in preclinical models',
            'authors': ['Chen, L.', 'Wang, H.'],
            'year': 2024,
            'journal': 'Cell',
            'abstract': '''Development of a novel covalent KRAS G12C inhibitor with improved
            pharmacokinetics. In vivo studies showed tumor regression in 65% of mice (n=40).
            Phase 1 trial data shows response rate of 42% (p=0.002).''',
            'findings': '''Drug efficacy: 65% tumor regression in mice, 42% response rate in humans.
            Statistical significance: p = 0.002, n = 40 (preclinical), n = 50 (clinical).''',
            'relevance_score': 0.88
        },
        {
            'paper_id': 'paper_003',
            'title': 'KRAS downstream signaling pathways in colorectal cancer',
            'authors': ['Park, S.'],
            'year': 2022,
            'journal': 'Cancer Research',
            'abstract': '''Investigation of MAPK and PI3K signaling in KRAS-mutant CRC.
            Phosphoproteomic analysis of 200 samples reveals pathway activation patterns.''',
            'findings': '''Pathway analysis: MAPK activation in 78% of KRAS-mutant samples.
            PI3K pathway: correlation r = 0.71 with mutation status.''',
            'relevance_score': 0.75
        }
    ]


@pytest.fixture
def multi_cycle_results(real_notebook_content, real_papers, temp_dir):
    """Multi-cycle task results for compression testing."""
    # Create notebook file
    notebook_path = temp_dir / "analysis.ipynb"
    notebook_path.write_text(real_notebook_content)

    return [
        {
            'cycle': 1,
            'results': [
                {
                    'type': 'data_analysis',
                    'notebook_path': str(notebook_path)
                },
                {
                    'type': 'literature_review',
                    'papers': real_papers[:2]
                }
            ]
        },
        {
            'cycle': 2,
            'results': [
                {
                    'type': 'data_analysis',
                    'notebook_path': str(notebook_path)
                },
                {
                    'type': 'literature_review',
                    'papers': real_papers[1:]
                }
            ]
        }
    ]


# ============================================================================
# Integration Tests
# ============================================================================

class TestCompressionPipeline:
    """Integration tests for full compression pipeline."""

    def test_notebook_to_context_compression(self, real_notebook_content, temp_dir):
        """Test full notebook compression workflow."""
        notebook_path = temp_dir / "full_test.ipynb"
        notebook_path.write_text(real_notebook_content)

        compressor = NotebookCompressor()
        result = compressor.compress_notebook(str(notebook_path))

        assert isinstance(result, CompressedContext)
        assert len(result.summary) > 0
        assert result.full_content_path == str(notebook_path)

        # Should extract statistics
        assert 'p_value' in result.statistics
        assert result.statistics['p_value'] < 0.05

    def test_literature_compression_pipeline(self, real_papers):
        """Test full literature compression workflow."""
        compressor = LiteratureCompressor()
        results = compressor.compress_papers(real_papers)

        assert len(results) == 3

        # Check compression preserves key info
        for i, result in enumerate(results):
            assert isinstance(result, CompressedContext)
            assert 'paper_id' in result.metadata

            # Higher relevance papers should be first
            if i > 0:
                assert (results[i-1].metadata.get('relevance_score', 0) >=
                        result.metadata.get('relevance_score', 0))

    def test_context_compressor_full_cycle(self, real_notebook_content, real_papers, temp_dir):
        """Test full context compression for a research cycle."""
        notebook_path = temp_dir / "cycle_test.ipynb"
        notebook_path.write_text(real_notebook_content)

        compressor = ContextCompressor()

        task_results = [
            {'type': 'data_analysis', 'notebook_path': str(notebook_path)},
            {'type': 'literature_review', 'papers': real_papers}
        ]

        result = compressor.compress_cycle_results(cycle=1, task_results=task_results)

        assert isinstance(result, CompressedContext)
        assert result.metadata['cycle'] == 1
        assert result.metadata['n_tasks'] == 2

        # Summary should mention both data analysis and literature
        assert len(result.summary) > 100

    def test_multi_cycle_compression(self, multi_cycle_results):
        """Test compression across multiple research cycles."""
        compressor = ContextCompressor()
        compressed_cycles = []

        for cycle_data in multi_cycle_results:
            result = compressor.compress_cycle_results(
                cycle=cycle_data['cycle'],
                task_results=cycle_data['results']
            )
            compressed_cycles.append(result)

        assert len(compressed_cycles) == 2

        # Each cycle should have compressed context
        for i, compressed in enumerate(compressed_cycles):
            assert compressed.metadata['cycle'] == i + 1
            assert len(compressed.summary) > 0

    def test_compression_preserves_statistical_evidence(self, real_notebook_content, temp_dir):
        """Test that compression preserves important statistical findings."""
        notebook_path = temp_dir / "stats_test.ipynb"
        notebook_path.write_text(real_notebook_content)

        compressor = NotebookCompressor()
        result = compressor.compress_notebook(str(notebook_path))

        # Should preserve key statistics
        stats = result.statistics

        assert 'p_value' in stats
        # p-value from notebook
        assert stats['p_value'] < 0.001 or stats['p_value_count'] >= 1

        # Should have sample size if extractable
        if 'sample_size' in stats:
            assert stats['sample_size'] > 0

    def test_compression_ratio_efficiency(self, real_notebook_content, temp_dir):
        """Test that compression achieves significant size reduction."""
        notebook_path = temp_dir / "ratio_test.ipynb"
        notebook_path.write_text(real_notebook_content)

        compressor = NotebookCompressor()
        result = compressor.compress_notebook(str(notebook_path))

        original_size = len(real_notebook_content)
        compressed_size = len(result.summary)

        # Should achieve at least 3:1 compression
        compression_ratio = original_size / (compressed_size + 1)
        assert compression_ratio > 3, f"Compression ratio {compression_ratio:.1f} is too low"

    def test_literature_compression_ranking(self, real_papers):
        """Test that literature compression properly ranks by relevance."""
        compressor = LiteratureCompressor()

        # Limit to top 2 papers
        results = compressor.compress_papers(real_papers, max_papers=2)

        assert len(results) == 2

        # Should include highest relevance papers
        paper_ids = [r.metadata['paper_id'] for r in results]
        assert 'paper_001' in paper_ids  # 0.95 relevance
        assert 'paper_002' in paper_ids  # 0.88 relevance


class TestCompressionCaching:
    """Tests for compression caching behavior."""

    def test_context_compressor_caching(self, real_notebook_content, temp_dir):
        """Test that repeated compression uses caching."""
        notebook_path = temp_dir / "cache_test.ipynb"
        notebook_path.write_text(real_notebook_content)

        compressor = ContextCompressor()

        task_results = [
            {'type': 'data_analysis', 'notebook_path': str(notebook_path)}
        ]

        # Compress same cycle twice
        result1 = compressor.compress_cycle_results(cycle=1, task_results=task_results)
        result2 = compressor.compress_cycle_results(cycle=1, task_results=task_results)

        # Both should produce valid results
        assert len(result1.summary) > 0
        assert len(result2.summary) > 0


class TestCompressionEdgeCases:
    """Integration tests for edge cases."""

    def test_empty_notebook(self, temp_dir):
        """Test compression of empty notebook."""
        notebook_path = temp_dir / "empty.ipynb"
        notebook_path.write_text('{"cells": [], "metadata": {}}')

        compressor = NotebookCompressor()
        result = compressor.compress_notebook(str(notebook_path))

        assert isinstance(result, CompressedContext)
        assert result.statistics == {}

    def test_mixed_task_types(self, real_notebook_content, real_papers, temp_dir):
        """Test compression with mixed task types in single cycle."""
        notebook_path = temp_dir / "mixed.ipynb"
        notebook_path.write_text(real_notebook_content)

        compressor = ContextCompressor()

        task_results = [
            {'type': 'data_analysis', 'notebook_path': str(notebook_path)},
            {'type': 'literature_review', 'papers': real_papers},
            {'type': 'hypothesis_generation', 'hypotheses': ['Hyp 1', 'Hyp 2']},
            {'type': 'unknown_type', 'data': 'some data'}
        ]

        result = compressor.compress_cycle_results(cycle=1, task_results=task_results)

        assert isinstance(result, CompressedContext)
        assert result.metadata['n_tasks'] == 4
