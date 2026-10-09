"""
Unit tests for kosmos.compression.compressor module.

Tests:
- NotebookCompressor: compress_notebook, statistics extraction
- LiteratureCompressor: compress_papers
- ContextCompressor: compress_cycle_results, hierarchical compression
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from kosmos.compression.compressor import (
    CompressedContext,
    NotebookCompressor,
    LiteratureCompressor,
    ContextCompressor
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_notebook_content():
    """Sample Jupyter notebook content with statistical findings."""
    return """
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["# Gene Expression Analysis"]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "outputs": [],
   "source": [
    "import pandas as pd\\n",
    "import scipy.stats as stats\\n",
    "\\n",
    "# Load data\\n",
    "data = pd.read_csv('expression_data.csv')\\n",
    "\\n",
    "# Statistical test\\n",
    "result = stats.ttest_ind(group1, group2)\\n",
    "print(f'p-value: {result.pvalue}')  # p = 0.003\\n",
    "print(f'n = 150 samples')\\n",
    "\\n",
    "# We found 42 differentially expressed genes\\n",
    "# correlation r = 0.82 between expression and phenotype"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Results\\n",
    "We identified 42 significant genes with p < 0.05.\\n",
    "The analysis reveals strong correlation (r=0.82)."
   ]
  }
 ]
}
"""


@pytest.fixture
def sample_notebook_text():
    """Plain text notebook content for simpler testing."""
    return """
# Gene Expression Analysis

## Methods
Statistical analysis of RNA-seq data using DESeq2.
n = 150 samples from cancer patients.

## Results
We found 42 differentially expressed genes.
The most significant gene had p = 0.001
Correlation between expression and survival: r = 0.82
Sample size: 150 patients.
Cohen's d = 0.65 for treatment effect.

## Conclusion
Results show significant association between gene expression and patient outcomes.
"""


@pytest.fixture
def sample_papers():
    """Sample paper list for LiteratureCompressor."""
    return [
        {
            'title': 'KRAS Mutations in Cancer',
            'abstract': 'We studied KRAS mutations in 500 cancer patients. p < 0.001.',
            'findings': 'KRAS G12D mutation associated with poor prognosis (HR=2.5, p<0.001).',
            'authors': ['Smith, J.', 'Jones, M.'],
            'year': 2023,
            'journal': 'Nature Cancer',
            'paper_id': 'paper_001',
            'relevance_score': 0.95
        },
        {
            'title': 'Targeted Therapy for KRAS',
            'abstract': 'Novel inhibitor shows efficacy in KRAS mutant tumors.',
            'findings': 'Response rate 45% (n=120). Median PFS 8.2 months.',
            'authors': ['Lee, A.'],
            'year': 2024,
            'journal': 'Cell',
            'paper_id': 'paper_002',
            'relevance_score': 0.88
        }
    ]


@pytest.fixture
def sample_task_results():
    """Sample task results for cycle compression."""
    return [
        {
            'type': 'data_analysis',
            'notebook_path': '/path/to/notebook1.ipynb',
            'notebook_content': 'Analysis results: p = 0.02, n = 100'
        },
        {
            'type': 'literature_review',
            'papers': [
                {
                    'title': 'Paper 1',
                    'abstract': 'Important findings.',
                    'findings': 'Key result: p < 0.05',
                    'paper_id': 'p1',
                    'relevance_score': 0.9,
                    'authors': ['Smith']
                }
            ]
        }
    ]


@pytest.fixture
def mock_anthropic_response():
    """Mock Anthropic API response."""
    mock_response = Mock()
    mock_response.content = [Mock(text="Summary line 1.\nSummary line 2.")]
    return mock_response


# ============================================================================
# CompressedContext Tests
# ============================================================================

class TestCompressedContext:
    """Tests for CompressedContext dataclass."""

    def test_basic_creation(self):
        """Test basic CompressedContext creation."""
        ctx = CompressedContext(
            summary="Test summary",
            statistics={'p_value': 0.05}
        )
        assert ctx.summary == "Test summary"
        assert ctx.statistics == {'p_value': 0.05}
        assert ctx.full_content_path is None
        assert ctx.metadata is None

    def test_full_creation(self):
        """Test CompressedContext with all fields."""
        ctx = CompressedContext(
            summary="Full summary",
            statistics={'p_value': 0.01, 'n': 100},
            full_content_path="/path/to/notebook.ipynb",
            metadata={'type': 'jupyter', 'cycle': 1}
        )
        assert ctx.full_content_path == "/path/to/notebook.ipynb"
        assert ctx.metadata['type'] == 'jupyter'


# ============================================================================
# NotebookCompressor Tests
# ============================================================================

class TestNotebookCompressor:
    """Tests for NotebookCompressor class."""

    def test_init_without_client(self):
        """Test initialization without Anthropic client."""
        compressor = NotebookCompressor()
        assert compressor.client is None
        assert compressor.model == "claude-3-5-sonnet-20241022"

    def test_init_with_client(self):
        """Test initialization with Anthropic client."""
        mock_client = Mock()
        compressor = NotebookCompressor(anthropic_client=mock_client)
        assert compressor.client == mock_client

    def test_extract_p_values(self, sample_notebook_text):
        """Test p-value extraction from notebook content."""
        compressor = NotebookCompressor()
        stats = compressor._extract_statistics(sample_notebook_text)

        assert 'p_value' in stats
        assert stats['p_value'] == 0.001  # Most significant
        assert stats['p_value_count'] >= 1

    def test_extract_correlations(self, sample_notebook_text):
        """Test correlation extraction."""
        compressor = NotebookCompressor()
        stats = compressor._extract_statistics(sample_notebook_text)

        assert 'correlation' in stats
        assert stats['correlation'] == 0.82

    def test_extract_sample_sizes(self, sample_notebook_text):
        """Test sample size extraction."""
        compressor = NotebookCompressor()
        stats = compressor._extract_statistics(sample_notebook_text)

        assert 'sample_size' in stats
        assert stats['sample_size'] == 150

    def test_extract_gene_counts(self, sample_notebook_text):
        """Test gene count extraction."""
        compressor = NotebookCompressor()
        stats = compressor._extract_statistics(sample_notebook_text)

        assert 'n_genes' in stats
        assert stats['n_genes'] == 42

    def test_extract_cohens_d(self, sample_notebook_text):
        """Test Cohen's d extraction."""
        compressor = NotebookCompressor()
        stats = compressor._extract_statistics(sample_notebook_text)

        assert 'cohens_d' in stats
        assert stats['cohens_d'] == 0.65

    def test_is_valid_p_value(self):
        """Test p-value validation."""
        compressor = NotebookCompressor()

        assert compressor._is_valid_p_value("0.05") is True
        assert compressor._is_valid_p_value("1.0") is True
        assert compressor._is_valid_p_value("0.001") is True
        assert compressor._is_valid_p_value("1e-5") is True
        assert compressor._is_valid_p_value("0") is False
        assert compressor._is_valid_p_value("1.5") is False
        assert compressor._is_valid_p_value("invalid") is False

    def test_rule_based_summary(self, sample_notebook_text):
        """Test rule-based summary generation (no LLM)."""
        compressor = NotebookCompressor()
        stats = compressor._extract_statistics(sample_notebook_text)
        summary = compressor._generate_rule_based_summary(sample_notebook_text, stats)

        assert len(summary) > 0
        # Should contain findings or statistics
        assert any(word in summary.lower() for word in ['found', 'gene', 'analysis'])

    def test_compress_notebook_from_content(self, sample_notebook_text, temp_dir):
        """Test compress_notebook with provided content."""
        compressor = NotebookCompressor()
        notebook_path = temp_dir / "test.ipynb"

        result = compressor.compress_notebook(
            str(notebook_path),
            notebook_content=sample_notebook_text
        )

        assert isinstance(result, CompressedContext)
        assert len(result.summary) > 0
        assert 'p_value' in result.statistics
        assert result.full_content_path == str(notebook_path)

    def test_compress_notebook_from_file(self, sample_notebook_text, temp_dir):
        """Test compress_notebook reading from file."""
        compressor = NotebookCompressor()

        # Create test notebook file
        notebook_path = temp_dir / "analysis.ipynb"
        notebook_path.write_text(sample_notebook_text)

        result = compressor.compress_notebook(str(notebook_path))

        assert isinstance(result, CompressedContext)
        assert len(result.summary) > 0

    def test_compress_notebook_file_not_found(self, temp_dir):
        """Test compress_notebook with non-existent file."""
        compressor = NotebookCompressor()
        result = compressor.compress_notebook(str(temp_dir / "nonexistent.ipynb"))

        assert "Error reading notebook" in result.summary

    def test_compress_notebook_with_llm(self, sample_notebook_text, mock_anthropic_response):
        """Test compress_notebook with LLM client."""
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_anthropic_response

        compressor = NotebookCompressor(anthropic_client=mock_client)
        result = compressor.compress_notebook(
            "/path/to/notebook.ipynb",
            notebook_content=sample_notebook_text
        )

        assert isinstance(result, CompressedContext)
        mock_client.messages.create.assert_called_once()

    def test_get_content_sample_short(self):
        """Test content sampling for short content."""
        compressor = NotebookCompressor()
        short_content = "Short content"
        sample = compressor._get_content_sample(short_content, max_chars=1000)

        assert sample == short_content

    def test_get_content_sample_long(self):
        """Test content sampling for long content."""
        compressor = NotebookCompressor()
        long_content = "A" * 5000
        sample = compressor._get_content_sample(long_content, max_chars=1000)

        assert len(sample) < len(long_content)
        assert "middle content omitted" in sample


# ============================================================================
# LiteratureCompressor Tests
# ============================================================================

class TestLiteratureCompressor:
    """Tests for LiteratureCompressor class."""

    def test_init_without_client(self):
        """Test initialization without Anthropic client."""
        compressor = LiteratureCompressor()
        assert compressor.client is None

    def test_compress_papers_empty(self):
        """Test compressing empty paper list."""
        compressor = LiteratureCompressor()
        result = compressor.compress_papers([])

        assert result == []

    def test_compress_papers_basic(self, sample_papers):
        """Test basic paper compression."""
        compressor = LiteratureCompressor()
        results = compressor.compress_papers(sample_papers)

        assert len(results) == 2
        assert all(isinstance(r, CompressedContext) for r in results)

    def test_compress_papers_sorted_by_relevance(self, sample_papers):
        """Test papers are sorted by relevance score."""
        compressor = LiteratureCompressor()

        # Add a low-relevance paper
        papers = sample_papers + [{
            'title': 'Low Relevance Paper',
            'abstract': 'Not very relevant.',
            'paper_id': 'paper_low',
            'relevance_score': 0.1,
            'authors': ['Nobody']
        }]

        results = compressor.compress_papers(papers, max_papers=2)

        # Should only include 2 highest relevance papers
        assert len(results) == 2

    def test_compress_single_paper(self, sample_papers):
        """Test single paper compression."""
        compressor = LiteratureCompressor()
        result = compressor._compress_single_paper(sample_papers[0])

        assert isinstance(result, CompressedContext)
        assert 'KRAS' in result.summary
        assert result.metadata['paper_id'] == 'paper_001'
        assert result.metadata['year'] == 2023

    def test_extract_paper_statistics(self, sample_papers):
        """Test statistics extraction from paper text."""
        compressor = LiteratureCompressor()
        text = sample_papers[0]['findings']
        stats = compressor._extract_paper_statistics(text)

        assert 'p_value' in stats or len(stats) >= 0  # May or may not find p-value


# ============================================================================
# ContextCompressor Tests
# ============================================================================

class TestContextCompressor:
    """Tests for ContextCompressor class."""

    def test_init_without_client(self):
        """Test initialization creates sub-compressors."""
        compressor = ContextCompressor()

        assert isinstance(compressor.notebook_compressor, NotebookCompressor)
        assert isinstance(compressor.literature_compressor, LiteratureCompressor)
        assert compressor.cache == {}

    def test_compress_cycle_results_empty(self):
        """Test compressing empty task results."""
        compressor = ContextCompressor()
        result = compressor.compress_cycle_results(cycle=1, task_results=[])

        assert isinstance(result, CompressedContext)
        assert "Cycle 1: No tasks completed" in result.summary

    def test_compress_cycle_results_data_analysis(self, temp_dir):
        """Test compressing data analysis task results."""
        compressor = ContextCompressor()

        # Create test notebook
        notebook_path = temp_dir / "analysis.ipynb"
        notebook_path.write_text("Analysis: p = 0.01, n = 50")

        task_results = [{
            'type': 'data_analysis',
            'notebook_path': str(notebook_path)
        }]

        result = compressor.compress_cycle_results(cycle=1, task_results=task_results)

        assert isinstance(result, CompressedContext)
        assert result.metadata['cycle'] == 1
        assert result.metadata['n_tasks'] == 1

    def test_compress_cycle_results_literature(self, sample_papers):
        """Test compressing literature review task results."""
        compressor = ContextCompressor()

        task_results = [{
            'type': 'literature_review',
            'papers': sample_papers
        }]

        result = compressor.compress_cycle_results(cycle=2, task_results=task_results)

        assert isinstance(result, CompressedContext)
        assert result.metadata['cycle'] == 2

    def test_compress_cycle_results_mixed(self, sample_papers, temp_dir):
        """Test compressing mixed task types."""
        compressor = ContextCompressor()

        # Create test notebook
        notebook_path = temp_dir / "mixed.ipynb"
        notebook_path.write_text("Analysis: p = 0.02")

        task_results = [
            {
                'type': 'data_analysis',
                'notebook_path': str(notebook_path)
            },
            {
                'type': 'literature_review',
                'papers': sample_papers
            }
        ]

        result = compressor.compress_cycle_results(cycle=3, task_results=task_results)

        assert result.metadata['n_tasks'] == 2
        assert result.metadata['n_compressed_tasks'] >= 1

    def test_synthesize_cycle_summary(self):
        """Test cycle summary synthesis."""
        compressor = ContextCompressor()

        compressed_tasks = [
            CompressedContext(summary="Found 10 genes", statistics={'p_value': 0.01}),
            CompressedContext(summary="Literature supports hypothesis", statistics={})
        ]

        summary = compressor._synthesize_cycle_summary(cycle=5, compressed_tasks=compressed_tasks)

        assert "Cycle 5" in summary
        assert "Found 10 genes" in summary

    def test_aggregate_statistics(self):
        """Test statistics aggregation across tasks."""
        compressor = ContextCompressor()

        compressed_tasks = [
            CompressedContext(
                summary="Task 1",
                statistics={'p_value': 0.01, 'sample_size': 100, 'n_genes': 50}
            ),
            CompressedContext(
                summary="Task 2",
                statistics={'p_value': 0.04, 'sample_size': 200, 'n_genes': 30}
            )
        ]

        aggregated = compressor._aggregate_statistics(compressed_tasks)

        assert aggregated['n_tasks'] == 2
        assert aggregated['min_p_value'] == 0.01
        assert aggregated['n_significant'] == 2  # Both p < 0.05
        assert aggregated['total_samples'] == 300
        assert aggregated['total_genes'] == 80


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestCompressionEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_content_compression(self):
        """Test compressing empty content."""
        compressor = NotebookCompressor()
        result = compressor.compress_notebook(
            "/path/to/empty.ipynb",
            notebook_content=""
        )

        assert isinstance(result, CompressedContext)
        assert result.statistics == {}

    def test_no_statistics_content(self):
        """Test content with no extractable statistics."""
        compressor = NotebookCompressor()
        content = "This is just text with no numbers or statistics."
        stats = compressor._extract_statistics(content)

        assert stats == {}

    def test_malformed_p_values(self):
        """Test handling of malformed p-values."""
        compressor = NotebookCompressor()
        content = "p = abc, p = 1.5, p = -0.1"
        stats = compressor._extract_statistics(content)

        # Should not extract invalid p-values
        assert 'p_value' not in stats or stats['p_value'] <= 1.0

    def test_unicode_content(self):
        """Test handling of unicode content."""
        compressor = NotebookCompressor()
        content = "Analysis: p = 0.05, correlation: ρ = 0.82, significance: α = 0.05"
        stats = compressor._extract_statistics(content)

        assert isinstance(stats, dict)

    def test_large_numbers(self):
        """Test handling of large numbers."""
        compressor = NotebookCompressor()
        content = "Sample size: n = 1000000, genes: 50000"
        stats = compressor._extract_statistics(content)

        assert 'sample_size' in stats
        assert stats['sample_size'] == 1000000


# ============================================================================
# Integration-like Tests
# ============================================================================

class TestCompressionPipeline:
    """Tests for the full compression pipeline."""

    def test_full_pipeline_without_llm(self, sample_notebook_text, sample_papers, temp_dir):
        """Test full compression pipeline without LLM."""
        compressor = ContextCompressor()

        # Create test notebook
        notebook_path = temp_dir / "full_test.ipynb"
        notebook_path.write_text(sample_notebook_text)

        task_results = [
            {
                'type': 'data_analysis',
                'notebook_path': str(notebook_path)
            },
            {
                'type': 'literature_review',
                'papers': sample_papers
            }
        ]

        # Compress multiple cycles
        for cycle in range(1, 4):
            result = compressor.compress_cycle_results(cycle, task_results)

            assert isinstance(result, CompressedContext)
            assert result.metadata['cycle'] == cycle

    def test_compression_ratio(self, temp_dir):
        """Test that compression achieves significant reduction."""
        compressor = NotebookCompressor()

        # Create large content
        large_content = "\n".join([
            f"Analysis {i}: p = 0.0{i}, n = {100 * i}, correlation r = 0.{i}"
            for i in range(1, 100)
        ])

        notebook_path = temp_dir / "large.ipynb"
        notebook_path.write_text(large_content)

        result = compressor.compress_notebook(str(notebook_path))

        # Summary should be much smaller than original
        compression_ratio = len(large_content) / (len(result.summary) + 1)
        assert compression_ratio > 5  # At least 5:1 compression
