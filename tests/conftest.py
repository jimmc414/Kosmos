"""
Shared pytest fixtures for Kosmos tests.

This module provides common fixtures used across all test suites.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, Mock

import dotenv
import pytest


# ============================================================================
# Hermetic environment at import time (before pytest collection)
# ============================================================================
# Unit and integration tests see neither .env nor credentials exported by the shell,
# exactly as CI does: a failing assertion on a config value must never print a real
# key. tests/e2e/conftest.py loads .env itself for the live tests.

_CREDENTIAL_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL")
for _name in list(os.environ):
    _upper = _name.upper()
    if any(marker in _upper for marker in _CREDENTIAL_MARKERS) or _upper.endswith("_URL"):
        del os.environ[_name]

# Writable paths default into a per-run temp dir, never the repo root (kosmos.db,
# human_review_audit.jsonl). scripts/verify_isolate.py overrides the ones it sets.
_TEST_RUN_DIR = Path(tempfile.mkdtemp(prefix="kosmos-tests-"))
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_RUN_DIR}/test.db"
os.environ["AUDIT_LOG_PATH"] = str(_TEST_RUN_DIR / "human_review_audit.jsonl")
os.environ["INCIDENT_LOG_PATH"] = str(_TEST_RUN_DIR / "safety_incidents.jsonl")
os.environ["KOSMOS_LITERATURE_CACHE_DIR"] = str(_TEST_RUN_DIR / "literature_cache")

# Fixed, non-secret provider settings (the shape of the project's DeepSeek .env). The key
# is a placeholder: a test that reaches the network fails authentication instead of spending.
os.environ["LLM_PROVIDER"] = "litellm"
os.environ["LITELLM_MODEL"] = "deepseek/deepseek-chat"
os.environ["DEEPSEEK_API_KEY"] = "test-placeholder-not-a-real-key"
# litellm otherwise downloads its model-cost map at import (a network call)
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

# What the suite starts from (tests/unit/test_hermetic_env.py checks it holds no credential)
HERMETIC_ENV_AT_IMPORT = dict(os.environ)


def _no_dotenv(*args, **kwargs):
    """Stand-in for dotenv.load_dotenv (kosmos/cli/main.py calls it on every command)."""
    return False


dotenv.load_dotenv = _no_dotenv

import kosmos.config as _kosmos_config  # noqa: E402

for _settings_class in (_kosmos_config.KosmosConfig, _kosmos_config.LiteLLMConfig):
    _settings_class.model_config["env_file"] = None

import kosmos.knowledge.graph as _kosmos_graph  # noqa: E402

# Tests never start containers: KnowledgeGraph would otherwise run
# `docker-compose up -d neo4j` in the repo when kosmos-neo4j is not running.
_kosmos_graph.KnowledgeGraph._ensure_container_running = lambda self: None

from kosmos.literature.base_client import Author, PaperMetadata, PaperSource  # noqa: E402


# ============================================================================
# Path and File Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def hermetic_env_at_import() -> Dict[str, str]:
    """The environment as this conftest left it at import, before any test ran."""
    return dict(HERMETIC_ENV_AT_IMPORT)


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Return path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def sample_papers_json(fixtures_dir: Path) -> Path:
    """Return path to sample papers JSON file."""
    return fixtures_dir / "sample_papers.json"


@pytest.fixture(scope="session")
def sample_arxiv_xml(fixtures_dir: Path) -> Path:
    """Return path to sample arXiv XML response."""
    return fixtures_dir / "sample_arxiv_response.xml"


@pytest.fixture(scope="session")
def sample_semantic_scholar_json(fixtures_dir: Path) -> Path:
    """Return path to sample Semantic Scholar JSON response."""
    return fixtures_dir / "sample_semantic_scholar_response.json"


@pytest.fixture(scope="session")
def sample_pubmed_xml(fixtures_dir: Path) -> Path:
    """Return path to sample PubMed XML response."""
    return fixtures_dir / "sample_pubmed_response.xml"


@pytest.fixture(scope="session")
def sample_bibtex(fixtures_dir: Path) -> Path:
    """Return path to sample BibTeX file."""
    return fixtures_dir / "sample.bib"


@pytest.fixture(scope="session")
def sample_ris(fixtures_dir: Path) -> Path:
    """Return path to sample RIS file."""
    return fixtures_dir / "sample.ris"


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_file(temp_dir):
    """Create a temporary file for tests."""
    def _create_temp_file(filename: str, content: str = "") -> Path:
        file_path = temp_dir / filename
        file_path.write_text(content)
        return file_path
    return _create_temp_file


# ============================================================================
# Sample Data Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def sample_papers_data(sample_papers_json: Path) -> List[Dict]:
    """Load sample papers data from JSON."""
    with open(sample_papers_json) as f:
        data = json.load(f)
    return data["papers"]


def _source_to_enum(source_str: str) -> PaperSource:
    """Convert source string to PaperSource enum."""
    mapping = {
        "arxiv": PaperSource.ARXIV,
        "semantic_scholar": PaperSource.SEMANTIC_SCHOLAR,
        "pubmed": PaperSource.PUBMED,
        "unknown": PaperSource.UNKNOWN,
        "manual": PaperSource.MANUAL,
    }
    return mapping.get(source_str.lower(), PaperSource.UNKNOWN)


@pytest.fixture
def sample_paper_metadata(sample_papers_data: List[Dict]) -> PaperMetadata:
    """Return a single sample PaperMetadata object."""
    paper_dict = sample_papers_data[0]  # "Attention Is All You Need"
    source_str = paper_dict.get("source", "unknown")
    paper_id = paper_dict.get("arxiv_id") or paper_dict.get("doi") or f"paper_{hash(paper_dict['title'])}"
    return PaperMetadata(
        id=paper_id,
        source=_source_to_enum(source_str),
        title=paper_dict["title"],
        authors=[Author(name=name) for name in paper_dict["authors"]],
        abstract=paper_dict["abstract"],
        year=paper_dict["year"],
        venue=paper_dict.get("venue"),
        doi=paper_dict.get("doi"),
        arxiv_id=paper_dict.get("arxiv_id"),
        pubmed_id=paper_dict.get("pubmed_id"),
        url=paper_dict.get("url"),
        pdf_url=paper_dict.get("pdf_url"),
        citation_count=paper_dict.get("citation_count", 0),
    )


@pytest.fixture
def sample_papers_list(sample_papers_data: List[Dict]) -> List[PaperMetadata]:
    """Return a list of sample PaperMetadata objects."""
    papers = []
    for paper_dict in sample_papers_data:
        source_str = paper_dict.get("source", "unknown")
        paper_id = paper_dict.get("arxiv_id") or paper_dict.get("doi") or f"paper_{hash(paper_dict['title'])}"
        papers.append(
            PaperMetadata(
                id=paper_id,
                source=_source_to_enum(source_str),
                title=paper_dict["title"],
                authors=[Author(name=name) for name in paper_dict["authors"]],
                abstract=paper_dict["abstract"],
                year=paper_dict["year"],
                venue=paper_dict.get("venue"),
                doi=paper_dict.get("doi"),
                arxiv_id=paper_dict.get("arxiv_id"),
                pubmed_id=paper_dict.get("pubmed_id"),
                url=paper_dict.get("url"),
                pdf_url=paper_dict.get("pdf_url"),
                citation_count=paper_dict.get("citation_count", 0),
            )
        )
    return papers


# ============================================================================
# Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_llm_client():
    """Mock Claude LLM client."""
    mock = Mock()
    mock.generate.return_value = "Mocked Claude response"
    mock.generate_structured.return_value = {
        "executive_summary": "This is a summary.",
        "key_findings": ["Finding 1", "Finding 2"],
        "methodology": "The methodology used.",
        "significance": "Significance of the work.",
        "limitations": ["Limitation 1"],
        "confidence_score": 0.85,
    }
    return mock


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic API client."""
    mock = Mock()
    mock.messages.create.return_value = Mock(
        content=[Mock(text="Mocked API response")],
        usage=Mock(input_tokens=100, output_tokens=50),
    )
    return mock


@pytest.fixture
def mock_knowledge_graph():
    """Mock knowledge graph."""
    mock = Mock()
    mock.add_paper.return_value = "paper_id_123"
    mock.add_concept.return_value = "concept_id_456"
    mock.add_citation.return_value = "citation_id_789"
    mock.get_citations.return_value = []
    mock.get_concept_papers.return_value = []
    mock.get_stats.return_value = {
        "total_papers": 100,
        "total_concepts": 50,
        "total_citations": 200,
    }
    return mock


@pytest.fixture
def mock_vector_db():
    """Mock vector database."""
    mock = Mock()
    mock.add_papers.return_value = None
    mock.search.return_value = []
    mock.get_paper_count.return_value = 0
    return mock


@pytest.fixture
def mock_concept_extractor():
    """Mock concept extractor."""
    mock = Mock()
    mock.extract_from_paper.return_value = Mock(
        paper_id="paper_123",
        concepts=[
            Mock(name="Machine Learning", category="Method", relevance=0.9),
            Mock(name="Neural Networks", category="Concept", relevance=0.85),
        ],
        methods=[
            Mock(name="Transformer Architecture", category="Architecture", relevance=0.95),
        ],
        relationships=[],
        extraction_time=1.5,
        model_used="claude-sonnet-4.5",
    )
    return mock


@pytest.fixture
def mock_cache():
    """Mock cache with simple dict-based storage."""
    cache_dict = {}

    class MockCache:
        def get(self, key):
            return cache_dict.get(key)

        def set(self, key, value, ttl=None):
            cache_dict[key] = value

        def delete(self, key):
            if key in cache_dict:
                del cache_dict[key]

        def clear(self):
            cache_dict.clear()

        def exists(self, key):
            return key in cache_dict

    return MockCache()


# ============================================================================
# API Response Fixtures
# ============================================================================

@pytest.fixture
def arxiv_response_xml(sample_arxiv_xml: Path) -> str:
    """Load sample arXiv XML response."""
    return sample_arxiv_xml.read_text()


@pytest.fixture
def semantic_scholar_response_json(sample_semantic_scholar_json: Path) -> Dict:
    """Load sample Semantic Scholar JSON response."""
    return json.loads(sample_semantic_scholar_json.read_text())


@pytest.fixture
def pubmed_response_xml(sample_pubmed_xml: Path) -> str:
    """Load sample PubMed XML response."""
    return sample_pubmed_xml.read_text()


# ============================================================================
# Environment and Configuration Fixtures
# ============================================================================

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up mock environment variables for testing."""
    env_vars = {
        "ANTHROPIC_API_KEY": "test_api_key",
        "SEMANTIC_SCHOLAR_API_KEY": "test_s2_key",
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "test_password",
        "CHROMA_PERSIST_DIR": ".chroma_test_db",
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    return env_vars


@pytest.fixture
def test_config():
    """Provide test configuration."""
    return {
        "model": "claude-sonnet-4-5",
        "max_tokens": 1024,
        "temperature": 0.7,
        "cache_ttl": 3600,
        "use_knowledge_graph": False,
        "use_semantic_similarity": False,
    }


# ============================================================================
# Cleanup Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset all singleton instances before each test."""
    # This fixture is optional - tests should still run even if reset functions aren't available
    reset_funcs = []

    # Try to import reset functions, but don't fail if they're not available
    try:
        from kosmos.knowledge.graph import reset_knowledge_graph
        reset_funcs.append((reset_knowledge_graph, "knowledge_graph"))
    except ImportError:
        pass

    try:
        from kosmos.knowledge.vector_db import reset_vector_db
        reset_funcs.append((reset_vector_db, "vector_db"))
    except ImportError:
        pass

    try:
        from kosmos.knowledge.embeddings import reset_embedder
        reset_funcs.append((reset_embedder, "embedder"))
    except ImportError:
        pass

    try:
        from kosmos.knowledge.concept_extractor import reset_concept_extractor
        reset_funcs.append((reset_concept_extractor, "concept_extractor"))
    except ImportError:
        pass

    try:
        from kosmos.literature.reference_manager import reset_reference_manager
        reset_funcs.append((reset_reference_manager, "reference_manager"))
    except ImportError:
        pass

    try:
        from kosmos.world_model.factory import reset_world_model
        reset_funcs.append((reset_world_model, "world_model"))
    except ImportError:
        pass

    # This ensures tests don't interfere with each other
    yield

    # Reset singletons after test
    for reset_func, name in reset_funcs:
        try:
            reset_func()
        except Exception as e:
            # Log the error but continue with other resets
            import warnings
            warnings.warn(f"Failed to reset {name}: {e}", RuntimeWarning)


@pytest.fixture(autouse=True)
def cleanup_test_files(temp_dir):
    """Clean up any test files created during tests."""
    yield
    # Cleanup happens automatically with temp_dir


# ============================================================================
# Marker-based Skipping
# ============================================================================

def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow"
    )
    config.addinivalue_line(
        "markers", "requires_api_key: mark test as requiring API keys"
    )
    config.addinivalue_line(
        "markers", "requires_neo4j: mark test as requiring Neo4j"
    )
    config.addinivalue_line(
        "markers", "requires_chromadb: mark test as requiring ChromaDB"
    )
    config.addinivalue_line(
        "markers", "requires_claude: mark test as requiring Claude API"
    )
    config.addinivalue_line(
        "markers", "requires_network: calls a public API over the network (KOSMOS_TEST_NETWORK=1 to run)"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically skip tests based on markers if dependencies not available."""
    skip_api_key = pytest.mark.skip(reason="API keys not configured")
    skip_neo4j = pytest.mark.skip(reason="Neo4j not available")
    skip_chromadb = pytest.mark.skip(reason="ChromaDB not available")
    skip_claude = pytest.mark.skip(reason="Claude API not configured")
    skip_execution = pytest.mark.skip(reason="Execution environment not available")
    skip_network = pytest.mark.skip(reason="network tests are opt-in: KOSMOS_TEST_NETWORK=1")

    # Check environment
    has_api_keys = os.getenv("ANTHROPIC_API_KEY") and os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    has_neo4j = os.getenv("NEO4J_URI")
    has_claude = os.getenv("ANTHROPIC_API_KEY")
    run_network = os.getenv("KOSMOS_TEST_NETWORK") == "1"

    for item in items:
        if "requires_api_key" in item.keywords and not has_api_keys:
            item.add_marker(skip_api_key)
        if "requires_neo4j" in item.keywords and not has_neo4j:
            item.add_marker(skip_neo4j)
        if "requires_claude" in item.keywords and not has_claude:
            item.add_marker(skip_claude)
        if "requires_execution_env" in item.keywords:
            item.add_marker(skip_execution)
        if "requires_network" in item.keywords and not run_network:
            item.add_marker(skip_network)


# ============================================================================
# Gap Module Fixtures (Compression, Orchestration, Validation, Workflow)
# ============================================================================

@pytest.fixture
def mock_context_compressor():
    """Mock context compressor for testing."""
    mock = Mock()
    mock.compress_cycle_results.return_value = Mock(
        summary="Cycle summary",
        statistics={'n_tasks': 5},
        metadata={'cycle': 1}
    )
    mock.notebook_compressor = Mock()
    mock.literature_compressor = Mock()
    return mock


@pytest.fixture
def mock_artifact_state_manager(temp_dir):
    """Mock artifact state manager for testing."""
    from unittest.mock import AsyncMock

    mock = Mock()
    mock.artifacts_dir = temp_dir / "artifacts"
    mock.artifacts_dir.mkdir(parents=True, exist_ok=True)

    mock.save_finding_artifact = AsyncMock(return_value=mock.artifacts_dir / "finding.json")
    mock.save_hypothesis = AsyncMock(return_value="hyp_001")
    mock.get_finding.return_value = None
    mock.get_all_cycle_findings.return_value = []
    mock.get_all_findings.return_value = []
    mock.get_validated_findings.return_value = []
    mock.get_cycle_context.return_value = {
        'cycle': 1,
        'findings_count': 0,
        'recent_findings': [],
        'unsupported_hypotheses': [],
        'validated_discoveries': [],
        'statistics': {}
    }
    mock.generate_cycle_summary = AsyncMock(return_value="# Summary")
    mock.get_statistics.return_value = {
        'total_findings': 0,
        'validated_findings': 0,
        'validation_rate': 0
    }
    return mock


@pytest.fixture
def mock_skill_loader():
    """Mock skill loader for testing."""
    mock = Mock()
    mock.load_skills_for_task.return_value = "# Skills\n\nAvailable libraries..."
    mock.get_available_bundles.return_value = ['single_cell_analysis', 'genomics_analysis']
    mock.get_bundle_skills.return_value = ['scanpy', 'anndata']
    mock.search_skills.return_value = []
    mock.get_statistics.return_value = {
        'total_skills': 100,
        'predefined_bundles': 8
    }
    return mock


@pytest.fixture
def mock_scholar_eval_validator():
    """Mock ScholarEval validator for testing."""
    mock = Mock()
    mock.evaluate_finding = Mock(return_value=Mock(
        novelty=0.8,
        rigor=0.85,
        clarity=0.75,
        reproducibility=0.80,
        impact=0.70,
        coherence=0.75,
        limitations=0.65,
        ethics=0.70,
        overall_score=0.78,
        passes_threshold=True,
        feedback='Good finding',
        to_dict=lambda: {'overall_score': 0.78, 'passes_threshold': True}
    ))
    mock.threshold = 0.75
    mock.min_rigor_score = 0.70
    return mock


@pytest.fixture
def mock_plan_creator():
    """Mock plan creator for testing."""
    from unittest.mock import AsyncMock

    mock = Mock()
    mock.create_plan = AsyncMock()
    mock.revise_plan = AsyncMock()
    mock._get_exploration_ratio.return_value = 0.7
    return mock


@pytest.fixture
def mock_plan_reviewer():
    """Mock plan reviewer for testing."""
    from unittest.mock import AsyncMock

    mock = Mock()
    mock.review_plan = AsyncMock(return_value=Mock(
        approved=True,
        scores={'specificity': 8.0, 'relevance': 8.0, 'novelty': 7.0,
                'coverage': 7.5, 'feasibility': 8.0},
        average_score=7.7,
        min_score=7.0,
        feedback='Good plan',
        required_changes=[],
        suggestions=[],
        to_dict=lambda: {'approved': True, 'average_score': 7.7}
    ))
    mock.min_average_score = 7.0
    mock.min_dimension_score = 5.0
    return mock


@pytest.fixture
def mock_delegation_manager():
    """Mock delegation manager for testing."""
    from unittest.mock import AsyncMock

    mock = Mock()
    mock.execute_plan = AsyncMock(return_value={
        'completed_tasks': [
            {'task_id': 1, 'status': 'completed', 'finding': {'summary': 'Test finding'}}
        ],
        'failed_tasks': [],
        'execution_summary': {
            'total_tasks': 1,
            'completed_tasks': 1,
            'failed_tasks': 0,
            'success_rate': 1.0
        }
    })
    mock.max_parallel_tasks = 3
    mock.max_retries = 2
    return mock


@pytest.fixture
def mock_novelty_detector():
    """Mock novelty detector for testing."""
    mock = Mock()
    mock.index_past_tasks.return_value = None
    mock.check_task_novelty.return_value = {
        'is_novel': True,
        'novelty_score': 0.9,
        'max_similarity': 0.1,
        'similar_tasks': []
    }
    mock.check_plan_novelty.return_value = {
        'plan_novelty_score': 0.85,
        'novel_task_count': 8,
        'redundant_task_count': 2,
        'task_novelties': []
    }
    mock.clear_index.return_value = None
    mock.get_statistics.return_value = {
        'total_indexed_tasks': 50,
        'novelty_threshold': 0.75
    }
    return mock


@pytest.fixture
def sample_research_finding():
    """Sample research finding for testing."""
    return {
        'finding_id': 'cycle1_task1',
        'cycle': 1,
        'task_id': 1,
        'summary': 'Found 42 differentially expressed genes with p < 0.001',
        'statistics': {
            'p_value': 0.001,
            'sample_size': 150,
            'n_genes': 42,
            'effect_size': 0.85
        },
        'methods': 'DESeq2 differential expression analysis',
        'interpretation': 'Significant gene expression changes in treatment group',
        'evidence_type': 'data_analysis',
        'notebook_path': '/path/to/analysis.ipynb'
    }


@pytest.fixture
def sample_research_hypothesis():
    """Sample research hypothesis for testing."""
    return {
        'hypothesis_id': 'hyp_001',
        'statement': 'KRAS mutations are associated with poor prognosis in pancreatic cancer',
        'status': 'unknown',
        'domain': 'oncology',
        'confidence': 0.0,
        'supporting_evidence': [],
        'refuting_evidence': []
    }


@pytest.fixture
def sample_research_plan():
    """Sample research plan for testing."""
    return {
        'cycle': 1,
        'tasks': [
            {
                'id': 1,
                'type': 'data_analysis',
                'description': 'Analyze gene expression data',
                'expected_output': 'DEG list',
                'required_skills': ['deseq2', 'pandas'],
                'exploration': True,
                'priority': 1
            },
            {
                'id': 2,
                'type': 'literature_review',
                'description': 'Review KRAS mutation papers',
                'expected_output': 'Literature summary',
                'required_skills': [],
                'exploration': False,
                'priority': 2
            },
            {
                'id': 3,
                'type': 'data_analysis',
                'description': 'Validate findings',
                'expected_output': 'Validation results',
                'required_skills': ['scipy'],
                'exploration': False,
                'priority': 2
            },
            {
                'id': 4,
                'type': 'data_analysis',
                'description': 'Additional analysis',
                'expected_output': 'Results',
                'required_skills': [],
                'exploration': True,
                'priority': 3
            }
        ],
        'rationale': 'Investigate KRAS mutations',
        'exploration_ratio': 0.5
    }


# ============================================================================
# Real Service Fixtures (for integration/E2E tests)
# ============================================================================

@pytest.fixture
def deepseek_client():
    """DeepSeek client via LiteLLM for cost-effective unit tests.

    Uses DeepSeek 3.2 model for real LLM testing.
    Skips if DEEPSEEK_API_KEY is not set.
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        pytest.skip("DEEPSEEK_API_KEY required for real LLM tests")
    try:
        import litellm
        litellm.set_verbose = False
        return litellm
    except ImportError:
        pytest.skip("litellm package required")


@pytest.fixture
def real_anthropic_client():
    """Real Anthropic client for E2E tests.

    Uses Anthropic Haiku for production-compatible testing.
    Skips if ANTHROPIC_API_KEY is not set.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY required for E2E tests")
    try:
        import anthropic
        return anthropic.Anthropic(api_key=api_key)
    except ImportError:
        pytest.skip("anthropic package required")


@pytest.fixture
def real_vector_db(temp_dir):
    """Real ChromaDB with ephemeral storage.

    Creates an in-memory ChromaDB client for testing.
    """
    try:
        import chromadb
        return chromadb.Client()
    except ImportError:
        pytest.skip("chromadb package required")


@pytest.fixture
def real_embedder():
    """Real SentenceTransformer embedder.

    Downloads and uses all-MiniLM-L6-v2 model.
    First run may take time to download the model.
    """
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        pytest.skip("sentence-transformers package required")


@pytest.fixture
def real_knowledge_graph():
    """Real Neo4j connection.

    Connects to Neo4j database using environment configuration.
    Skips if Neo4j is not available.
    """
    try:
        from kosmos.knowledge.graph import KnowledgeGraph
        kg = KnowledgeGraph()
        if not kg.connected:
            pytest.skip("Neo4j not available")
        yield kg
    except Exception as e:
        pytest.skip(f"KnowledgeGraph initialization failed: {e}")
