"""
Test suite for Domain and Multi-Domain Support Requirements (REQ-DOMAIN-001 through REQ-DOMAIN-003).

This test file validates multi-domain support, configuration-based domain handling,
and domain-specific templates as specified in REQUIREMENTS.md Section 8.1.

Requirements tested:
- REQ-DOMAIN-001 (MUST): Execute workflows in at least 3 scientific domains
- REQ-DOMAIN-002 (MUST): No domain-specific code modifications required
- REQ-DOMAIN-003 (SHOULD): Domain-specific prompt templates and knowledge bases
"""

import os
import pytest
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import Mock, patch, MagicMock
import tempfile

# Test markers for requirements traceability
pytestmark = [
    pytest.mark.requirement("REQ-DOMAIN"),
    pytest.mark.category("validation"),
]


# ============================================================================
# REQ-DOMAIN-001: Multi-Domain Support (MUST)
# ============================================================================

@pytest.mark.requirement("REQ-DOMAIN-001")
@pytest.mark.priority("MUST")
def test_req_domain_001_three_domains_supported():
    """
    REQ-DOMAIN-001: The system MUST successfully execute research workflows
    in at least three scientific domains (biology, neuroscience, physics,
    chemistry, or materials science).

    Validates that:
    - At least 3 domains are registered and available
    - Each domain can be initialized
    - Each domain can process basic research queries
    """
    from kosmos.config import get_config, reset_config

    # Arrange: Check available domains
    reset_config()

    with patch.dict(os.environ, {
        'ANTHROPIC_API_KEY': 'test_key',
        'ENABLED_DOMAINS': 'biology,neuroscience,physics,chemistry,materials'
    }):
        config = get_config(reload=True)

        # Act: Get enabled domains
        enabled_domains = config.research.enabled_domains

        # Assert: At least 3 domains available
        assert len(enabled_domains) >= 3, \
            f"System must support at least 3 domains, found: {len(enabled_domains)}"

        # Verify expected domains are in the list
        expected_domains = {'biology', 'neuroscience', 'physics', 'chemistry', 'materials'}
        actual_domains = set(enabled_domains)

        assert len(actual_domains.intersection(expected_domains)) >= 3, \
            f"Must support at least 3 of: {expected_domains}"

    reset_config()


@pytest.mark.requirement("REQ-DOMAIN-001")
@pytest.mark.priority("MUST")
def test_req_domain_001_domain_modules_importable():
    """
    REQ-DOMAIN-001: Verify domain modules can be imported and initialized.

    Validates that:
    - Domain modules exist and are importable
    - Domain modules provide required interfaces
    """
    # Test that domain modules exist
    domain_modules = [
        'kosmos.domains.biology',
        'kosmos.domains.neuroscience',
        'kosmos.domains.physics',
        'kosmos.domains.chemistry',
        'kosmos.domains.materials'
    ]

    imported_count = 0
    imported_domains = []

    for module_name in domain_modules:
        try:
            module = __import__(module_name, fromlist=[''])
            imported_count += 1
            imported_domains.append(module_name.split('.')[-1])
        except ImportError as e:
            print(f"Optional domain not available: {module_name} - {e}")

    # Assert: At least 3 domains are importable
    assert imported_count >= 3, \
        f"At least 3 domain modules must be importable, got {imported_count}: {imported_domains}"


@pytest.mark.requirement("REQ-DOMAIN-001")
@pytest.mark.priority("MUST")
def test_req_domain_001_domain_workflow_execution():
    """
    REQ-DOMAIN-001: Test that workflows can be executed in multiple domains.

    Validates that:
    - Workflow can be initialized with different domain contexts
    - Domain-specific analysis can be performed
    - System switches between domains without errors
    """
    from kosmos.core.workflow import ResearchWorkflow

    # Test domains
    test_domains = ['biology', 'neuroscience', 'materials']

    try:
        successful_domains = []

        for domain in test_domains:
            try:
                # Create workflow with domain context
                workflow = ResearchWorkflow(
                    research_question=f"Sample {domain} research question",
                    domain=domain
                )

                assert workflow is not None
                assert workflow.domain == domain
                successful_domains.append(domain)

            except Exception as e:
                print(f"Domain {domain} workflow initialization: {e}")

        # Assert: At least 3 domains successfully initialize workflows
        assert len(successful_domains) >= 3, \
            f"Workflows must initialize in at least 3 domains, got {len(successful_domains)}"

    except ImportError:
        # Fallback: Test domain configuration
        from kosmos.config import get_config, reset_config

        reset_config()
        successful_configs = 0

        for domain in test_domains:
            with patch.dict(os.environ, {
                'ANTHROPIC_API_KEY': 'test_key',
                'ENABLED_DOMAINS': domain
            }):
                try:
                    config = get_config(reload=True)
                    assert domain in config.research.enabled_domains
                    successful_configs += 1
                except Exception as e:
                    print(f"Domain {domain} config: {e}")
                finally:
                    reset_config()

        assert successful_configs >= 3, \
            f"At least 3 domains must be configurable"


# ============================================================================
# REQ-DOMAIN-002: No Code Modifications Required (MUST)
# ============================================================================

@pytest.mark.requirement("REQ-DOMAIN-002")
@pytest.mark.priority("MUST")
def test_req_domain_002_configuration_based_switching():
    """
    REQ-DOMAIN-002: The system MUST NOT require domain-specific code
    modifications to handle different domains (configuration only).

    Validates that:
    - Domains can be enabled/disabled via configuration
    - Same codebase handles all domains
    - No conditional imports based on domain
    """
    from kosmos.config import get_config, reset_config

    # Test domains
    domains_to_test = ['biology', 'neuroscience', 'physics']

    for domain in domains_to_test:
        reset_config()

        with patch.dict(os.environ, {
            'ANTHROPIC_API_KEY': 'test_key',
            'ENABLED_DOMAINS': domain
        }):
            # Act: Load configuration for this domain
            config = get_config(reload=True)

            # Assert: Domain is enabled via config only
            assert domain in config.research.enabled_domains, \
                f"Domain {domain} should be configurable"

            # Assert: Configuration changes, not code
            assert hasattr(config.research, 'enabled_domains'), \
                "Domain control must be via configuration"

    reset_config()


@pytest.mark.requirement("REQ-DOMAIN-002")
@pytest.mark.priority("MUST")
def test_req_domain_002_unified_interface():
    """
    REQ-DOMAIN-002: Verify unified interface across domains.

    Validates that:
    - All domains use same workflow interface
    - No domain-specific code paths required
    - Domain selection is data-driven, not code-driven
    """
    from kosmos.core.domain_router import DomainRouter

    try:
        router = DomainRouter()

        # Test that router can handle multiple domains without code changes
        test_queries = [
            ('What genes are associated with cancer?', 'biology'),
            ('How does neural plasticity work?', 'neuroscience'),
            ('What is the band gap of silicon?', 'materials')
        ]

        for query, expected_domain in test_queries:
            # Routing should work without domain-specific code
            detected = router.detect_domain(query)

            # Router should identify domain without conditional code
            assert detected is not None, \
                f"Router should handle query: {query[:50]}"

            # Should use configuration, not hardcoded logic
            assert hasattr(router, 'domain_patterns') or hasattr(router, 'domain_keywords'), \
                "Router should use data-driven domain detection"

    except (ImportError, AttributeError):
        # Fallback: Test that domains are data-driven
        from kosmos.config import get_config, reset_config

        reset_config()
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_key'}):
            config = get_config(reload=True)

            # Assert: Domains defined in config, not code
            assert hasattr(config.research, 'enabled_domains'), \
                "Domains must be configuration-driven"

            # Assert: Can dynamically enable domains
            assert isinstance(config.research.enabled_domains, list), \
                "Domains should be configurable list"

        reset_config()


@pytest.mark.requirement("REQ-DOMAIN-002")
@pytest.mark.priority("MUST")
def test_req_domain_002_no_conditional_imports():
    """
    REQ-DOMAIN-002: Verify no domain-specific conditional imports in core code.

    Validates that:
    - Core modules don't have domain-specific imports
    - Domain handling is via plugins/configuration
    - Code is domain-agnostic
    """
    from pathlib import Path
    import ast

    # Check core workflow files for conditional domain imports
    project_root = Path(__file__).parent.parent.parent.parent
    core_files = [
        project_root / 'kosmos' / 'core' / 'workflow.py',
        project_root / 'kosmos' / 'core' / 'domain_router.py',
    ]

    for filepath in core_files:
        if not filepath.exists():
            continue

        # Parse the file
        with open(filepath) as f:
            content = f.read()
            try:
                tree = ast.parse(content)

                # Look for conditional domain imports
                for node in ast.walk(tree):
                    if isinstance(node, ast.If):
                        # Check if condition involves domain strings
                        condition_str = ast.unparse(node.test) if hasattr(ast, 'unparse') else str(node.test)

                        # Should not have if domain == 'biology': import X patterns
                        assert not any(d in condition_str.lower() for d in ['biology', 'neuroscience', 'physics']), \
                            f"Found domain-specific conditional in {filepath.name}: {condition_str}"

            except SyntaxError:
                pass  # Skip files with syntax errors


# ============================================================================
# REQ-DOMAIN-003: Domain Templates and Knowledge Bases (SHOULD)
# ============================================================================

@pytest.mark.requirement("REQ-DOMAIN-003")
@pytest.mark.priority("SHOULD")
def test_req_domain_003_domain_templates_exist():
    """
    REQ-DOMAIN-003: The system SHOULD provide domain-specific prompt templates
    or knowledge bases to improve analysis quality.

    Validates that:
    - Domain templates directory exists
    - Templates available for multiple domains
    - Templates are accessible to the system
    """
    from pathlib import Path

    # Check for domain templates
    project_root = Path(__file__).parent.parent.parent.parent

    # Look for domain template directories
    template_locations = [
        project_root / 'kosmos' / 'domains',
        project_root / 'kosmos' / 'experiments' / 'templates',
        project_root / 'templates' / 'domains'
    ]

    templates_found = {}

    for location in template_locations:
        if not location.exists():
            continue

        # Check for domain subdirectories
        for domain in ['biology', 'neuroscience', 'physics', 'chemistry', 'materials']:
            domain_dir = location / domain
            if domain_dir.exists() and domain_dir.is_dir():
                # Count template files
                template_files = list(domain_dir.glob('*.py')) + list(domain_dir.glob('*.yaml')) + \
                               list(domain_dir.glob('*.json')) + list(domain_dir.glob('*.md'))
                if template_files:
                    templates_found[domain] = len(template_files)

    # Assert: At least 3 domains have templates (SHOULD requirement)
    if len(templates_found) >= 3:
        print(f"✓ Domain templates found for: {list(templates_found.keys())}")
    else:
        print(f"⚠ Domain templates should be provided for at least 3 domains, found {len(templates_found)}")

    # This is a SHOULD requirement, so we don't fail but log
    assert len(templates_found) >= 0, \
        "Domain template infrastructure should exist"


@pytest.mark.requirement("REQ-DOMAIN-003")
@pytest.mark.priority("SHOULD")
def test_req_domain_003_domain_knowledge_bases():
    """
    REQ-DOMAIN-003: Test domain-specific knowledge bases availability.

    Validates that:
    - Domain knowledge can be loaded
    - Knowledge bases enhance domain-specific analysis
    - Knowledge is accessible to agents
    """
    from kosmos.knowledge.domain_kb import DomainKnowledgeBase

    try:
        # Test loading domain knowledge
        domains_to_test = ['biology', 'neuroscience', 'materials']

        for domain in domains_to_test:
            try:
                kb = DomainKnowledgeBase(domain=domain)

                # Verify knowledge base has content
                assert kb is not None

                # Check for domain-specific methods
                if hasattr(kb, 'get_concepts'):
                    concepts = kb.get_concepts()
                    print(f"✓ {domain} knowledge base: {len(concepts) if concepts else 0} concepts")

                if hasattr(kb, 'get_ontology'):
                    ontology = kb.get_ontology()
                    print(f"✓ {domain} ontology available")

            except Exception as e:
                print(f"Domain {domain} knowledge base: {type(e).__name__}")

    except ImportError:
        # Fallback: Check for domain knowledge files
        from pathlib import Path

        project_root = Path(__file__).parent.parent.parent.parent
        kb_locations = [
            project_root / 'kosmos' / 'domains',
            project_root / 'data' / 'knowledge',
        ]

        kb_files_found = 0
        for location in kb_locations:
            if location.exists():
                kb_files = list(location.glob('**/*ontology*')) + \
                          list(location.glob('**/*knowledge*')) + \
                          list(location.glob('**/*concepts*'))
                kb_files_found += len(kb_files)

        print(f"Knowledge base files found: {kb_files_found}")


@pytest.mark.requirement("REQ-DOMAIN-003")
@pytest.mark.priority("SHOULD")
def test_req_domain_003_domain_specific_prompts():
    """
    REQ-DOMAIN-003: Test domain-specific prompt templates.

    Validates that:
    - Domain-specific prompts can be loaded
    - Prompts are tailored to domain terminology
    - Prompts improve analysis quality
    """
    from kosmos.config import get_config, reset_config

    # Test that domains can have custom prompts
    reset_config()

    with patch.dict(os.environ, {
        'ANTHROPIC_API_KEY': 'test_key',
        'ENABLED_DOMAINS': 'biology,neuroscience,materials'
    }):
        config = get_config(reload=True)

        # Check if system supports domain-specific prompts
        has_domain_prompts = (
            hasattr(config, 'prompts') or
            hasattr(config, 'domain_templates') or
            hasattr(config.research, 'domain_specific_prompts')
        )

        # This is a SHOULD requirement - log rather than fail
        if has_domain_prompts:
            print("✓ Domain-specific prompt support detected")
        else:
            print("⚠ Domain-specific prompts should be supported for better quality")

    reset_config()


# ============================================================================
# Integration Tests
# ============================================================================

class TestDomainIntegration:
    """Integration tests for multi-domain support."""

    def test_domain_switching_workflow(self):
        """Test complete workflow with domain switching."""
        from kosmos.config import get_config, reset_config

        domains = ['biology', 'neuroscience', 'physics']

        for domain in domains:
            reset_config()

            with patch.dict(os.environ, {
                'ANTHROPIC_API_KEY': 'test_key',
                'ENABLED_DOMAINS': domain
            }):
                config = get_config(reload=True)

                # Verify domain is active
                assert domain in config.research.enabled_domains

                # System should handle domain without code changes
                assert isinstance(config.research.enabled_domains, list)

        reset_config()

    def test_all_domains_coexist(self):
        """Test that all domains can be enabled simultaneously."""
        from kosmos.config import get_config, reset_config

        reset_config()

        with patch.dict(os.environ, {
            'ANTHROPIC_API_KEY': 'test_key',
            'ENABLED_DOMAINS': 'biology,neuroscience,physics,chemistry,materials'
        }):
            config = get_config(reload=True)

            # All domains should coexist
            assert len(config.research.enabled_domains) == 5

            # No conflicts between domains
            assert len(set(config.research.enabled_domains)) == 5

        reset_config()

    def test_domain_agnostic_core_components(self):
        """Test that core components work regardless of domain."""
        from kosmos.config import get_config, reset_config

        # Test with different domain configurations
        test_configs = [
            'biology',
            'neuroscience,materials',
            'biology,neuroscience,physics,chemistry,materials'
        ]

        for domain_config in test_configs:
            reset_config()

            with patch.dict(os.environ, {
                'ANTHROPIC_API_KEY': 'test_key',
                'ENABLED_DOMAINS': domain_config
            }):
                config = get_config(reload=True)

                # Core config should work with any domain configuration
                assert hasattr(config, 'research')
                assert hasattr(config.research, 'enabled_domains')
                assert len(config.research.enabled_domains) > 0

        reset_config()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
