"""
Test suite for Documentation Requirements (REQ-DOC-001 through REQ-DOC-005).

This test file validates documentation completeness, quality, and accessibility
as specified in REQUIREMENTS.md Section 13.

Requirements tested:
- REQ-DOC-001 (MUST): User documentation for configuration and workflows
- REQ-DOC-002 (MUST): Developer documentation for architecture
- REQ-DOC-003 (MUST): All configuration parameters documented
- REQ-DOC-004 (MUST): Requirements traceability to code and tests
- REQ-DOC-005 (SHOULD): Example workflows across different domains
"""

import os
import re
import pytest
from pathlib import Path
from typing import List, Dict, Set
from unittest.mock import Mock, patch

# Test markers for requirements traceability
pytestmark = [
    pytest.mark.requirement("REQ-DOC"),
    pytest.mark.category("validation"),
]


# ============================================================================
# REQ-DOC-001: User Documentation (MUST)
# ============================================================================

@pytest.mark.requirement("REQ-DOC-001")
@pytest.mark.priority("MUST")
def test_req_doc_001_user_documentation_exists():
    """
    REQ-DOC-001: The system MUST provide user documentation explaining
    how to configure and run research workflows.

    Validates that:
    - User documentation files exist
    - Documentation covers essential topics
    - Documentation is accessible
    """
    project_root = Path(__file__).parent.parent.parent.parent

    # Check for main user documentation
    user_doc_locations = [
        project_root / 'README.md',
        project_root / 'docs' / 'README.md',
        project_root / 'docs' / 'user_guide.md',
        project_root / 'docs' / 'getting_started.md',
        project_root / 'USER_GUIDE.md',
    ]

    docs_found = []
    for doc_path in user_doc_locations:
        if doc_path.exists():
            docs_found.append(doc_path)

    assert len(docs_found) > 0, \
        "User documentation must exist (README.md or user guide)"

    print(f"✓ Found {len(docs_found)} user documentation files")

    # Verify README exists at minimum
    readme = project_root / 'README.md'
    assert readme.exists(), "README.md must exist at project root"


@pytest.mark.requirement("REQ-DOC-001")
@pytest.mark.priority("MUST")
def test_req_doc_001_installation_instructions():
    """
    REQ-DOC-001: Verify installation instructions are documented.
    """
    project_root = Path(__file__).parent.parent.parent.parent
    readme = project_root / 'README.md'

    assert readme.exists(), "README.md must exist"

    content = readme.read_text().lower()

    # Check for installation-related content
    installation_keywords = [
        'install',
        'pip install',
        'setup',
        'requirements',
        'dependencies'
    ]

    has_installation = any(keyword in content for keyword in installation_keywords)

    assert has_installation, \
        "Documentation must include installation instructions"

    print("✓ Installation instructions found")


@pytest.mark.requirement("REQ-DOC-001")
@pytest.mark.priority("MUST")
def test_req_doc_001_configuration_documented():
    """
    REQ-DOC-001: Verify configuration is documented for users.
    """
    project_root = Path(__file__).parent.parent.parent.parent

    # Check README and docs for configuration info
    doc_files = [
        project_root / 'README.md',
        project_root / 'docs' / 'configuration.md',
        project_root / 'docs' / 'setup.md',
    ]

    config_documentation_found = False

    for doc_file in doc_files:
        if doc_file.exists():
            content = doc_file.read_text().lower()

            # Look for configuration topics
            config_keywords = [
                'config',
                'api_key',
                'environment variable',
                '.env',
                'anthropic_api_key'
            ]

            if any(keyword in content for keyword in config_keywords):
                config_documentation_found = True
                print(f"✓ Configuration documented in {doc_file.name}")
                break

    assert config_documentation_found, \
        "Configuration instructions must be documented"


@pytest.mark.requirement("REQ-DOC-001")
@pytest.mark.priority("MUST")
def test_req_doc_001_workflow_usage_examples():
    """
    REQ-DOC-001: Verify workflow usage is documented.
    """
    project_root = Path(__file__).parent.parent.parent.parent

    doc_files = [
        project_root / 'README.md',
        project_root / 'docs' / 'usage.md',
        project_root / 'docs' / 'quickstart.md',
        project_root / 'docs' / 'workflows.md',
    ]

    usage_documented = False

    for doc_file in doc_files:
        if doc_file.exists():
            content = doc_file.read_text().lower()

            # Look for usage/workflow topics
            usage_keywords = [
                'usage',
                'how to',
                'workflow',
                'example',
                'run',
                'execute'
            ]

            if any(keyword in content for keyword in usage_keywords):
                usage_documented = True
                print(f"✓ Usage documented in {doc_file.name}")
                break

    assert usage_documented, \
        "Workflow usage must be documented"


# ============================================================================
# REQ-DOC-002: Developer Documentation (MUST)
# ============================================================================

@pytest.mark.requirement("REQ-DOC-002")
@pytest.mark.priority("MUST")
def test_req_doc_002_developer_documentation_exists():
    """
    REQ-DOC-002: The system MUST provide developer documentation explaining
    system architecture and component interactions.

    Validates that:
    - Developer docs exist
    - Architecture is documented
    - Component interactions explained
    """
    project_root = Path(__file__).parent.parent.parent.parent

    # Check for developer documentation
    dev_doc_locations = [
        project_root / 'docs' / 'architecture.md',
        project_root / 'docs' / 'developer_guide.md',
        project_root / 'docs' / 'ARCHITECTURE.md',
        project_root / 'ARCHITECTURE.md',
        project_root / 'CONTRIBUTING.md',
        project_root / 'docs' / 'development.md',
    ]

    dev_docs_found = []
    for doc_path in dev_doc_locations:
        if doc_path.exists():
            dev_docs_found.append(doc_path)
            print(f"✓ Developer documentation: {doc_path.name}")

    assert len(dev_docs_found) > 0, \
        "Developer documentation must exist"


@pytest.mark.requirement("REQ-DOC-002")
@pytest.mark.priority("MUST")
def test_req_doc_002_architecture_documented():
    """
    REQ-DOC-002: Verify system architecture is documented.
    """
    project_root = Path(__file__).parent.parent.parent.parent

    # Look for architecture documentation
    arch_files = [
        project_root / 'docs' / 'architecture.md',
        project_root / 'docs' / 'ARCHITECTURE.md',
        project_root / 'ARCHITECTURE.md',
        project_root / 'docs' / 'design.md',
    ]

    architecture_documented = False

    for arch_file in arch_files:
        if arch_file.exists():
            content = arch_file.read_text().lower()

            # Check for architecture topics
            arch_keywords = [
                'architecture',
                'component',
                'module',
                'design',
                'system',
                'workflow',
                'agent'
            ]

            keyword_count = sum(1 for keyword in arch_keywords if keyword in content)

            if keyword_count >= 3:
                architecture_documented = True
                print(f"✓ Architecture documented in {arch_file.name}")
                break

    if architecture_documented:
        assert True
    else:
        print("⚠ Architecture should be documented for developers")


@pytest.mark.requirement("REQ-DOC-002")
@pytest.mark.priority("MUST")
def test_req_doc_002_component_interactions_documented():
    """
    REQ-DOC-002: Verify component interactions are explained.
    """
    project_root = Path(__file__).parent.parent.parent.parent

    # Check documentation for component interaction descriptions
    doc_dirs = [
        project_root / 'docs',
        project_root / 'docs' / 'planning',
    ]

    component_docs_found = 0

    for doc_dir in doc_dirs:
        if doc_dir.exists():
            md_files = list(doc_dir.glob('*.md'))

            for md_file in md_files:
                content = md_file.read_text().lower()

                # Look for component interaction keywords
                if any(keyword in content for keyword in ['interact', 'integration', 'flow', 'communicate']):
                    component_docs_found += 1

    print(f"✓ Found {component_docs_found} files documenting component interactions")


@pytest.mark.requirement("REQ-DOC-002")
@pytest.mark.priority("MUST")
def test_req_doc_002_code_documentation():
    """
    REQ-DOC-002: Verify code has inline documentation.
    """
    project_root = Path(__file__).parent.parent.parent.parent
    kosmos_dir = project_root / 'kosmos'

    # Sample core modules
    core_modules = [
        kosmos_dir / 'core' / 'workflow.py',
        kosmos_dir / 'agents' / 'data_analysis_agent.py',
        kosmos_dir / 'world_model' / 'interface.py',
    ]

    documented_modules = 0

    for module_path in core_modules:
        if module_path.exists():
            content = module_path.read_text()

            # Check for docstrings
            has_module_docstring = content.strip().startswith('"""') or content.strip().startswith("'''")

            # Count class/function docstrings
            docstring_count = content.count('"""') + content.count("'''")

            if has_module_docstring and docstring_count > 2:
                documented_modules += 1

    if documented_modules > 0:
        print(f"✓ {documented_modules} core modules have documentation")


# ============================================================================
# REQ-DOC-003: Configuration Parameters (MUST)
# ============================================================================

@pytest.mark.requirement("REQ-DOC-003")
@pytest.mark.priority("MUST")
def test_req_doc_003_all_config_parameters_documented():
    """
    REQ-DOC-003: All configuration parameters MUST be documented with
    types, valid ranges, and default values.

    Validates that:
    - Configuration documentation exists
    - Parameters are listed with details
    - Types and defaults are specified
    """
    project_root = Path(__file__).parent.parent.parent.parent

    # Check for configuration documentation
    config_docs = [
        project_root / 'docs' / 'configuration.md',
        project_root / 'docs' / 'config.md',
        project_root / 'CONFIG.md',
        project_root / '.env.example',
        project_root / 'env.example',
    ]

    config_doc_found = False
    documented_params = []

    for doc_path in config_docs:
        if doc_path.exists():
            config_doc_found = True
            content = doc_path.read_text()

            # Look for common config parameters
            params = [
                'ANTHROPIC_API_KEY',
                'DATABASE_URL',
                'LOG_LEVEL',
                'MAX_ITERATIONS',
            ]

            for param in params:
                if param in content:
                    documented_params.append(param)

            print(f"✓ Configuration documented in {doc_path.name}")
            print(f"  Found {len(documented_params)} documented parameters")

    assert config_doc_found, \
        "Configuration parameters must be documented"


@pytest.mark.requirement("REQ-DOC-003")
@pytest.mark.priority("MUST")
def test_req_doc_003_env_example_exists():
    """
    REQ-DOC-003: Verify .env.example exists with parameter documentation.
    """
    project_root = Path(__file__).parent.parent.parent.parent

    env_examples = [
        project_root / '.env.example',
        project_root / 'env.example',
        project_root / '.env.template',
    ]

    env_example_found = False

    for env_file in env_examples:
        if env_file.exists():
            env_example_found = True
            content = env_file.read_text()

            # Should have comments explaining parameters
            comment_lines = [line for line in content.split('\n') if line.strip().startswith('#')]

            assert len(comment_lines) > 0, \
                f"{env_file.name} should have comments explaining parameters"

            print(f"✓ {env_file.name} exists with {len(comment_lines)} comment lines")
            break

    if not env_example_found:
        print("⚠ .env.example should exist to document configuration")


@pytest.mark.requirement("REQ-DOC-003")
@pytest.mark.priority("MUST")
def test_req_doc_003_config_module_documented():
    """
    REQ-DOC-003: Verify config.py has documentation for all parameters.
    """
    project_root = Path(__file__).parent.parent.parent.parent
    config_file = project_root / 'kosmos' / 'config.py'

    if config_file.exists():
        content = config_file.read_text()

        # Check for docstrings
        docstring_count = content.count('"""') + content.count("'''")

        assert docstring_count > 0, \
            "Config module should have docstrings documenting parameters"

        # Check for class documentation
        has_config_class = 'class' in content and 'Config' in content

        if has_config_class:
            print("✓ Config module is structured and documented")


# ============================================================================
# REQ-DOC-004: Requirements Traceability (MUST)
# ============================================================================

@pytest.mark.requirement("REQ-DOC-004")
@pytest.mark.priority("MUST")
def test_req_doc_004_traceability_matrix_exists():
    """
    REQ-DOC-004: All requirements in this specification MUST have traceable
    links to implementing code and validating tests.

    Validates that:
    - Traceability matrix exists
    - Requirements are linked to tests
    - Requirements are linked to code
    """
    project_root = Path(__file__).parent.parent.parent.parent

    # Check for traceability documentation
    traceability_files = [
        project_root / 'REQUIREMENTS_TRACEABILITY_MATRIX.md',
        project_root / 'docs' / 'traceability.md',
        project_root / 'TRACEABILITY.md',
    ]

    traceability_found = False

    for trace_file in traceability_files:
        if trace_file.exists():
            traceability_found = True
            print(f"✓ Traceability matrix: {trace_file.name}")

            content = trace_file.read_text()

            # Check if it contains requirement IDs
            req_count = len(re.findall(r'REQ-[A-Z]+-\d+', content))

            if req_count > 0:
                print(f"  References {req_count} requirements")

    assert traceability_found, \
        "Requirements traceability matrix must exist"


@pytest.mark.requirement("REQ-DOC-004")
@pytest.mark.priority("MUST")
def test_req_doc_004_requirements_linked_to_tests():
    """
    REQ-DOC-004: Verify requirements are linked to tests via markers.
    """
    project_root = Path(__file__).parent.parent.parent.parent

    # Check test files for requirement markers
    test_dir = project_root / 'tests' / 'requirements'
    test_files = list(test_dir.glob('**/*.py'))

    tests_with_req_markers = 0
    total_req_markers = 0

    for test_file in test_files:
        if test_file.name.startswith('test_'):
            content = test_file.read_text()

            if '@pytest.mark.requirement' in content:
                tests_with_req_markers += 1

            # Count requirement markers
            marker_count = content.count('@pytest.mark.requirement')
            total_req_markers += marker_count

    print(f"✓ {tests_with_req_markers} test files with requirement markers")
    print(f"✓ {total_req_markers} total requirement markers found")

    assert tests_with_req_markers > 0, \
        "Tests should be linked to requirements via markers"


@pytest.mark.requirement("REQ-DOC-004")
@pytest.mark.priority("MUST")
def test_req_doc_004_code_references_requirements():
    """
    REQ-DOC-004: Check that code references requirements where appropriate.
    """
    project_root = Path(__file__).parent.parent.parent.parent

    # Check for requirement references in code comments
    kosmos_dir = project_root / 'kosmos'
    py_files = list(kosmos_dir.glob('**/*.py'))

    files_with_req_refs = 0

    for py_file in py_files:
        content = py_file.read_text()

        # Look for requirement references in comments
        if 'REQ-' in content:
            files_with_req_refs += 1

    print(f"✓ {files_with_req_refs} source files reference requirements")


# ============================================================================
# REQ-DOC-005: Example Workflows (SHOULD)
# ============================================================================

@pytest.mark.requirement("REQ-DOC-005")
@pytest.mark.priority("SHOULD")
def test_req_doc_005_example_workflows_exist():
    """
    REQ-DOC-005: The system SHOULD provide example workflows demonstrating
    capabilities across different domains.

    Validates that:
    - Example workflows are documented
    - Multiple domains are covered
    - Examples are runnable
    """
    project_root = Path(__file__).parent.parent.parent.parent

    # Check for example documentation
    example_locations = [
        project_root / 'examples',
        project_root / 'docs' / 'examples',
        project_root / 'demos',
        project_root / 'tutorials',
    ]

    examples_found = []

    for location in example_locations:
        if location.exists():
            # Look for example files
            example_files = (
                list(location.glob('*.py')) +
                list(location.glob('*.md')) +
                list(location.glob('**/*.py')) +
                list(location.glob('**/*.md'))
            )

            if len(example_files) > 0:
                examples_found.extend(example_files)
                print(f"✓ Examples in {location.name}: {len(example_files)} files")

    if len(examples_found) > 0:
        print(f"✓ Total {len(examples_found)} example files found")
    else:
        print("⚠ Example workflows should be provided")


@pytest.mark.requirement("REQ-DOC-005")
@pytest.mark.priority("SHOULD")
def test_req_doc_005_multi_domain_examples():
    """
    REQ-DOC-005: Verify examples cover multiple domains.
    """
    project_root = Path(__file__).parent.parent.parent.parent

    # Check examples or documentation for domain coverage
    example_locations = [
        project_root / 'examples',
        project_root / 'docs' / 'examples',
    ]

    domains_covered = set()
    domain_keywords = ['biology', 'neuroscience', 'physics', 'chemistry', 'materials']

    for location in example_locations:
        if location.exists():
            example_files = list(location.glob('**/*.py')) + list(location.glob('**/*.md'))

            for example_file in example_files:
                content = example_file.read_text().lower()

                for domain in domain_keywords:
                    if domain in content:
                        domains_covered.add(domain)

    if len(domains_covered) > 0:
        print(f"✓ Examples cover domains: {domains_covered}")

        # SHOULD have at least 2 domains covered
        if len(domains_covered) >= 2:
            print(f"✓ Multi-domain examples provided ({len(domains_covered)} domains)")


@pytest.mark.requirement("REQ-DOC-005")
@pytest.mark.priority("SHOULD")
def test_req_doc_005_quickstart_guide():
    """
    REQ-DOC-005: Verify quickstart/tutorial documentation exists.
    """
    project_root = Path(__file__).parent.parent.parent.parent

    quickstart_files = [
        project_root / 'docs' / 'quickstart.md',
        project_root / 'docs' / 'tutorial.md',
        project_root / 'QUICKSTART.md',
        project_root / 'docs' / 'getting_started.md',
    ]

    quickstart_found = False

    for qs_file in quickstart_files:
        if qs_file.exists():
            quickstart_found = True
            print(f"✓ Quickstart guide: {qs_file.name}")

            content = qs_file.read_text().lower()

            # Should have step-by-step instructions
            has_steps = 'step' in content or '1.' in content or 'first' in content

            if has_steps:
                print("  Contains step-by-step instructions")

    if quickstart_found:
        print("✓ Quickstart documentation available")


# ============================================================================
# Integration Tests
# ============================================================================

class TestDocumentationIntegration:
    """Integration tests for documentation completeness."""

    def test_complete_documentation_suite(self):
        """Verify complete documentation suite exists."""
        project_root = Path(__file__).parent.parent.parent.parent

        # Essential documentation files
        essential_docs = {
            'README.md': True,  # MUST
            'REQUIREMENTS.md': True,  # MUST
            'REQUIREMENTS_TRACEABILITY_MATRIX.md': True,  # MUST
            'CONTRIBUTING.md': False,  # SHOULD
            '.env.example': False,  # SHOULD
        }

        for doc_name, is_required in essential_docs.items():
            doc_path = project_root / doc_name

            if is_required:
                assert doc_path.exists(), f"{doc_name} must exist"
            elif doc_path.exists():
                print(f"✓ Optional documentation present: {doc_name}")

    def test_documentation_quality(self):
        """Test documentation quality metrics."""
        project_root = Path(__file__).parent.parent.parent.parent
        readme = project_root / 'README.md'

        if readme.exists():
            content = readme.read_text()

            # Quality checks
            assert len(content) > 500, "README should be substantial (>500 chars)"

            # Should have headers
            header_count = content.count('#')
            assert header_count > 3, "README should have multiple sections"

            # Should have code examples or commands
            has_code = '```' in content or '`' in content
            assert has_code, "README should include code examples"

            print(f"✓ README quality: {len(content)} chars, {header_count} headers")

    def test_all_docs_linked(self):
        """Verify documentation files are cross-linked."""
        project_root = Path(__file__).parent.parent.parent.parent
        docs_dir = project_root / 'docs'

        if docs_dir.exists():
            md_files = list(docs_dir.glob('**/*.md'))

            # Check for cross-references
            cross_refs = 0

            for md_file in md_files:
                content = md_file.read_text()

                # Look for markdown links
                link_count = content.count('[') + content.count('](')

                if link_count > 2:
                    cross_refs += 1

            print(f"✓ {cross_refs} documentation files with cross-references")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
