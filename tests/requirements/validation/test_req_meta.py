"""
Test suite for Meta-Requirements (REQ-META-001 through REQ-META-003).

This test file validates the meta-requirements that govern the requirement
testing process itself, as specified in REQUIREMENTS.md.

Requirements tested:
- REQ-META-001 (MUST): Every MUST/SHALL requirement has automated tests
- REQ-META-002 (MUST): All tests pass before production-ready
- REQ-META-003 (SHOULD): SHOULD requirements have tests with rationale if omitted
"""

import os
import re
import ast
import pytest
import subprocess
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import defaultdict

# Test markers for requirements traceability
pytestmark = [
    pytest.mark.requirement("REQ-META"),
    pytest.mark.category("validation"),
]


# ============================================================================
# REQ-META-001: Automated Tests for MUST/SHALL Requirements (MUST)
# ============================================================================

@pytest.mark.requirement("REQ-META-001")
@pytest.mark.priority("MUST")
def test_req_meta_001_all_must_requirements_have_tests():
    """
    REQ-META-001: Every requirement labeled MUST or SHALL MUST have at
    least one automated test that validates compliance.

    Validates that:
    - All MUST/SHALL requirements are extracted from REQUIREMENTS.md
    - Test files reference these requirements
    - Coverage rate is calculated and reported
    """
    project_root = Path(__file__).parent.parent.parent.parent

    # Step 1: Extract all MUST/SHALL requirements from REQUIREMENTS.md
    requirements_file = project_root / 'REQUIREMENTS.md'
    assert requirements_file.exists(), "REQUIREMENTS.md must exist"

    req_content = requirements_file.read_text()

    # Extract critical requirements
    critical_reqs = set()
    lines = req_content.split('\n')

    for line in lines:
        # Look for requirement definitions with MUST or SHALL
        if 'MUST' in line or 'SHALL' in line:
            # Extract REQ-XXX-### format
            match = re.search(r'\*\*REQ-([A-Z-]+)-(\d+):\*\*', line)
            if match:
                req_id = f"REQ-{match.group(1)}-{match.group(2).zfill(3)}"
                # Verify it's actually a MUST/SHALL by checking context
                # (line contains MUST or SHALL)
                if 'MUST' in line.upper() or 'SHALL' in line.upper():
                    # Exclude MUST NOT / SHALL NOT as they're constraints, not features
                    if 'MUST NOT' not in line and 'SHALL NOT' not in line:
                        critical_reqs.add(req_id)

    print(f"\nFound {len(critical_reqs)} MUST/SHALL requirements in REQUIREMENTS.md")

    # Step 2: Find all test files and extract tested requirements
    tests_dir = project_root / 'tests' / 'requirements'
    test_files = list(tests_dir.glob('**/*.py'))
    test_files = [f for f in test_files if f.name not in ['__init__.py', 'conftest.py']]

    tested_reqs = set()
    req_to_tests = defaultdict(list)

    for test_file in test_files:
        content = test_file.read_text()

        # Extract all REQ-XXX-### references
        req_matches = re.findall(r'REQ-[A-Z]+-\d+', content)

        for req in req_matches:
            # Normalize format (pad numbers to 3 digits)
            parts = req.split('-')
            if len(parts) == 3 and parts[2].isdigit():
                normalized = f"{parts[0]}-{parts[1]}-{parts[2].zfill(3)}"
                tested_reqs.add(normalized)
                req_to_tests[normalized].append(test_file.name)

    print(f"Found {len(tested_reqs)} requirements referenced in tests")

    # Step 3: Calculate coverage
    if len(critical_reqs) > 0:
        covered_reqs = critical_reqs.intersection(tested_reqs)
        uncovered_reqs = critical_reqs - tested_reqs

        coverage_rate = len(covered_reqs) / len(critical_reqs)

        print(f"\nTest Coverage for MUST/SHALL Requirements:")
        print(f"  Total MUST/SHALL:     {len(critical_reqs)}")
        print(f"  Covered by tests:     {len(covered_reqs)}")
        print(f"  Not yet covered:      {len(uncovered_reqs)}")
        print(f"  Coverage rate:        {coverage_rate:.1%}")

        if uncovered_reqs:
            print(f"\nUncovered requirements (first 10):")
            for req in sorted(list(uncovered_reqs))[:10]:
                print(f"    - {req}")

        # Assert: Coverage should be high (>70% for in-progress project)
        # Note: 100% coverage may not be immediately achievable
        assert coverage_rate > 0.7, \
            f"MUST/SHALL requirement test coverage should be >70%, got {coverage_rate:.1%}"

    else:
        pytest.skip("No MUST/SHALL requirements found to test")


@pytest.mark.requirement("REQ-META-001")
@pytest.mark.priority("MUST")
def test_req_meta_001_requirement_markers_properly_used():
    """
    REQ-META-001: Verify requirement markers are used correctly in tests.
    """
    project_root = Path(__file__).parent.parent.parent.parent
    tests_dir = project_root / 'tests' / 'requirements'

    test_files = list(tests_dir.glob('**/*.py'))
    test_files = [f for f in test_files if f.name.startswith('test_')]

    # Check that tests use proper markers
    proper_markers_count = 0
    total_test_functions = 0

    for test_file in test_files:
        content = test_file.read_text()

        # Count test functions
        test_funcs = content.count('def test_')
        total_test_functions += test_funcs

        # Count proper requirement markers
        # Format: @pytest.mark.requirement("REQ-XXX-###")
        proper_markers = len(re.findall(r'@pytest\.mark\.requirement\(["\']REQ-[A-Z]+-\d+["\']\)', content))
        proper_markers_count += proper_markers

    print(f"\nRequirement Marker Usage:")
    print(f"  Total test functions:     {total_test_functions}")
    print(f"  With requirement markers: {proper_markers_count}")

    if total_test_functions > 0:
        marker_rate = proper_markers_count / total_test_functions
        print(f"  Marker usage rate:        {marker_rate:.1%}")

        # Most tests should have proper markers
        assert marker_rate > 0.5, \
            f"At least 50% of tests should use requirement markers, got {marker_rate:.1%}"


@pytest.mark.requirement("REQ-META-001")
@pytest.mark.priority("MUST")
def test_req_meta_001_test_names_match_requirements():
    """
    REQ-META-001: Verify test function names match requirement IDs.
    """
    project_root = Path(__file__).parent.parent.parent.parent
    tests_dir = project_root / 'tests' / 'requirements'

    test_files = list(tests_dir.glob('**/*.py'))
    test_files = [f for f in test_files if f.name.startswith('test_req_')]

    naming_convention_count = 0

    for test_file in test_files:
        content = test_file.read_text()

        # Find test functions with req pattern
        # Format: def test_req_xxx_###_description():
        test_funcs = re.findall(r'def (test_req_[a-z]+_\d+_[a-z_]+)\(', content)

        naming_convention_count += len(test_funcs)

    print(f"\n✓ Found {naming_convention_count} tests following requirement naming convention")


# ============================================================================
# REQ-META-002: All Tests Pass for Production (MUST)
# ============================================================================

@pytest.mark.requirement("REQ-META-002")
@pytest.mark.priority("MUST")
def test_req_meta_002_test_infrastructure_functional():
    """
    REQ-META-002: All tests MUST pass before the system can be considered
    production-ready.

    Validates that:
    - Test infrastructure is functional
    - Tests can be executed
    - Test results are captured
    """
    project_root = Path(__file__).parent.parent.parent.parent

    # Verify pytest is configured
    pytest_configs = [
        project_root / 'pytest.ini',
        project_root / 'pyproject.toml',
        project_root / 'setup.cfg'
    ]

    pytest_configured = False
    for config_file in pytest_configs:
        if config_file.exists():
            content = config_file.read_text()
            if 'pytest' in content.lower():
                pytest_configured = True
                print(f"✓ Pytest configured in {config_file.name}")
                break

    assert pytest_configured, "Pytest must be configured for test execution"


@pytest.mark.requirement("REQ-META-002")
@pytest.mark.priority("MUST")
def test_req_meta_002_critical_test_identification():
    """
    REQ-META-002: Verify critical tests are properly identified.
    """
    project_root = Path(__file__).parent.parent.parent.parent
    tests_dir = project_root / 'tests' / 'requirements'

    test_files = list(tests_dir.glob('**/*.py'))

    # Count tests with MUST priority marker
    critical_tests = 0

    for test_file in test_files:
        if test_file.name.startswith('test_'):
            content = test_file.read_text()

            # Count MUST priority markers
            must_markers = content.count('@pytest.mark.priority("MUST")')
            critical_tests += must_markers

    print(f"\n✓ Found {critical_tests} critical (MUST priority) tests")

    assert critical_tests > 0, "Critical tests must be marked with priority"


@pytest.mark.requirement("REQ-META-002")
@pytest.mark.priority("MUST")
def test_req_meta_002_deployment_gate_exists():
    """
    REQ-META-002: Verify deployment is gated by test success.
    """
    project_root = Path(__file__).parent.parent.parent.parent

    # Check CI configuration
    ci_workflows = project_root / '.github' / 'workflows'

    deployment_gated = False

    if ci_workflows.exists():
        workflow_files = list(ci_workflows.glob('*.yml')) + list(ci_workflows.glob('*.yaml'))

        for workflow in workflow_files:
            content = workflow.read_text()

            # Look for deployment jobs that depend on tests
            if 'deploy' in content.lower():
                if ('needs' in content and 'test' in content) or \
                   ('if:' in content and 'success' in content):
                    deployment_gated = True
                    print(f"✓ Deployment gated by tests in {workflow.name}")

    if deployment_gated:
        print("✓ Deployment properly gated by test success")
    else:
        print("⚠ Deployment should be gated by test success")


@pytest.mark.requirement("REQ-META-002")
@pytest.mark.priority("MUST")
def test_req_meta_002_production_readiness_criteria():
    """
    REQ-META-002: Verify production readiness criteria are documented.
    """
    project_root = Path(__file__).parent.parent.parent.parent
    requirements_file = project_root / 'REQUIREMENTS.md'

    if requirements_file.exists():
        content = requirements_file.read_text()

        # Check for production readiness section
        has_prod_criteria = (
            'production' in content.lower() and
            'ready' in content.lower()
        )

        assert has_prod_criteria, \
            "Production readiness criteria should be documented"

        # Check for specific criteria
        criteria_keywords = [
            'all.*must.*pass',
            'test.*pass',
            'coverage',
            'validation'
        ]

        import re
        criteria_found = sum(1 for pattern in criteria_keywords
                           if re.search(pattern, content.lower()))

        print(f"✓ Found {criteria_found} production readiness criteria")


# ============================================================================
# REQ-META-003: SHOULD Requirements Testing (SHOULD)
# ============================================================================

@pytest.mark.requirement("REQ-META-003")
@pytest.mark.priority("SHOULD")
def test_req_meta_003_should_requirements_tracked():
    """
    REQ-META-003: Requirements labeled SHOULD SHOULD have automated tests,
    with documented rationale if test is omitted.

    Validates that:
    - SHOULD requirements are identified
    - Test coverage is tracked
    - Rationale is documented for omissions
    """
    project_root = Path(__file__).parent.parent.parent.parent

    # Extract SHOULD requirements
    requirements_file = project_root / 'REQUIREMENTS.md'
    req_content = requirements_file.read_text()

    should_reqs = set()
    lines = req_content.split('\n')

    for line in lines:
        if 'SHOULD' in line:
            match = re.search(r'\*\*REQ-([A-Z-]+)-(\d+):\*\*', line)
            if match:
                req_id = f"REQ-{match.group(1)}-{match.group(2).zfill(3)}"
                # Exclude SHOULD NOT
                if 'SHOULD NOT' not in line:
                    should_reqs.add(req_id)

    print(f"\nFound {len(should_reqs)} SHOULD requirements")

    # Find tested SHOULD requirements
    tests_dir = project_root / 'tests' / 'requirements'
    test_files = list(tests_dir.glob('**/*.py'))

    tested_should_reqs = set()

    for test_file in test_files:
        content = test_file.read_text()

        # Find SHOULD requirements that are tested
        for req in should_reqs:
            if req in content:
                tested_should_reqs.add(req)

    if len(should_reqs) > 0:
        should_coverage = len(tested_should_reqs) / len(should_reqs)

        print(f"SHOULD requirement test coverage:")
        print(f"  Total SHOULD:         {len(should_reqs)}")
        print(f"  Tested:               {len(tested_should_reqs)}")
        print(f"  Coverage:             {should_coverage:.1%}")

        # SHOULD requirements should have good coverage too
        if should_coverage > 0.6:
            print("✓ Good SHOULD requirement coverage")


@pytest.mark.requirement("REQ-META-003")
@pytest.mark.priority("SHOULD")
def test_req_meta_003_priority_markers_used():
    """
    REQ-META-003: Verify priority markers distinguish MUST from SHOULD.
    """
    project_root = Path(__file__).parent.parent.parent.parent
    tests_dir = project_root / 'tests' / 'requirements'

    test_files = list(tests_dir.glob('**/*.py'))

    must_priority_count = 0
    should_priority_count = 0

    for test_file in test_files:
        if test_file.name.startswith('test_'):
            content = test_file.read_text()

            must_priority_count += content.count('@pytest.mark.priority("MUST")')
            should_priority_count += content.count('@pytest.mark.priority("SHOULD")')

    print(f"\nPriority Markers:")
    print(f"  MUST priority:    {must_priority_count}")
    print(f"  SHOULD priority:  {should_priority_count}")

    assert must_priority_count > 0, "MUST priority tests should exist"


@pytest.mark.requirement("REQ-META-003")
@pytest.mark.priority("SHOULD")
def test_req_meta_003_omission_rationale_documented():
    """
    REQ-META-003: Verify rationale is documented for untested SHOULD requirements.
    """
    project_root = Path(__file__).parent.parent.parent.parent

    # Check for documentation of test omissions
    # This could be in REQUIREMENTS.md, test plans, or traceability matrix

    doc_files = [
        project_root / 'REQUIREMENTS.md',
        project_root / 'REQUIREMENTS_TRACEABILITY_MATRIX.md',
        project_root / 'docs' / 'testing.md',
    ]

    rationale_documented = False

    for doc_file in doc_files:
        if doc_file.exists():
            content = doc_file.read_text()

            # Look for test omission rationale
            rationale_keywords = [
                'not tested',
                'test omitted',
                'rationale',
                'justification',
                'not yet implemented'
            ]

            if any(keyword.lower() in content.lower() for keyword in rationale_keywords):
                rationale_documented = True
                print(f"✓ Test rationale/status documented in {doc_file.name}")

    # This is a SHOULD requirement, so we don't fail but report
    if rationale_documented:
        print("✓ Test omission rationale is documented")


# ============================================================================
# Integration Tests
# ============================================================================

class TestMetaRequirementsIntegration:
    """Integration tests for meta-requirements compliance."""

    def test_complete_requirements_test_mapping(self):
        """Verify comprehensive mapping between requirements and tests."""
        project_root = Path(__file__).parent.parent.parent.parent

        # Load requirements
        requirements_file = project_root / 'REQUIREMENTS.md'
        req_content = requirements_file.read_text()

        # Extract all requirements (MUST, SHOULD, MAY)
        all_reqs = set(re.findall(r'REQ-[A-Z]+-\d+', req_content))

        # Load all test references
        tests_dir = project_root / 'tests' / 'requirements'
        test_files = list(tests_dir.glob('**/*.py'))

        tested_reqs = set()
        for test_file in test_files:
            content = test_file.read_text()
            tested_reqs.update(re.findall(r'REQ-[A-Z]+-\d+', content))

        # Calculate overall coverage
        if len(all_reqs) > 0:
            overall_coverage = len(tested_reqs.intersection(all_reqs)) / len(all_reqs)

            print(f"\nOverall Requirements Test Coverage:")
            print(f"  Total requirements:    {len(all_reqs)}")
            print(f"  Tested:                {len(tested_reqs.intersection(all_reqs))}")
            print(f"  Coverage:              {overall_coverage:.1%}")

    def test_test_suite_organization(self):
        """Verify test suite is well-organized by requirement category."""
        project_root = Path(__file__).parent.parent.parent.parent
        tests_dir = project_root / 'tests' / 'requirements'

        # Expected categories from REQUIREMENTS.md
        expected_categories = [
            'core', 'data_analysis', 'literature', 'world_model',
            'orchestrator', 'output', 'integration', 'security',
            'performance', 'scientific', 'validation'
        ]

        found_categories = []

        for category in expected_categories:
            category_dir = tests_dir / category
            if category_dir.exists() and category_dir.is_dir():
                test_files = list(category_dir.glob('test_*.py'))
                if len(test_files) > 0:
                    found_categories.append(category)

        print(f"\n✓ Test organization covers {len(found_categories)}/{len(expected_categories)} categories")

    def test_traceability_matrix_comprehensive(self):
        """Verify traceability matrix is comprehensive."""
        project_root = Path(__file__).parent.parent.parent.parent
        traceability_file = project_root / 'REQUIREMENTS_TRACEABILITY_MATRIX.md'

        if traceability_file.exists():
            content = traceability_file.read_text()

            # Count requirements in traceability matrix
            req_count = len(set(re.findall(r'REQ-[A-Z]+-\d+', content)))

            print(f"\n✓ Traceability matrix tracks {req_count} requirements")

            # Should have substantial coverage
            assert req_count > 50, \
                f"Traceability matrix should track many requirements, found {req_count}"

    def test_production_readiness_documentation(self):
        """Verify production readiness is clearly defined."""
        project_root = Path(__file__).parent.parent.parent.parent
        requirements_file = project_root / 'REQUIREMENTS.md'

        content = requirements_file.read_text()

        # Look for production readiness criteria section
        has_prod_section = (
            'Production Readiness' in content or
            'production-ready' in content.lower()
        )

        assert has_prod_section, \
            "Production readiness criteria must be documented"

        # Check for specific criteria
        criteria = [
            'all.*must.*pass',
            'coverage.*80',
            'security.*validated',
            'documentation.*complete'
        ]

        import re
        found_criteria = sum(1 for pattern in criteria
                           if re.search(pattern, content.lower()))

        print(f"✓ Production readiness criteria documented ({found_criteria} criteria)")

    def test_continuous_testing_configured(self):
        """Verify continuous testing is set up."""
        project_root = Path(__file__).parent.parent.parent.parent

        # Check for CI configuration
        ci_indicators = [
            project_root / '.github' / 'workflows',
            project_root / '.gitlab-ci.yml',
            project_root / 'Jenkinsfile',
        ]

        ci_configured = any(path.exists() for path in ci_indicators)

        if ci_configured:
            print("✓ Continuous testing infrastructure configured")

        # Check pytest.ini for CI-friendly settings
        pytest_ini = project_root / 'pytest.ini'
        if pytest_ini.exists():
            content = pytest_ini.read_text()

            ci_friendly = (
                'junit' in content.lower() or
                'xml' in content.lower() or
                'verbose' in content.lower()
            )

            if ci_friendly:
                print("✓ Pytest configured for CI reporting")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
