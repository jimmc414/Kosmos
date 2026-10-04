#!/usr/bin/env python3
"""
Smoke Test Suite for Kosmos.

Quick tests to verify basic functionality of all components.
Run time target: < 30 seconds.
"""

import asyncio
import sys
from pathlib import Path


def test_imports():
    """Test that all critical modules import correctly."""
    print("Testing imports...")

    modules = [
        ("kosmos.compression.compressor", "ContextCompressor"),
        ("kosmos.world_model.artifacts", "ArtifactStateManager"),
        ("kosmos.world_model.artifacts", "Finding"),
        ("kosmos.orchestration.plan_creator", "PlanCreatorAgent"),
        ("kosmos.orchestration.plan_reviewer", "PlanReviewerAgent"),
        ("kosmos.orchestration.delegation", "DelegationManager"),
        ("kosmos.orchestration.novelty_detector", "NoveltyDetector"),
        ("kosmos.agents.skill_loader", "SkillLoader"),
        ("kosmos.validation.scholar_eval", "ScholarEvalValidator"),
        ("kosmos.workflow.research_loop", "ResearchWorkflow"),
    ]

    for module_name, class_name in modules:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"  [OK] {module_name}.{class_name}")
        except ImportError as e:
            print(f"  [FAIL] {module_name}.{class_name}: {e}")
            return False

    return True


def test_compression():
    """Test context compression."""
    print("\nTesting compression...")
    from kosmos.compression.compressor import NotebookCompressor, ContextCompressor

    # Test NotebookCompressor
    compressor = NotebookCompressor()
    stats = compressor._extract_statistics("p = 0.001, n = 100, r = 0.85")
    assert 'p_value' in stats, "P-value extraction failed"
    assert stats['sample_size'] == 100, "Sample size extraction failed"
    print("  [OK] NotebookCompressor statistics extraction")

    # Test ContextCompressor
    ctx = ContextCompressor()
    result = ctx.compress_cycle_results(1, [{'type': 'generic', 'content': 'test'}])
    assert result.summary is not None, "Cycle compression failed"
    print("  [OK] ContextCompressor cycle compression")

    return True


async def test_state_manager_async():
    """Test state manager operations."""
    print("\nTesting state manager...")
    import tempfile
    from kosmos.world_model.artifacts import ArtifactStateManager, Finding

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ArtifactStateManager(artifacts_dir=tmpdir)

        # Test finding storage
        finding = Finding(
            finding_id="test_1",
            cycle=1,
            task_id=1,
            summary="Test finding",
            statistics={"p_value": 0.01}
        )
        await manager.save_finding_artifact(1, 1, finding.to_dict())

        # Verify retrieval
        findings = manager.get_all_findings()
        assert len(findings) == 1, "Finding retrieval failed"
        print("  [OK] ArtifactStateManager finding storage")

        # Test statistics
        stats = manager.get_statistics()
        assert 'total_findings' in stats, "Statistics generation failed"
        print("  [OK] ArtifactStateManager statistics")

    return True


def test_state_manager():
    """Wrapper for async state manager test."""
    return asyncio.run(test_state_manager_async())


def test_orchestration():
    """Test orchestration components."""
    print("\nTesting orchestration...")
    from kosmos.orchestration.plan_creator import PlanCreatorAgent
    from kosmos.orchestration.plan_reviewer import PlanReviewerAgent
    from kosmos.orchestration.novelty_detector import NoveltyDetector

    # Test plan creator
    creator = PlanCreatorAgent()
    plan = creator._create_mock_plan(1, "Test objective", {}, 5, 0.7)
    assert len(plan.tasks) == 5, "Plan creation failed"
    print("  [OK] PlanCreatorAgent mock plan creation")

    # Test plan reviewer
    reviewer = PlanReviewerAgent()
    plan_dict = {'tasks': plan.to_dict()['tasks']}
    is_valid = reviewer._meets_structural_requirements(plan_dict)
    assert is_valid, "Structural validation failed"
    print("  [OK] PlanReviewerAgent structural validation")

    # Test novelty detector
    detector = NoveltyDetector()
    past_tasks = [
        {"description": "Task 1: Analyze data"},
        {"description": "Task 2: Review literature"}
    ]
    detector.index_past_tasks(past_tasks)
    result = detector.check_task_novelty({"description": "Task 3: Analyze similar data"})
    assert 'novelty_score' in result, "Novelty result missing score"
    assert 0 <= result['novelty_score'] <= 1, "Novelty score out of range"
    print("  [OK] NoveltyDetector novelty scoring")

    return True


async def test_validation_async():
    """Test ScholarEval validation."""
    print("\nTesting validation...")
    from kosmos.validation.scholar_eval import ScholarEvalValidator

    validator = ScholarEvalValidator(allow_mock=True)
    finding = {
        "summary": "KRAS mutation correlates with poor outcomes",
        "statistics": {"p_value": 0.001, "sample_size": 500},
        "methods": "Statistical analysis"
    }

    score = validator.evaluate_finding(finding)
    assert hasattr(score, 'overall_score'), "ScholarEval scoring failed"
    assert 0 <= score.overall_score <= 10, "Score out of range"
    print("  [OK] ScholarEvalValidator mock scoring")

    return True


def test_validation():
    """Wrapper for async validation test."""
    return asyncio.run(test_validation_async())


async def test_workflow():
    """Test workflow execution."""
    print("\nTesting workflow...")
    import tempfile
    from kosmos.workflow.research_loop import ResearchWorkflow

    with tempfile.TemporaryDirectory() as tmpdir:
        workflow = ResearchWorkflow(
            research_objective="Test objective",
            artifacts_dir=tmpdir
        )

        result = await workflow.run(num_cycles=1, tasks_per_cycle=5)

        assert result['cycles_completed'] == 1, "Workflow cycle failed"
        assert result['total_tasks_generated'] >= 5, "Task generation failed"
        print("  [OK] ResearchWorkflow single-cycle execution")

    return True


def test_skill_loader():
    """Test skill loader."""
    print("\nTesting skill loader...")
    from kosmos.agents.skill_loader import SkillLoader

    # Test without auto-discover
    loader = SkillLoader(auto_discover=False, skills_dir=None)
    assert loader.skills_dir is None, "Skill loader init failed"
    print("  [OK] SkillLoader initialization")

    # Test bundle lookup
    bundle = loader.SKILL_BUNDLES.get('single_cell_analysis', [])
    assert 'scanpy' in bundle, "Bundle lookup failed"
    print("  [OK] SkillLoader bundle configuration")

    return True


def main():
    """Run all smoke tests."""
    print("=" * 50)
    print("KOSMOS SMOKE TEST SUITE")
    print("=" * 50)

    tests = [
        ("Imports", test_imports),
        ("Compression", test_compression),
        ("State Manager", test_state_manager),
        ("Orchestration", test_orchestration),
        ("Validation", test_validation),
        ("Skill Loader", test_skill_loader),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            results.append((name, False))

    # Async test
    try:
        passed = asyncio.run(test_workflow())
        results.append(("Workflow", passed))
    except Exception as e:
        print(f"  [ERROR] Workflow: {e}")
        results.append(("Workflow", False))

    # Summary
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)

    passed = sum(1 for _, p in results if p)
    total = len(results)

    for name, p in results:
        status = "[PASS]" if p else "[FAIL]"
        print(f"  {status} {name}")

    print(f"\nTotal: {passed}/{total} test suites passed")
    print("=" * 50)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
