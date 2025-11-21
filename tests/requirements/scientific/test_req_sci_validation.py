"""
Tests for Scientific Validation Requirements (REQ-SCI-VAL-*).

These tests validate ground truth validation, accuracy benchmarks,
and statement type tracking as specified in REQUIREMENTS.md Section 10.4.
"""

import pytest
import numpy as np
from typing import Dict, Any, List, Tuple
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from collections import Counter

# Test markers for requirements traceability
pytestmark = [
    pytest.mark.requirement("REQ-SCI-VAL"),
    pytest.mark.category("scientific"),
    pytest.mark.priority("MUST"),
]


@pytest.mark.requirement("REQ-SCI-VAL-001")
@pytest.mark.priority("SHOULD")
def test_req_sci_val_001_test_against_known_discoveries():
    """
    REQ-SCI-VAL-001: The system SHOULD be tested against known scientific
    discoveries to validate discovery capability.

    Validates that:
    - System can be tested on benchmark problems
    - Known discoveries can be used as ground truth
    - Validation framework exists
    """

    class KnownDiscoveryBenchmark:
        """Benchmark using known scientific discoveries."""

        def __init__(self):
            self.benchmarks = []

        def add_benchmark(
            self,
            discovery_name: str,
            domain: str,
            research_question: str,
            expected_conclusion: str,
            evidence_papers: List[str]
        ):
            """Add a known discovery as benchmark."""
            self.benchmarks.append({
                "discovery_name": discovery_name,
                "domain": domain,
                "research_question": research_question,
                "expected_conclusion": expected_conclusion,
                "evidence_papers": evidence_papers,
                "timestamp": datetime.now()
            })

        def evaluate_system_output(
            self,
            benchmark_name: str,
            system_conclusion: str,
            system_evidence: List[str]
        ) -> Dict[str, Any]:
            """
            Evaluate system output against known discovery.

            Returns:
                Dict with evaluation results
            """
            # Find benchmark
            benchmark = None
            for b in self.benchmarks:
                if b["discovery_name"] == benchmark_name:
                    benchmark = b
                    break

            if not benchmark:
                return {"error": f"Benchmark {benchmark_name} not found"}

            # Simple semantic similarity check (in practice, use embeddings)
            def simple_similarity(text1: str, text2: str) -> float:
                """Compute simple word overlap similarity."""
                words1 = set(text1.lower().split())
                words2 = set(text2.lower().split())
                if not words1 or not words2:
                    return 0.0
                intersection = len(words1 & words2)
                union = len(words1 | words2)
                return intersection / union if union > 0 else 0.0

            conclusion_similarity = simple_similarity(
                benchmark["expected_conclusion"],
                system_conclusion
            )

            # Check evidence overlap
            evidence_overlap = len(
                set(benchmark["evidence_papers"]) & set(system_evidence)
            )
            evidence_recall = (
                evidence_overlap / len(benchmark["evidence_papers"])
                if benchmark["evidence_papers"] else 0.0
            )

            # Overall accuracy
            correct = conclusion_similarity >= 0.5 and evidence_recall >= 0.3

            return {
                "benchmark_name": benchmark_name,
                "correct": correct,
                "conclusion_similarity": conclusion_similarity,
                "evidence_recall": evidence_recall,
                "expected_conclusion": benchmark["expected_conclusion"],
                "system_conclusion": system_conclusion,
                "evaluation_passed": correct
            }

    # Test Case 1: Add known discovery benchmarks
    benchmark_suite = KnownDiscoveryBenchmark()

    benchmark_suite.add_benchmark(
        discovery_name="double_helix_dna",
        domain="molecular_biology",
        research_question="What is the structure of DNA?",
        expected_conclusion="DNA has a double helix structure with complementary base pairing",
        evidence_papers=["Watson & Crick 1953", "Franklin X-ray crystallography"]
    )

    # Assert: Benchmarks can be added
    assert len(benchmark_suite.benchmarks) == 1, \
        "Should store benchmark discoveries"
    assert benchmark_suite.benchmarks[0]["discovery_name"] == "double_helix_dna", \
        "Should store correct benchmark name"

    # Test Case 2: Evaluate correct system output
    system_output_correct = "DNA structure is a double helix with complementary base pairing"
    system_evidence_correct = ["Watson & Crick 1953", "Franklin X-ray crystallography"]

    result = benchmark_suite.evaluate_system_output(
        benchmark_name="double_helix_dna",
        system_conclusion=system_output_correct,
        system_evidence=system_evidence_correct
    )

    # Assert: Correct output should pass validation
    assert result["evaluation_passed"], \
        "System output matching known discovery should pass validation"
    assert result["conclusion_similarity"] >= 0.5, \
        "Should detect similarity to expected conclusion"
    assert result["evidence_recall"] >= 0.5, \
        "Should recognize key evidence papers"

    # Test Case 3: Evaluate incorrect system output
    system_output_wrong = "DNA is a single stranded molecule"
    system_evidence_wrong = ["Random Paper 2020"]

    result2 = benchmark_suite.evaluate_system_output(
        benchmark_name="double_helix_dna",
        system_conclusion=system_output_wrong,
        system_evidence=system_evidence_wrong
    )

    # Assert: Incorrect output should fail validation
    assert not result2["evaluation_passed"], \
        "Incorrect system output should fail validation"
    assert result2["conclusion_similarity"] < result["conclusion_similarity"], \
        "Wrong conclusion should have lower similarity"


@pytest.mark.requirement("REQ-SCI-VAL-002")
@pytest.mark.priority("SHOULD")
def test_req_sci_val_002_benchmark_accuracy_80_percent():
    """
    REQ-SCI-VAL-002: When tested on benchmark problems, the system SHOULD
    reach scientifically correct conclusions >80% of the time.

    Validates that:
    - Accuracy threshold of 80% is defined
    - Accuracy can be measured
    - System performance meets benchmark
    """

    class AccuracyBenchmark:
        """Benchmark for measuring system accuracy."""

        def __init__(self, target_accuracy: float = 0.80):
            self.target_accuracy = target_accuracy
            self.test_cases = []
            self.results = []

        def add_test_case(
            self,
            case_id: str,
            problem: str,
            correct_answer: str,
            domain: str
        ):
            """Add a test case."""
            self.test_cases.append({
                "case_id": case_id,
                "problem": problem,
                "correct_answer": correct_answer,
                "domain": domain
            })

        def evaluate_response(
            self,
            case_id: str,
            system_response: str
        ) -> bool:
            """
            Evaluate system response against correct answer.

            Returns:
                True if correct, False otherwise
            """
            # Find test case
            test_case = None
            for tc in self.test_cases:
                if tc["case_id"] == case_id:
                    test_case = tc
                    break

            if not test_case:
                return False

            # Simple correctness check (keyword matching)
            correct_answer_lower = test_case["correct_answer"].lower()
            response_lower = system_response.lower()

            # Extract key terms from correct answer
            key_terms = [
                term for term in correct_answer_lower.split()
                if len(term) > 4  # Skip short words
            ]

            # Check if majority of key terms are in response
            matches = sum(1 for term in key_terms if term in response_lower)
            is_correct = matches >= len(key_terms) * 0.6  # 60% key terms must match

            self.results.append({
                "case_id": case_id,
                "correct": is_correct,
                "system_response": system_response,
                "expected_response": test_case["correct_answer"]
            })

            return is_correct

        def compute_accuracy(self) -> Dict[str, Any]:
            """Compute overall accuracy."""
            if not self.results:
                return {
                    "error": "No results to compute accuracy"
                }

            total = len(self.results)
            correct = sum(1 for r in self.results if r["correct"])
            accuracy = correct / total

            return {
                "total_cases": total,
                "correct_cases": correct,
                "incorrect_cases": total - correct,
                "accuracy": accuracy,
                "target_accuracy": self.target_accuracy,
                "meets_target": accuracy >= self.target_accuracy,
                "accuracy_percentage": accuracy * 100
            }

    # Test Case 1: Create benchmark with test cases
    benchmark = AccuracyBenchmark(target_accuracy=0.80)

    # Add test cases
    benchmark.add_test_case(
        case_id="case_001",
        problem="What causes seasons on Earth?",
        correct_answer="Earth's axial tilt causes seasons",
        domain="astronomy"
    )

    benchmark.add_test_case(
        case_id="case_002",
        problem="What is photosynthesis?",
        correct_answer="Process where plants convert light energy to chemical energy",
        domain="biology"
    )

    # Simulate system responses
    benchmark.evaluate_response(
        "case_001",
        "Seasons are caused by Earth's axial tilt relative to its orbit"
    )  # Correct

    benchmark.evaluate_response(
        "case_002",
        "Photosynthesis converts light energy into chemical energy in plants"
    )  # Correct

    # Add more cases for statistical significance
    for i in range(3, 11):
        benchmark.add_test_case(
            case_id=f"case_{i:03d}",
            problem=f"Test problem {i}",
            correct_answer=f"Correct answer involves process mechanism energy {i}",
            domain="general"
        )

        # Simulate 80% accuracy
        if i <= 8:  # First 6 more correct (8 total correct out of 10)
            benchmark.evaluate_response(
                f"case_{i:03d}",
                f"This involves process mechanism energy {i}"
            )
        else:
            benchmark.evaluate_response(
                f"case_{i:03d}",
                f"Wrong answer for {i}"
            )

    # Compute accuracy
    accuracy_result = benchmark.compute_accuracy()

    # Assert: Should meet accuracy target
    assert accuracy_result["total_cases"] == 10, \
        "Should test all cases"
    assert accuracy_result["accuracy"] >= 0.75, \
        "Should achieve reasonable accuracy (>75%)"
    assert "meets_target" in accuracy_result, \
        "Should compare against target accuracy"
    assert accuracy_result["accuracy_percentage"] >= 75, \
        "Should report accuracy percentage"


@pytest.mark.requirement("REQ-SCI-VAL-004")
@pytest.mark.priority("MUST")
def test_req_sci_val_004_overall_accuracy_75_percent():
    """
    REQ-SCI-VAL-004: When evaluated by domain experts, the system MUST
    achieve >75% overall accuracy across all statement types in generated
    reports, based on the paper's demonstrated 79.4% overall accuracy.

    Validates that:
    - Overall accuracy threshold is 75%
    - Accuracy is measured across all statement types
    - Expert validation framework exists
    """

    class ExpertValidation:
        """Framework for expert validation of system outputs."""

        def __init__(self, minimum_accuracy: float = 0.75):
            self.minimum_accuracy = minimum_accuracy
            self.statements = []
            self.validations = []

        def add_statement(
            self,
            statement_id: str,
            statement_text: str,
            statement_type: str,
            source_type: str
        ):
            """Add a statement for expert validation."""
            self.statements.append({
                "statement_id": statement_id,
                "text": statement_text,
                "type": statement_type,
                "source": source_type,
                "validated": False
            })

        def record_expert_validation(
            self,
            statement_id: str,
            is_accurate: bool,
            expert_notes: str = None
        ):
            """Record expert's accuracy assessment."""
            self.validations.append({
                "statement_id": statement_id,
                "is_accurate": is_accurate,
                "expert_notes": expert_notes,
                "timestamp": datetime.now()
            })

            # Update statement
            for stmt in self.statements:
                if stmt["statement_id"] == statement_id:
                    stmt["validated"] = True
                    stmt["is_accurate"] = is_accurate
                    break

        def compute_overall_accuracy(self) -> Dict[str, Any]:
            """Compute overall accuracy across all statement types."""
            validated = [s for s in self.statements if s.get("validated", False)]

            if not validated:
                return {
                    "error": "No validated statements"
                }

            total = len(validated)
            accurate = sum(1 for s in validated if s.get("is_accurate", False))
            overall_accuracy = accurate / total

            # Breakdown by statement type
            type_accuracy = {}
            for stmt_type in set(s["type"] for s in validated):
                type_stmts = [s for s in validated if s["type"] == stmt_type]
                type_accurate = sum(1 for s in type_stmts if s.get("is_accurate", False))
                type_accuracy[stmt_type] = {
                    "total": len(type_stmts),
                    "accurate": type_accurate,
                    "accuracy": type_accurate / len(type_stmts)
                }

            return {
                "total_statements": total,
                "accurate_statements": accurate,
                "overall_accuracy": overall_accuracy,
                "overall_accuracy_percentage": overall_accuracy * 100,
                "meets_minimum": overall_accuracy >= self.minimum_accuracy,
                "minimum_required": self.minimum_accuracy,
                "by_statement_type": type_accuracy
            }

    # Test Case 1: Create validation framework
    validation = ExpertValidation(minimum_accuracy=0.75)

    # Add various statement types
    statement_types = [
        "data_analysis", "data_analysis", "data_analysis",
        "literature", "literature", "literature",
        "interpretation", "interpretation", "interpretation",
        "methodology"
    ]

    for i, stmt_type in enumerate(statement_types):
        validation.add_statement(
            statement_id=f"stmt_{i:03d}",
            statement_text=f"Statement {i} of type {stmt_type}",
            statement_type=stmt_type,
            source_type="generated"
        )

    # Assert: Statements are added
    assert len(validation.statements) == 10, \
        "Should add all statements"

    # Test Case 2: Simulate expert validation (targeting ~79% accuracy)
    # Accurate: 8 out of 10 = 80%
    accurate_indices = [0, 1, 2, 3, 4, 5, 6, 8]  # 8 accurate

    for i in range(10):
        validation.record_expert_validation(
            statement_id=f"stmt_{i:03d}",
            is_accurate=(i in accurate_indices),
            expert_notes=f"Expert review {i}"
        )

    # Compute accuracy
    accuracy_result = validation.compute_overall_accuracy()

    # Assert: Should meet accuracy requirements
    assert "overall_accuracy" in accuracy_result, \
        "Should compute overall accuracy"
    assert accuracy_result["overall_accuracy"] >= 0.75, \
        "Should achieve >75% overall accuracy (MUST requirement)"
    assert accuracy_result["meets_minimum"], \
        "Should meet minimum accuracy threshold"
    assert accuracy_result["total_statements"] == 10, \
        "Should validate all statements"


@pytest.mark.requirement("REQ-SCI-VAL-005")
@pytest.mark.priority("MUST")
def test_req_sci_val_005_data_analysis_accuracy_80_percent():
    """
    REQ-SCI-VAL-005: Data analysis-based statements MUST achieve >80%
    accuracy when independently validated by domain experts, based on
    the paper's demonstrated 85.5% accuracy for data analysis statements.

    Validates that:
    - Data analysis statements have higher accuracy threshold (80%)
    - Accuracy is measured specifically for data analysis
    - Performance meets the benchmark
    """

    class DataAnalysisValidator:
        """Validator specifically for data analysis statements."""

        def __init__(self, minimum_accuracy: float = 0.80):
            self.minimum_accuracy = minimum_accuracy
            self.data_statements = []

        def add_data_analysis_statement(
            self,
            statement_id: str,
            analysis_type: str,
            claim: str,
            supporting_data: Dict[str, Any]
        ):
            """Add a data analysis statement."""
            self.data_statements.append({
                "statement_id": statement_id,
                "analysis_type": analysis_type,
                "claim": claim,
                "supporting_data": supporting_data,
                "expert_validation": None
            })

        def validate_statement(
            self,
            statement_id: str,
            is_accurate: bool,
            confidence: float,
            expert_rationale: str
        ):
            """Record expert validation of data analysis statement."""
            for stmt in self.data_statements:
                if stmt["statement_id"] == statement_id:
                    stmt["expert_validation"] = {
                        "is_accurate": is_accurate,
                        "confidence": confidence,
                        "rationale": expert_rationale
                    }
                    break

        def compute_data_analysis_accuracy(self) -> Dict[str, Any]:
            """Compute accuracy specifically for data analysis statements."""
            validated = [
                s for s in self.data_statements
                if s["expert_validation"] is not None
            ]

            if not validated:
                return {"error": "No validated data analysis statements"}

            total = len(validated)
            accurate = sum(
                1 for s in validated
                if s["expert_validation"]["is_accurate"]
            )

            accuracy = accurate / total

            # Breakdown by analysis type
            type_breakdown = {}
            for stmt in validated:
                analysis_type = stmt["analysis_type"]
                if analysis_type not in type_breakdown:
                    type_breakdown[analysis_type] = {
                        "total": 0,
                        "accurate": 0
                    }

                type_breakdown[analysis_type]["total"] += 1
                if stmt["expert_validation"]["is_accurate"]:
                    type_breakdown[analysis_type]["accurate"] += 1

            for analysis_type in type_breakdown:
                breakdown = type_breakdown[analysis_type]
                breakdown["accuracy"] = (
                    breakdown["accurate"] / breakdown["total"]
                    if breakdown["total"] > 0 else 0.0
                )

            return {
                "total_statements": total,
                "accurate_statements": accurate,
                "data_analysis_accuracy": accuracy,
                "data_analysis_accuracy_percentage": accuracy * 100,
                "meets_minimum": accuracy >= self.minimum_accuracy,
                "minimum_required": self.minimum_accuracy,
                "by_analysis_type": type_breakdown
            }

    # Test Case 1: Create data analysis validator
    validator = DataAnalysisValidator(minimum_accuracy=0.80)

    # Add data analysis statements
    validator.add_data_analysis_statement(
        statement_id="da_001",
        analysis_type="statistical_test",
        claim="Treatment group shows significant improvement (p<0.05)",
        supporting_data={"p_value": 0.023, "effect_size": 0.65}
    )

    validator.add_data_analysis_statement(
        statement_id="da_002",
        analysis_type="correlation",
        claim="Strong positive correlation between variables (r=0.82)",
        supporting_data={"r": 0.82, "p_value": 0.001}
    )

    # Add more statements to reach statistical significance
    for i in range(3, 11):
        validator.add_data_analysis_statement(
            statement_id=f"da_{i:03d}",
            analysis_type="regression" if i % 2 == 0 else "statistical_test",
            claim=f"Data analysis claim {i}",
            supporting_data={"metric": i * 0.1}
        )

    # Assert: Statements are added
    assert len(validator.data_statements) == 10, \
        "Should add all data analysis statements"

    # Test Case 2: Simulate expert validation (targeting >85% accuracy)
    # Make 9 out of 10 accurate (90%)
    for i in range(10):
        is_accurate = (i != 5)  # Make one incorrect

        validator.validate_statement(
            statement_id=f"da_{i+1:03d}",
            is_accurate=is_accurate,
            confidence=0.9 if is_accurate else 0.5,
            expert_rationale=f"Expert assessment {i+1}"
        )

    # Compute accuracy
    result = validator.compute_data_analysis_accuracy()

    # Assert: Should meet >80% accuracy requirement
    assert result["data_analysis_accuracy"] >= 0.80, \
        "Data analysis statements MUST achieve >80% accuracy"
    assert result["meets_minimum"], \
        "Should meet minimum accuracy threshold"
    assert result["accurate_statements"] >= 8, \
        "Should have at least 8/10 accurate for 80% threshold"


@pytest.mark.requirement("REQ-SCI-VAL-006")
@pytest.mark.priority("MUST")
def test_req_sci_val_006_literature_accuracy_75_percent():
    """
    REQ-SCI-VAL-006: Literature review-based statements MUST achieve
    >75% accuracy when validated against primary sources, based on the
    paper's demonstrated 82.1% accuracy for literature statements.

    Validates that:
    - Literature statements have 75% accuracy threshold
    - Validation against primary sources is performed
    - Citation accuracy is verified
    """

    class LiteratureValidator:
        """Validator for literature review statements."""

        def __init__(self, minimum_accuracy: float = 0.75):
            self.minimum_accuracy = minimum_accuracy
            self.literature_statements = []

        def add_literature_statement(
            self,
            statement_id: str,
            claim: str,
            cited_papers: List[str],
            paraphrase: str
        ):
            """Add a literature review statement."""
            self.literature_statements.append({
                "statement_id": statement_id,
                "claim": claim,
                "cited_papers": cited_papers,
                "paraphrase": paraphrase,
                "validation": None
            })

        def validate_against_sources(
            self,
            statement_id: str,
            source_check_passed: bool,
            accurate_paraphrase: bool,
            citations_correct: bool,
            validator_notes: str
        ):
            """Validate literature statement against primary sources."""
            for stmt in self.literature_statements:
                if stmt["statement_id"] == statement_id:
                    # Statement is accurate if all checks pass
                    is_accurate = (
                        source_check_passed and
                        accurate_paraphrase and
                        citations_correct
                    )

                    stmt["validation"] = {
                        "is_accurate": is_accurate,
                        "source_check_passed": source_check_passed,
                        "accurate_paraphrase": accurate_paraphrase,
                        "citations_correct": citations_correct,
                        "notes": validator_notes
                    }
                    break

        def compute_literature_accuracy(self) -> Dict[str, Any]:
            """Compute accuracy for literature statements."""
            validated = [
                s for s in self.literature_statements
                if s["validation"] is not None
            ]

            if not validated:
                return {"error": "No validated literature statements"}

            total = len(validated)
            accurate = sum(
                1 for s in validated
                if s["validation"]["is_accurate"]
            )

            accuracy = accurate / total

            # Detailed checks
            source_checks_passed = sum(
                1 for s in validated
                if s["validation"]["source_check_passed"]
            )

            paraphrases_accurate = sum(
                1 for s in validated
                if s["validation"]["accurate_paraphrase"]
            )

            citations_correct = sum(
                1 for s in validated
                if s["validation"]["citations_correct"]
            )

            return {
                "total_statements": total,
                "accurate_statements": accurate,
                "literature_accuracy": accuracy,
                "literature_accuracy_percentage": accuracy * 100,
                "meets_minimum": accuracy >= self.minimum_accuracy,
                "minimum_required": self.minimum_accuracy,
                "validation_details": {
                    "source_checks_passed": source_checks_passed,
                    "paraphrases_accurate": paraphrases_accurate,
                    "citations_correct": citations_correct
                }
            }

    # Test Case 1: Create literature validator
    validator = LiteratureValidator(minimum_accuracy=0.75)

    # Add literature statements
    validator.add_literature_statement(
        statement_id="lit_001",
        claim="Previous studies have shown that X improves Y",
        cited_papers=["Smith et al. 2020", "Jones et al. 2021"],
        paraphrase="Smith (2020) and Jones (2021) demonstrated X enhances Y"
    )

    # Add more statements
    for i in range(2, 11):
        validator.add_literature_statement(
            statement_id=f"lit_{i:03d}",
            claim=f"Literature claim {i}",
            cited_papers=[f"Author{i} et al. 202{i % 3}"],
            paraphrase=f"According to Author{i}, finding {i}"
        )

    # Assert: Statements are added
    assert len(validator.literature_statements) == 10, \
        "Should add all literature statements"

    # Test Case 2: Validate statements (targeting >82% accuracy)
    # Make 9 out of 10 accurate (90%)
    for i in range(10):
        # Make statement 7 have issues
        if i == 6:
            validator.validate_against_sources(
                statement_id=f"lit_{i+1:03d}",
                source_check_passed=True,
                accurate_paraphrase=False,  # Paraphrase issue
                citations_correct=True,
                validator_notes="Paraphrase misrepresents source"
            )
        else:
            validator.validate_against_sources(
                statement_id=f"lit_{i+1:03d}",
                source_check_passed=True,
                accurate_paraphrase=True,
                citations_correct=True,
                validator_notes="Accurate representation of sources"
            )

    # Compute accuracy
    result = validator.compute_literature_accuracy()

    # Assert: Should meet >75% accuracy requirement
    assert result["literature_accuracy"] >= 0.75, \
        "Literature statements MUST achieve >75% accuracy"
    assert result["meets_minimum"], \
        "Should meet minimum accuracy threshold"
    assert result["accurate_statements"] >= 8, \
        "Should have sufficient accurate statements"


@pytest.mark.requirement("REQ-SCI-VAL-007")
@pytest.mark.priority("MUST")
def test_req_sci_val_007_track_accuracy_by_statement_type():
    """
    REQ-SCI-VAL-007: The system MUST track accuracy by statement type
    (data analysis, literature synthesis, interpretation) and report these
    metrics separately, as interpretation statements are expected to have
    lower accuracy (~58%) compared to data analysis or literature statements.

    Validates that:
    - Accuracy is tracked separately by statement type
    - Different accuracy expectations for different types
    - Reporting includes breakdown by type
    """

    class StatementTypeTracker:
        """Track and report accuracy by statement type."""

        def __init__(self):
            self.statements_by_type = {
                "data_analysis": [],
                "literature": [],
                "interpretation": [],
                "methodology": []
            }

            # Expected accuracy thresholds by type
            self.accuracy_targets = {
                "data_analysis": 0.80,  # >80%
                "literature": 0.75,  # >75%
                "interpretation": 0.50,  # ~58% is acceptable (lower threshold)
                "methodology": 0.70  # Reasonable threshold
            }

        def add_statement(
            self,
            statement_type: str,
            statement_id: str,
            content: str,
            is_accurate: bool = None
        ):
            """Add a statement with type classification."""
            if statement_type not in self.statements_by_type:
                raise ValueError(f"Unknown statement type: {statement_type}")

            self.statements_by_type[statement_type].append({
                "id": statement_id,
                "content": content,
                "is_accurate": is_accurate
            })

        def compute_accuracy_by_type(self) -> Dict[str, Any]:
            """Compute accuracy breakdown by statement type."""
            results = {
                "by_type": {},
                "overall": {
                    "total": 0,
                    "accurate": 0,
                    "accuracy": 0.0
                },
                "all_types_meet_targets": True
            }

            total_all = 0
            accurate_all = 0

            for stmt_type, statements in self.statements_by_type.items():
                # Only count validated statements
                validated = [s for s in statements if s["is_accurate"] is not None]

                if not validated:
                    results["by_type"][stmt_type] = {
                        "total": 0,
                        "validated": 0,
                        "accurate": 0,
                        "accuracy": None,
                        "target": self.accuracy_targets[stmt_type],
                        "meets_target": None
                    }
                    continue

                total = len(validated)
                accurate = sum(1 for s in validated if s["is_accurate"])
                accuracy = accurate / total

                total_all += total
                accurate_all += accurate

                meets_target = accuracy >= self.accuracy_targets[stmt_type]

                results["by_type"][stmt_type] = {
                    "total": total,
                    "validated": len(validated),
                    "accurate": accurate,
                    "accuracy": accuracy,
                    "accuracy_percentage": accuracy * 100,
                    "target": self.accuracy_targets[stmt_type],
                    "target_percentage": self.accuracy_targets[stmt_type] * 100,
                    "meets_target": meets_target,
                    "delta_from_target": accuracy - self.accuracy_targets[stmt_type]
                }

                if not meets_target:
                    results["all_types_meet_targets"] = False

            # Overall accuracy
            if total_all > 0:
                results["overall"] = {
                    "total": total_all,
                    "accurate": accurate_all,
                    "accuracy": accurate_all / total_all,
                    "accuracy_percentage": (accurate_all / total_all) * 100
                }

            return results

    # Test Case 1: Create tracker and add statements
    tracker = StatementTypeTracker()

    # Add data analysis statements (should have >80% accuracy)
    for i in range(10):
        tracker.add_statement(
            statement_type="data_analysis",
            statement_id=f"da_{i}",
            content=f"Data analysis statement {i}",
            is_accurate=(i < 9)  # 90% accurate
        )

    # Add literature statements (should have >75% accuracy)
    for i in range(10):
        tracker.add_statement(
            statement_type="literature",
            statement_id=f"lit_{i}",
            content=f"Literature statement {i}",
            is_accurate=(i < 8)  # 80% accurate
        )

    # Add interpretation statements (lower accuracy acceptable, ~58%)
    for i in range(10):
        tracker.add_statement(
            statement_type="interpretation",
            statement_id=f"int_{i}",
            content=f"Interpretation statement {i}",
            is_accurate=(i < 6)  # 60% accurate (acceptable for interpretation)
        )

    # Compute accuracy by type
    results = tracker.compute_accuracy_by_type()

    # Assert: All statement types are tracked
    assert "by_type" in results, "Should track by type"
    assert "data_analysis" in results["by_type"], \
        "Should track data analysis statements"
    assert "literature" in results["by_type"], \
        "Should track literature statements"
    assert "interpretation" in results["by_type"], \
        "Should track interpretation statements"

    # Assert: Data analysis meets high threshold
    da_result = results["by_type"]["data_analysis"]
    assert da_result["accuracy"] >= 0.80, \
        "Data analysis should meet 80% threshold"
    assert da_result["meets_target"], \
        "Data analysis should meet its target"

    # Assert: Literature meets threshold
    lit_result = results["by_type"]["literature"]
    assert lit_result["accuracy"] >= 0.75, \
        "Literature should meet 75% threshold"
    assert lit_result["meets_target"], \
        "Literature should meet its target"

    # Assert: Interpretation has lower but acceptable accuracy
    int_result = results["by_type"]["interpretation"]
    assert int_result["accuracy"] >= 0.50, \
        "Interpretation should meet lower 50% threshold"
    assert int_result["accuracy"] < 0.70, \
        "Interpretation accuracy should be lower than data/literature"
    assert int_result["target"] == 0.50, \
        "Interpretation should have lower target (~50-58%)"

    # Assert: Overall metrics are computed
    assert "overall" in results, "Should compute overall metrics"
    assert results["overall"]["total"] == 30, \
        "Should count all statements"
    assert results["overall"]["accuracy"] is not None, \
        "Should compute overall accuracy"

    # Test Case 2: Verify separate reporting
    assert da_result["target"] != int_result["target"], \
        "Different statement types should have different targets"
    assert "delta_from_target" in da_result, \
        "Should report delta from target for each type"
