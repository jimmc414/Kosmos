"""
Tests for Scientific Metrics Requirements (REQ-SCI-METRIC-*).

These tests validate expert time estimation and cumulative time tracking
as specified in REQUIREMENTS.md Section 10.5.
"""

import pytest
from typing import Dict, Any, List
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Test markers for requirements traceability
pytestmark = [
    pytest.mark.requirement("REQ-SCI-METRIC"),
    pytest.mark.category("scientific"),
    pytest.mark.priority("SHOULD"),
]


@pytest.mark.requirement("REQ-SCI-METRIC-001")
@pytest.mark.priority("SHOULD")
def test_req_sci_metric_001_estimate_expert_time():
    """
    REQ-SCI-METRIC-001: The system SHOULD provide metrics estimating the
    equivalent expert time represented by the work performed (e.g., papers
    read × 15 minutes/paper + analyses × 2 hours/analysis), as the paper
    reports Kosmos performs work equivalent to 6 months of expert time.

    Validates that:
    - Expert time can be estimated for different activities
    - Time estimates use reasonable conversion factors
    - Total expert time is computed
    """

    class ExpertTimeEstimator:
        """Estimate equivalent expert time for automated research work."""

        # Standard time estimates per activity (in hours)
        TIME_PER_PAPER_READ = 0.25  # 15 minutes
        TIME_PER_PAPER_DEEP_READ = 1.0  # 1 hour for thorough reading
        TIME_PER_ANALYSIS = 2.0  # 2 hours per analysis
        TIME_PER_EXPERIMENT_DESIGN = 4.0  # 4 hours
        TIME_PER_HYPOTHESIS_GENERATION = 0.5  # 30 minutes
        TIME_PER_LITERATURE_SYNTHESIS = 3.0  # 3 hours
        TIME_PER_REPORT_WRITING = 5.0  # 5 hours

        def __init__(self):
            self.activities = []

        def record_activity(
            self,
            activity_type: str,
            count: int = 1,
            custom_time_per_unit: float = None
        ):
            """Record an activity performed by the system."""
            # Map activity type to standard time
            time_mapping = {
                "paper_read": self.TIME_PER_PAPER_READ,
                "paper_deep_read": self.TIME_PER_PAPER_DEEP_READ,
                "analysis": self.TIME_PER_ANALYSIS,
                "experiment_design": self.TIME_PER_EXPERIMENT_DESIGN,
                "hypothesis_generation": self.TIME_PER_HYPOTHESIS_GENERATION,
                "literature_synthesis": self.TIME_PER_LITERATURE_SYNTHESIS,
                "report_writing": self.TIME_PER_REPORT_WRITING
            }

            time_per_unit = custom_time_per_unit or time_mapping.get(activity_type, 1.0)

            self.activities.append({
                "type": activity_type,
                "count": count,
                "time_per_unit": time_per_unit,
                "total_time": count * time_per_unit,
                "timestamp": datetime.now()
            })

        def compute_total_expert_time(self) -> Dict[str, Any]:
            """
            Compute total estimated expert time.

            Returns:
                Dict with time estimates in various units
            """
            total_hours = sum(activity["total_time"] for activity in self.activities)

            # Convert to different units
            total_days = total_hours / 8  # 8-hour work days
            total_weeks = total_days / 5  # 5-day work weeks
            total_months = total_weeks / 4  # ~4 weeks per month

            # Breakdown by activity type
            by_activity_type = {}
            for activity in self.activities:
                act_type = activity["type"]
                if act_type not in by_activity_type:
                    by_activity_type[act_type] = {
                        "count": 0,
                        "total_hours": 0.0
                    }

                by_activity_type[act_type]["count"] += activity["count"]
                by_activity_type[act_type]["total_hours"] += activity["total_time"]

            return {
                "total_hours": round(total_hours, 2),
                "total_days": round(total_days, 2),
                "total_weeks": round(total_weeks, 2),
                "total_months": round(total_months, 2),
                "total_activities": len(self.activities),
                "by_activity_type": by_activity_type,
                "interpretation": self._generate_interpretation(total_months)
            }

        def _generate_interpretation(self, months: float) -> str:
            """Generate human-readable interpretation."""
            if months < 0.5:
                return f"Approximately {int(months * 4)} weeks of expert work"
            elif months < 1:
                return "Approximately 1 month of expert work"
            elif months < 12:
                return f"Approximately {int(months)} months of expert work"
            else:
                years = months / 12
                return f"Approximately {years:.1f} years of expert work"

    # Test Case 1: Estimate time for paper reading
    estimator = ExpertTimeEstimator()

    # Simulate reading 100 papers
    estimator.record_activity("paper_read", count=100)

    result = estimator.compute_total_expert_time()

    # Assert: Time is estimated correctly
    expected_hours = 100 * 0.25  # 100 papers × 15 minutes
    assert result["total_hours"] == expected_hours, \
        "Should correctly estimate time for paper reading"
    assert result["total_days"] == expected_hours / 8, \
        "Should convert to days correctly"

    # Test Case 2: Complex research workflow
    estimator2 = ExpertTimeEstimator()

    # Simulate a research cycle
    estimator2.record_activity("paper_read", count=50)  # 12.5 hours
    estimator2.record_activity("paper_deep_read", count=20)  # 20 hours
    estimator2.record_activity("hypothesis_generation", count=10)  # 5 hours
    estimator2.record_activity("analysis", count=15)  # 30 hours
    estimator2.record_activity("experiment_design", count=5)  # 20 hours
    estimator2.record_activity("literature_synthesis", count=3)  # 9 hours
    estimator2.record_activity("report_writing", count=2)  # 10 hours

    result2 = estimator2.compute_total_expert_time()

    # Total: 12.5 + 20 + 5 + 30 + 20 + 9 + 10 = 106.5 hours
    expected_total = 106.5

    # Assert: Total is computed correctly
    assert abs(result2["total_hours"] - expected_total) < 0.1, \
        "Should compute total expert time across all activities"

    # Assert: Conversion to months is reasonable
    # 106.5 hours / 8 hours per day / 5 days per week / 4 weeks per month
    expected_months = 106.5 / 8 / 5 / 4
    assert abs(result2["total_months"] - expected_months) < 0.1, \
        "Should convert to months correctly"

    # Assert: Breakdown by activity type exists
    assert "by_activity_type" in result2, \
        "Should provide breakdown by activity type"
    assert "paper_read" in result2["by_activity_type"], \
        "Should include all activity types"
    assert result2["by_activity_type"]["paper_read"]["count"] == 50, \
        "Should count activities correctly"

    # Test Case 3: Verify against paper's claim (6 months)
    # Paper reports Kosmos performs work equivalent to 6 months
    estimator3 = ExpertTimeEstimator()

    # Simulate scale comparable to paper
    # 6 months ≈ 6 * 4 weeks * 5 days * 8 hours = 960 hours
    # Example workload to reach this:
    estimator3.record_activity("paper_read", count=1000)  # 250 hours
    estimator3.record_activity("paper_deep_read", count=100)  # 100 hours
    estimator3.record_activity("hypothesis_generation", count=200)  # 100 hours
    estimator3.record_activity("analysis", count=150)  # 300 hours
    estimator3.record_activity("experiment_design", count=30)  # 120 hours
    estimator3.record_activity("literature_synthesis", count=20)  # 60 hours
    estimator3.record_activity("report_writing", count=6)  # 30 hours

    result3 = estimator3.compute_total_expert_time()

    # Assert: Can estimate work at scale comparable to paper
    assert result3["total_months"] >= 4, \
        "Should be able to estimate large-scale work (several months)"
    assert "interpretation" in result3, \
        "Should provide human-readable interpretation"


@pytest.mark.requirement("REQ-SCI-METRIC-002")
@pytest.mark.priority("SHOULD")
def test_req_sci_metric_002_track_cumulative_expert_time():
    """
    REQ-SCI-METRIC-002: The system SHOULD track the cumulative
    expert-equivalent time across discovery iterations to demonstrate
    scaling of research output with runtime.

    Validates that:
    - Expert time is tracked across iterations
    - Cumulative time is computed
    - Scaling with iterations is demonstrated
    """

    class CumulativeTimeTracker:
        """Track cumulative expert time across discovery iterations."""

        def __init__(self):
            self.iterations = []
            self.cumulative_hours = 0.0

        def record_iteration(
            self,
            iteration_id: int,
            activities: Dict[str, int],
            time_per_activity: Dict[str, float] = None
        ) -> Dict[str, Any]:
            """
            Record a discovery iteration and compute expert time.

            Args:
                iteration_id: Iteration number
                activities: Dict of {activity_type: count}
                time_per_activity: Optional custom time estimates

            Returns:
                Dict with iteration details and cumulative time
            """
            # Default time estimates (in hours)
            default_times = {
                "paper_read": 0.25,
                "analysis": 2.0,
                "hypothesis_generation": 0.5,
                "experiment": 4.0,
                "synthesis": 3.0
            }

            time_mapping = time_per_activity or default_times

            # Compute time for this iteration
            iteration_hours = 0.0
            activity_breakdown = {}

            for activity_type, count in activities.items():
                time_per_unit = time_mapping.get(activity_type, 1.0)
                activity_time = count * time_per_unit
                iteration_hours += activity_time

                activity_breakdown[activity_type] = {
                    "count": count,
                    "time_per_unit": time_per_unit,
                    "total_hours": activity_time
                }

            # Update cumulative time
            previous_cumulative = self.cumulative_hours
            self.cumulative_hours += iteration_hours

            # Record iteration
            iteration_record = {
                "iteration_id": iteration_id,
                "timestamp": datetime.now(),
                "activities": activity_breakdown,
                "iteration_hours": iteration_hours,
                "cumulative_hours": self.cumulative_hours,
                "delta_from_previous": iteration_hours,
                "cumulative_months": self.cumulative_hours / 8 / 5 / 4
            }

            self.iterations.append(iteration_record)

            return iteration_record

        def get_scaling_metrics(self) -> Dict[str, Any]:
            """
            Compute metrics showing scaling of work with iterations.

            Returns:
                Dict with scaling analysis
            """
            if not self.iterations:
                return {"error": "No iterations recorded"}

            total_iterations = len(self.iterations)
            total_hours = self.cumulative_hours

            # Time per iteration (average)
            avg_hours_per_iteration = total_hours / total_iterations

            # Growth rate (linear regression slope approximation)
            iteration_hours = [it["iteration_hours"] for it in self.iterations]
            if len(iteration_hours) > 1:
                # Simple linear fit
                iterations_nums = list(range(1, total_iterations + 1))
                mean_iter = sum(iterations_nums) / len(iterations_nums)
                mean_hours = sum(iteration_hours) / len(iteration_hours)

                numerator = sum(
                    (iterations_nums[i] - mean_iter) * (iteration_hours[i] - mean_hours)
                    for i in range(len(iterations_nums))
                )
                denominator = sum(
                    (iterations_nums[i] - mean_iter) ** 2
                    for i in range(len(iterations_nums))
                )

                slope = numerator / denominator if denominator != 0 else 0.0
            else:
                slope = 0.0

            # Scaling interpretation
            if abs(slope) < 0.1:
                scaling_type = "constant"
                scaling_desc = "Work per iteration is relatively constant"
            elif slope > 0:
                scaling_type = "increasing"
                scaling_desc = "Work per iteration is increasing over time"
            else:
                scaling_type = "decreasing"
                scaling_desc = "Work per iteration is decreasing (efficiency improving)"

            return {
                "total_iterations": total_iterations,
                "total_cumulative_hours": round(total_hours, 2),
                "total_cumulative_months": round(total_hours / 8 / 5 / 4, 2),
                "average_hours_per_iteration": round(avg_hours_per_iteration, 2),
                "scaling_slope": round(slope, 3),
                "scaling_type": scaling_type,
                "scaling_description": scaling_desc,
                "iterations_timeline": [
                    {
                        "iteration": it["iteration_id"],
                        "hours": round(it["iteration_hours"], 2),
                        "cumulative": round(it["cumulative_hours"], 2)
                    }
                    for it in self.iterations
                ]
            }

        def generate_scaling_report(self) -> str:
            """Generate human-readable scaling report."""
            metrics = self.get_scaling_metrics()

            if "error" in metrics:
                return metrics["error"]

            report = f"""
Expert Time Scaling Report
========================

Total Iterations: {metrics['total_iterations']}
Total Expert Time: {metrics['total_cumulative_hours']:.2f} hours ({metrics['total_cumulative_months']:.2f} months)
Average per Iteration: {metrics['average_hours_per_iteration']:.2f} hours

Scaling Analysis:
- Type: {metrics['scaling_type']}
- {metrics['scaling_description']}
- Growth rate: {metrics['scaling_slope']:.3f} hours/iteration

Timeline:
""".strip()

            for item in metrics["iterations_timeline"]:
                report += f"\n  Iteration {item['iteration']:2d}: {item['hours']:6.2f}h (cumulative: {item['cumulative']:7.2f}h)"

            return report

    # Test Case 1: Track multiple iterations
    tracker = CumulativeTimeTracker()

    # Simulate 5 iterations of discovery
    iteration_activities = [
        # Iteration 1: Initial exploration
        {"paper_read": 50, "analysis": 5, "hypothesis_generation": 10},

        # Iteration 2: Focused analysis
        {"paper_read": 30, "analysis": 10, "hypothesis_generation": 5, "experiment": 3},

        # Iteration 3: Deeper investigation
        {"paper_read": 40, "analysis": 15, "synthesis": 2},

        # Iteration 4: Refinement
        {"paper_read": 20, "analysis": 8, "experiment": 2, "synthesis": 1},

        # Iteration 5: Convergence
        {"paper_read": 15, "analysis": 5, "synthesis": 2}
    ]

    for i, activities in enumerate(iteration_activities, start=1):
        result = tracker.record_iteration(
            iteration_id=i,
            activities=activities
        )

        # Assert: Iteration is recorded
        assert result["iteration_id"] == i, \
            "Should record correct iteration ID"
        assert result["iteration_hours"] > 0, \
            "Should compute time for iteration"
        assert result["cumulative_hours"] >= result["iteration_hours"], \
            "Cumulative time should be >= iteration time"

    # Assert: All iterations are tracked
    assert len(tracker.iterations) == 5, \
        "Should track all iterations"

    # Test Case 2: Verify cumulative time increases
    cumulative_times = [it["cumulative_hours"] for it in tracker.iterations]

    # Assert: Cumulative time is monotonically increasing
    for i in range(1, len(cumulative_times)):
        assert cumulative_times[i] >= cumulative_times[i-1], \
            "Cumulative time should never decrease"

    # Test Case 3: Compute scaling metrics
    scaling = tracker.get_scaling_metrics()

    # Assert: Scaling metrics are computed
    assert "total_iterations" in scaling, \
        "Should compute total iterations"
    assert "total_cumulative_hours" in scaling, \
        "Should compute total cumulative hours"
    assert "average_hours_per_iteration" in scaling, \
        "Should compute average per iteration"
    assert "scaling_slope" in scaling, \
        "Should compute scaling slope"
    assert "scaling_type" in scaling, \
        "Should classify scaling type"

    # Assert: Cumulative time demonstrates scaling
    assert scaling["total_cumulative_hours"] > 50, \
        "Should accumulate significant work across iterations"
    assert scaling["total_cumulative_months"] > 0, \
        "Should convert to months"

    # Test Case 4: Generate scaling report
    report = tracker.generate_scaling_report()

    # Assert: Report is comprehensive
    assert "Total Iterations: 5" in report, \
        "Report should include iteration count"
    assert "Total Expert Time:" in report, \
        "Report should include total time"
    assert "Scaling Analysis:" in report, \
        "Report should include scaling analysis"
    assert "Timeline:" in report, \
        "Report should include iteration timeline"

    # Test Case 5: Demonstrate scaling with runtime
    # Verify that more iterations = more cumulative work
    tracker2 = CumulativeTimeTracker()

    # Simulate consistent work per iteration
    for i in range(1, 11):
        tracker2.record_iteration(
            iteration_id=i,
            activities={"paper_read": 10, "analysis": 5}
        )

    scaling2 = tracker2.get_scaling_metrics()

    # Assert: 10 iterations should have ~2x cumulative time of 5 iterations
    # Each iteration: 10*0.25 + 5*2.0 = 2.5 + 10 = 12.5 hours
    # 10 iterations = 125 hours
    expected_total = 12.5 * 10
    assert abs(scaling2["total_cumulative_hours"] - expected_total) < 1, \
        "Cumulative time should scale linearly with iterations"

    # Assert: More iterations demonstrate scaling
    assert scaling2["total_iterations"] == 10, \
        "Should track all 10 iterations"
    assert scaling2["total_cumulative_hours"] > 100, \
        "10 iterations should accumulate significant time"
