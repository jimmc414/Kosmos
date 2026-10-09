"""
End-of-run results for `kosmos run`.

build_run_results reads the hypotheses, experiments and results of a run from
the database columns (never from repr strings of ORM rows) and the LLM usage
from the provider, so the report shows execution success, data source,
validation status, failed tasks and real cost.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from kosmos.db import get_session
from kosmos.db.operations import get_experiment, get_hypothesis, get_results_for_run

logger = logging.getLogger(__name__)


def _number(value: Any) -> Optional[float]:
    """Return value when it is a real number (not a bool or a mock), else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _hypothesis_row(h, plan) -> Dict[str, Any]:
    return {
        "id": h.id,
        "claim": h.statement,
        "novelty_score": h.novelty_score,
        "testability_score": h.testability_score,
        "priority_score": None,
        "status": _enum_value(h.status),
        "tested": h.id in plan.tested_hypotheses,
        "supported": h.id in plan.supported_hypotheses,
        "untestable": h.id in plan.untestable_hypotheses,
    }


def _experiment_row(e) -> Dict[str, Any]:
    return {
        "id": e.id,
        "type": e.experiment_type,
        "status": _enum_value(e.status),
        "created_at": e.created_at.isoformat() if e.created_at else None,
        "hypothesis_id": e.hypothesis_id,
    }


def _result_row(r) -> Dict[str, Any]:
    tests = r.statistical_tests or {}
    experiment = r.experiment
    return {
        "result_id": r.id,
        "experiment_id": r.experiment_id,
        "hypothesis_id": experiment.hypothesis_id if experiment else None,
        "execution_success": r.execution_success,
        "data_source": r.data_source,
        "test_type": tests.get("test_type"),
        "statistic": tests.get("statistic"),
        "p_value": r.p_value,
        "effect_size": r.effect_size,
        "n": tests.get("n"),
        "supports_hypothesis": r.supports_hypothesis,
        "validation_status": r.validation_status,
        "validation_reason": (r.validation_detail or {}).get("reason"),
        "random_seed": r.random_seed,
        "error_message": r.error_message,
        "cost_usd": r.cost_usd,
    }


def _usage(llm_client) -> Dict[str, Any]:
    """Provider usage stats; empty when the client has no get_usage_stats."""
    get_usage_stats = getattr(llm_client, "get_usage_stats", None)
    if not callable(get_usage_stats):
        return {}
    try:
        usage = get_usage_stats()
    except Exception as e:
        logger.warning(f"Could not read LLM usage stats: {e}")
        return {}
    return usage if isinstance(usage, dict) else {}


def result_metrics(results: List[Dict[str, Any]], total_cost: Optional[float]) -> Dict[str, Any]:
    """The plan section 7 counts over _result_row dicts, and the cost per validated finding."""
    validated = sum(1 for r in results if r["validation_status"] == "validated")
    return {
        "experiments_attempted": len(results),
        "experiments_succeeded": sum(1 for r in results if r["execution_success"] is True),
        "experiments_failed": sum(1 for r in results if r["execution_success"] is not True),
        "results_from_file": sum(1 for r in results if r["data_source"] == "file"),
        "results_synthetic": sum(1 for r in results if r["data_source"] == "synthetic"),
        "findings_validated": validated,
        "findings_rejected": sum(
            1 for r in results if r["validation_status"] in ("rejected", "rejected_unsafe")
        ),
        "cost_per_validated_finding": (
            total_cost / validated if validated and total_cost is not None else None
        ),
    }


def build_run_results(director, question: str, max_iterations: int) -> Dict[str, Any]:
    """Assemble the end-of-run report of a `kosmos run`.

    Args:
        director: The ResearchDirectorAgent that ran the research
        question: Research question
        max_iterations: Maximum iterations of the run

    Returns:
        Dict with the run overview, hypotheses, experiments, results and metrics
    """
    final_status = director.get_research_status()
    plan = director.research_plan
    run_id = getattr(director, "run_id", None)

    hypotheses: List[Dict[str, Any]] = []
    experiments: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []

    if not plan:
        logger.warning("No research plan available")
    else:
        try:
            with get_session() as session:
                for h_id in plan.hypothesis_pool:
                    h = get_hypothesis(session, h_id)
                    if h:
                        hypotheses.append(_hypothesis_row(h, plan))

                db_results = get_results_for_run(session, run_id) if run_id else []
                results = [_result_row(r) for r in db_results]

                # Completed experiments first, then failed ones that left a result
                experiment_ids = list(plan.completed_experiments)
                for r in db_results:
                    if r.experiment_id not in experiment_ids:
                        experiment_ids.append(r.experiment_id)
                for e_id in experiment_ids:
                    e = get_experiment(session, e_id)
                    if e:
                        experiments.append(_experiment_row(e))
        except Exception as e:
            logger.warning(f"Could not fetch run objects from database: {e}")

    usage = _usage(getattr(director, "llm_client", None))
    total_cost = _number(usage.get("total_cost_usd"))
    if total_cost is None:
        total_cost = _number(getattr(getattr(director, "llm_client", None), "total_cost_usd", None))

    untestable = getattr(plan, "untestable_hypotheses", []) if plan else []

    metrics = {
        "api_calls": usage.get("total_requests", 0),
        "total_cost_usd": total_cost,
        "input_tokens": usage.get("total_input_tokens", 0),
        "output_tokens": usage.get("total_output_tokens", 0),
        **result_metrics(results, total_cost),
        "hypotheses_untestable": len(untestable),
        "hypotheses_generated": final_status.get("hypothesis_pool_size", 0),
        "hypotheses_tested": final_status.get("hypotheses_tested", 0),
        "hypotheses_supported": final_status.get("hypotheses_supported", 0),
        "hypotheses_rejected": final_status.get("hypotheses_rejected", 0),
    }

    return {
        "id": run_id or f"research_{int(time.time())}",
        "run_id": run_id,
        "question": question,
        "domain": final_status.get("domain") or "auto",
        "state": final_status.get("workflow_state", "COMPLETED"),
        "current_iteration": final_status.get("iteration", 0),
        "max_iterations": max_iterations,
        "has_converged": final_status.get("has_converged", False),
        "convergence_reason": final_status.get("convergence_reason"),
        "hypotheses": hypotheses,
        "experiments": experiments,
        "results": results,
        "metrics": metrics,
    }
