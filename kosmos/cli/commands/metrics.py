"""
`kosmos validate-null` and `kosmos rerun`: the metrics that need re-execution.

validate-null measures the permutation null's false-positive rate on shuffled
copies of each result's own dataset (shuffled_pass_rate). rerun re-executes a
stored result's code on its dataset with its seed, checks that the statistic is
reproduced exactly, and records whether the conclusion holds under other seeds.
Both read the result rows the director stored (statistical_tests with
'columns', provenance with data_path, data_sha256, seed and sandbox_used) and
call no LLM.
"""

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import typer
from rich.markup import escape

from kosmos.cli.utils import console, print_error
from kosmos.db import get_session
from kosmos.db.operations import (
    get_result,
    get_results_for_run,
    update_result_provenance,
    update_result_validation,
)

logger = logging.getLogger(__name__)

# The director's default permutation count (config 'null_permutations', P2-2)
DEFAULT_NULL_PERMUTATIONS = 500


def _dataset_problem(provenance: Dict[str, Any]) -> Optional[str]:
    """Why the result's dataset cannot be used, or None when it is the recorded file."""
    from kosmos.execution.data_schema import _sha256

    data_path = provenance.get("data_path")
    expected = provenance.get("data_sha256")
    if not data_path:
        return "no data_path in provenance"
    if not expected:
        return "no data_sha256 in provenance"
    path = Path(data_path)
    if not path.is_file():
        return f"data file not found: {data_path}"
    actual = _sha256(path)
    if actual != expected:
        return f"data file changed: sha256 {actual[:12]} != recorded {str(expected)[:12]}"
    return None


def shuffled_pass_rate(
    df,
    test_type: str,
    x_col: str,
    y_col: str,
    groups: Optional[List[Any]],
    k: int,
    seed: int,
    n_permutations: int = DEFAULT_NULL_PERMUTATIONS,
) -> float:
    """Fraction of k shuffled copies of df on which the permutation null test passes.

    Each copy breaks the tested association (shuffle_target with rng seed + i); a
    well-calibrated null passes about alpha of them.
    """
    from kosmos.validation.analysis_fn import build_analysis_fn, shuffle_target
    from kosmos.validation.null_model import NullModelValidator

    fn = build_analysis_fn(test_type, x_col, y_col, groups=groups)
    passes = 0
    for i in range(k):
        df_s = shuffle_target(df, test_type, x_col, y_col, rng=np.random.default_rng(seed + i))
        null = NullModelValidator(n_permutations, random_seed=seed + i).validate_finding(
            {"statistics": fn(df_s)}, data=df_s, analysis_func=fn,
            # The same shuffle as the director's gate, so this measures that gate
            shuffle_func=lambda d, rng: shuffle_target(d, test_type, x_col, y_col, rng),
        )
        passes += int(bool(null.passes_null_test))
    return passes / k


def _eligible(r) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """The fields validate-null needs from a result row, or the reason it is skipped."""
    from kosmos.validation.analysis_fn import SUPPORTED_TESTS

    stats = r.statistical_tests or {}
    columns = stats.get("columns") if isinstance(stats.get("columns"), dict) else {}
    test_type = stats.get("test_type")
    if not columns.get("x") or not columns.get("y"):
        return None, "no columns in statistical_tests"
    if test_type not in SUPPORTED_TESTS:
        return None, f"test_type {test_type!r} is not recomputable"
    return {
        "id": r.id,
        "test_type": test_type,
        "x": columns["x"],
        "y": columns["y"],
        "groups": columns.get("groups"),
        "provenance": dict(r.provenance or {}),
    }, None


def validate_null(
    run_id: str = typer.Option(..., "--run-id", help="Run id printed by `kosmos run`"),
    k: int = typer.Option(20, "--k", min=1, help="Shuffled copies per result"),
    alpha: float = typer.Option(
        0.05, "--alpha", help="Exit 1 when the run's mean shuffled pass rate exceeds this",
    ),
    seed: int = typer.Option(0, "--seed", help="Base seed; copy i uses seed + i"),
    permutations: int = typer.Option(
        DEFAULT_NULL_PERMUTATIONS, "--permutations", min=1,
        help="Permutations per null test (the director's default)",
    ),
):
    """
    Measure the null model's false-positive rate on shuffled data.

    For every validated or rejected result of the run, shuffles the tested
    association out of its dataset k times and counts how often the permutation
    null test still passes. Stores shuffled_pass_rate in the result's
    validation_detail and prints the run-level mean.

    Examples:

        kosmos validate-null --run-id run_0123456789ab --k 20 --alpha 0.05 --seed 0
    """
    from kosmos.execution.data_schema import _read_table

    try:
        with get_session() as session:
            rows = [
                r for r in get_results_for_run(session, run_id)
                if r.validation_status in ("validated", "rejected")
            ]
            candidates = [(r.id, *_eligible(r)) for r in rows]
    except Exception as e:
        print_error(f"Could not read run {run_id} from the database: {e}")
        raise typer.Exit(1) from None

    rates: List[float] = []
    for result_id, item, skip in candidates:
        if item is not None:
            skip = _dataset_problem(item["provenance"])
        if skip is not None:
            console.print(f"{escape(result_id)}  skipped: {escape(skip)}", soft_wrap=True)
            continue
        try:
            df = _read_table(Path(item["provenance"]["data_path"]))
            rate = shuffled_pass_rate(
                df, item["test_type"], item["x"], item["y"], item["groups"],
                k=k, seed=seed, n_permutations=permutations,
            )
        except Exception as e:
            logger.warning("Shuffled null for result %s failed: %s", result_id, e)
            console.print(f"{escape(result_id)}  skipped: shuffled null failed: {escape(str(e))}",
                          soft_wrap=True)
            continue

        try:
            with get_session() as session:
                r = get_result(session, result_id)
                detail = dict(r.validation_detail or {})
                detail["shuffled_pass_rate"] = rate
                detail["shuffled_null"] = {"k": k, "seed": seed, "n_permutations": permutations}
                update_result_validation(
                    session, result_id, r.validation_status, detail=detail,
                    supports_hypothesis=r.supports_hypothesis,
                )
        except Exception as e:
            print_error(f"Could not store the shuffled pass rate of {result_id}: {e}")
            raise typer.Exit(1) from None

        rates.append(rate)
        console.print(
            f"{escape(result_id)}  {item['test_type']} {escape(item['x'])} -> {escape(item['y'])}  "
            f"shuffled_pass_rate = {rate:.3f} (k={k})",
            soft_wrap=True,
        )

    if not rates:
        print_error(
            f"No validated or rejected result of run '{run_id}' could be re-tested "
            "(each needs statistical_tests columns and its recorded data file)."
        )
        raise typer.Exit(1)

    mean = sum(rates) / len(rates)
    console.print(f"Run {escape(run_id)}: mean shuffled_pass_rate = {mean:.3f} over {len(rates)} "
                  f"result(s), alpha = {alpha}", soft_wrap=True)
    # A rate is a multiple of 1/k; averaging floats can land a hair above alpha
    # ((0.05 + 0.05 + 0.05) / 3 > 0.05), so compare with a tolerance far below 1/k
    if mean - alpha > 1e-9:
        print_error(f"The null model passes {mean:.3f} of shuffled datasets, above alpha {alpha}.")
        raise typer.Exit(1)


def _parse_seeds(seeds: str) -> List[int]:
    try:
        return [int(s) for s in seeds.split(",") if s.strip()]
    except ValueError:
        raise typer.BadParameter(f"--seeds must be comma-separated integers, got {seeds!r}") from None


def _p_value(return_value: Any) -> Optional[float]:
    if isinstance(return_value, dict) and isinstance(return_value.get("p_value"), (int, float)):
        return float(return_value["p_value"])
    return None


def rerun_result(
    result_id: str = typer.Option(..., "--result-id", help="Result id to re-execute"),
    seeds: str = typer.Option("", "--seeds", help="Extra seeds, comma-separated (e.g. 1,2,3)"),
    alpha: float = typer.Option(0.05, "--alpha", help="Significance threshold of the conclusion"),
):
    """
    Re-execute a stored result and check that it reproduces.

    Runs the experiment's stored code on the recorded dataset with the recorded
    seed (in the sandbox when the result ran there), compares the statistic with
    the stored one, then re-runs with each extra seed and checks that the
    p < alpha conclusion holds. Stores provenance['reproducibility'].

    Examples:

        kosmos rerun --result-id 0f8e... --seeds 1,2,3
    """
    from kosmos.execution.executor import CodeExecutor

    extra_seeds = _parse_seeds(seeds)
    try:
        with get_session() as session:
            r = get_result(session, result_id, with_experiment=True)
            if r is None:
                print_error(f"No result with id '{result_id}'.")
                raise typer.Exit(1)
            code = r.experiment.code_generated if r.experiment else None
            provenance = dict(r.provenance or {})
            stats = r.statistical_tests or {}
            stored_statistic = stats.get("statistic")
            stored_p = stats.get("p_value") if stats.get("p_value") is not None else r.p_value
    except typer.Exit:
        raise
    except Exception as e:
        print_error(f"Could not read result {result_id} from the database: {e}")
        raise typer.Exit(1) from None

    if not code:
        print_error(f"Result {result_id} has no stored code (experiment code_generated is empty).")
        raise typer.Exit(1)
    if not isinstance(stored_statistic, (int, float)):
        print_error(f"Result {result_id} has no stored statistic to compare.")
        raise typer.Exit(1)
    problem = _dataset_problem(provenance)
    if problem is not None:
        print_error(f"Cannot re-run result {result_id}: {problem}")
        raise typer.Exit(1)

    executor = CodeExecutor(use_sandbox=bool(provenance.get("sandbox_used")))
    data_path = provenance["data_path"]
    first = executor.execute_with_data(code, data_path, seed=provenance.get("seed"))
    statistic = first.return_value.get("statistic") if (
        first.success and isinstance(first.return_value, dict)) else None
    exact_match = isinstance(statistic, (int, float)) and math.isclose(
        float(statistic), float(stored_statistic), rel_tol=1e-9
    )
    if not first.success:
        console.print(f"Re-execution failed: {escape(str(first.error))}", soft_wrap=True)

    stored_conclusion = isinstance(stored_p, (int, float)) and stored_p < alpha
    seed_p: Dict[str, Optional[float]] = {}
    conclusion_stable = True
    for s in extra_seeds:
        run = executor.execute_with_data(code, data_path, seed=s)
        p = _p_value(run.return_value) if run.success else None
        seed_p[str(s)] = p
        if p is None or (p < alpha) != stored_conclusion:
            conclusion_stable = False

    reproducibility = {
        "exact_match": bool(exact_match),
        "seeds": seed_p,
        "conclusion_stable": conclusion_stable,
    }
    provenance["reproducibility"] = reproducibility
    try:
        with get_session() as session:
            update_result_provenance(session, result_id, provenance)
    except Exception as e:
        print_error(f"Could not store the reproducibility record of {result_id}: {e}")
        raise typer.Exit(1) from None

    console.print(
        f"Result {escape(result_id)}: statistic {statistic!r} vs stored {stored_statistic!r} "
        f"(seed {provenance.get('seed')!r}) -> exact_match = {bool(exact_match)}",
        soft_wrap=True,
    )
    for s, p in seed_p.items():
        console.print(f"  seed {s}: p_value = {p!r}", soft_wrap=True)
    console.print(f"  conclusion (p < {alpha}) stable across seeds: {conclusion_stable}", soft_wrap=True)
    if not exact_match:
        raise typer.Exit(1)
