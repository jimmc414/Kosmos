#!/usr/bin/env python3
"""Ladder gate 4: the plan's no-LLM template run through the real sandbox.

Generates the bound Correlation template for co2_ppm vs temp_anomaly_c (the P2-1 binding,
mirroring tests/unit/execution/test_column_binding.py::test_correlation_on_bound_columns),
runs it with CodeExecutor() — the Docker sandbox, image kosmos-sandbox:latest — on the
climate CSV with seed 42, and checks the result against the CSV's known truth
(r 0.9317, p 5.8e-29, n 64). Makes no LLM call and opens no database.

Usage: python scripts/verify_template_run.py --csv <path> --out <json>
Exit 0 on match; 1 on any mismatch, a sandbox failure or a missing dataset (prints why).

Known-RED controls: --csv /nonexistent.csv -> "gate 4 RED: dataset missing"; with the
daemon stopped or the image removed CodeExecutor() raises / the run fails -> RED.
Known-GREEN control: the climate CSV with the image present -> "gate 4 OK ... r=0.93".
"""
import argparse
import json
import os
import sys
import time


def build_protocol():
    from kosmos.models.experiment import (
        ExperimentProtocol,
        ExperimentType,
        ProtocolStep,
        ResourceRequirements,
        Variable,
        VariableType,
    )

    return ExperimentProtocol(
        id="exp-ladder-gate4",
        name="Bound analysis",
        hypothesis_id="hyp-ladder-gate4",
        domain="climate_science",
        description="Analysis of two bound columns of the climate dataset",
        objective="Test the association between the bound columns",
        experiment_type=ExperimentType.COMPUTATIONAL,
        steps=[ProtocolStep(step_number=1, title="Analyse", description="Run the analysis", action="analyse")],
        variables={
            "predictor": Variable(name="predictor", type=VariableType.INDEPENDENT,
                                  description="Independent variable from the dataset", column="co2_ppm"),
            "outcome": Variable(name="outcome", type=VariableType.DEPENDENT,
                                description="Dependent variable from the dataset", column="temp_anomaly_c"),
        },
        resource_requirements=ResourceRequirements(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not os.path.exists(args.csv):
        print(f"gate 4 RED: dataset missing: {args.csv}")
        return 1

    from kosmos.execution.code_generator import ExperimentCodeGenerator
    from kosmos.execution.executor import CodeExecutor

    t0 = time.time()
    code = ExperimentCodeGenerator(use_llm=False).generate(build_protocol())
    try:
        executor = CodeExecutor()  # the sandbox; raises when the daemon is unreachable
        result = executor.execute_with_data(code, args.csv, seed=42)
    except Exception as exc:  # noqa: BLE001 - any sandbox failure is a red gate
        print(f"gate 4 RED: sandbox execution raised {type(exc).__name__}: {exc}")
        return 1
    rv = result.return_value if isinstance(result.return_value, dict) else {}
    stat, p = rv.get("statistic"), rv.get("p_value")
    checks = {
        "success": bool(result.success),
        "sandbox_used": getattr(result, "sandbox_used", None) is True,
        "data_source_file": rv.get("data_source") == "file",
        "test_type_pearson": rv.get("test_type") == "pearson_correlation",
        "n_64": rv.get("n") == 64,
        "statistic_near_0.9317": isinstance(stat, (int, float)) and abs(stat - 0.9317) < 0.005,
        "p_below_1e-20": isinstance(p, (int, float)) and p < 1e-20,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"checks": checks, "return_value": rv, "error": result.error,
                   "elapsed_s": round(time.time() - t0, 1)}, fh, indent=2, default=str)
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        print(f"gate 4 RED: failed checks {failed}; error={result.error!r}; keys={sorted(rv)}")
        return 1
    print(f"gate 4 OK: sandbox template run r={stat:.4f} p={p:.2e} n={rv['n']} "
          f"in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
