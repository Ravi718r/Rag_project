from pathlib import Path
import json

THRESHOLDS = {
    "context_recall": 0.80,
    "context_precision": 0.70,
    "faithfulness": 0.85,
    "answer_relevancy": 0.80
}

MAX_REGRESSION = {
    "context_recall": 0.05,
    "context_precision": 0.05,
    "faithfulness": 0.05,
    "answer_relevancy": 0.05
}


def compare_with_baseline(
    current_summary,
    baseline_summary
):
    """
    Compare current evaluation against
    a previous baseline.
    """

    comparison = {}

    for metric in THRESHOLDS:
        current = current_summary.get(
            metric,
            0
        )

        baseline = baseline_summary.get(
            metric
        )

        delta = None
        regression = False

        if baseline is not None:

            delta = current - baseline

            if delta < -MAX_REGRESSION[metric]:

                regression = True

        comparison[metric] = {
            "current": current,
            "baseline": baseline,
            "delta": delta,
            "regression": regression,
            "threshold": THRESHOLDS[metric],
            "threshold_pass": (
                current >= THRESHOLDS[metric]
            )
        }
    return comparison


def evaluation_gate(
    comparison
):
    """
    Decide whether the evaluation passes.
    """

    failures = []

    for metric, result in comparison.items():

        if not result["threshold_pass"]:
            failures.append({
                "metric": metric,
                "type": "threshold",
                "current": result["current"],
                "required": result["threshold"]
            })

        if result["regression"]:

            failures.append({
                "metric": metric,
                "type": "regression",
                "current": result["current"],
                "baseline": result["baseline"],
                "delta": result["delta"]
            })

    return {
        "passed": len(failures) == 0,
        "failures": failures
    }


def load_baseline(path):
    """
    Load a previous experiment.
    """

    path = Path(path)

    if not path.exists():

        return None

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)