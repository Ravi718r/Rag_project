class EvaluationGate:

    def __init__(
        self,
        min_context_recall=0.80,
        min_context_precision=0.70,
        min_faithfulness=0.80,
        min_answer_relevancy=0.80
    ):

        self.thresholds = {
            "context_recall": min_context_recall,
            "context_precision": min_context_precision,
            "faithfulness": min_faithfulness,
            "answer_relevancy": min_answer_relevancy
        }

    def check(self, summary):

        metrics = summary.get("metrics", {})

        failures = []

        for metric, threshold in self.thresholds.items():

            actual = metrics.get(metric)

            # Missing metric
            if actual is None:

                failures.append({
                    "metric": metric,
                    "actual": None,
                    "required": threshold,
                    "reason": "Metric missing"
                })

                continue

            # Below threshold
            if actual < threshold:

                failures.append({
                    "metric": metric,
                    "actual": actual,
                    "required": threshold,
                    "reason": "Below threshold"
                })

        return {
            "passed": len(failures) == 0,
            "failures": failures,
            "thresholds": self.thresholds
        }