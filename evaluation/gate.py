# # evaluation/gate.py


# DEFAULT_THRESHOLDS = {
#     "context_recall": 0.80,
#     "context_precision": 0.75,
#     "faithfulness": 0.85,
#     "answer_relevancy": 0.80,
# }


# def evaluate_gate(
#     summary,
#     thresholds=None
# ):
#     """
#     Evaluate the RAG system against
#     predefined quality thresholds.
#     """

#     if thresholds is None:
#         thresholds = DEFAULT_THRESHOLDS

#     metrics = summary.get(
#         "metrics",
#         {}
#     )

#     checks = {}

#     for metric, threshold in thresholds.items():

#         score = metrics.get(
#             metric,
#             0
#         )

#         passed = score >= threshold

#         checks[metric] = {
#             "score": score,
#             "threshold": threshold,
#             "passed": passed
#         }

#     overall_passed = all(
#         check["passed"]
#         for check in checks.values()
#     )

#     return {
#         "passed": overall_passed,
#         "checks": checks
#     }