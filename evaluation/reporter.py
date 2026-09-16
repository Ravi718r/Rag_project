import csv
import json
from pathlib import Path
from statistics import mean

# ============================================================
# Configuration
# ============================================================

REPORT_DIR = Path("reports")

REPORT_DIR.mkdir(
    parents= True,
    exist_ok= True
)


# ============================================================
# Helper
# ============================================================

def safe_mean(values):

    """
    Calculate mean while safely ignoring
    missing/invalid values.
    """

    values = [
        value
        for value in values
        if value is not None
    ]

    if not values:
        return 0.0

    return sum(values) / len(values)


def format_score(value):
    """
    Format evaluation score safely.

    None means the evaluator did not
    produce a valid score.
    """

    if value is None:
        return "N/A"

    return f"{value:.2f}"


# ============================================================
# Build Summary
# ============================================================

def build_summary(results):
    """
    Build high-level evaluation statistics
    from per-question evaluation results.
    """

    total_questions = len(results)

    if total_questions == 0:
        return{
            "total_question": 0,
            "metrics": {},
            "failures": {}
        }

    context_recall = [
        result.get("context_recall")
        for result in results
    ]

    context_precision = [
        result.get("context_precision")
        for result in results
    ]


    faithfulness = [
        result.get("faithfulness")
        for result in results
    ]


    answer_relevancy  = [
        result.get("answer_relevancy")
        for result in results
    ]

    # --------------------------------------------------------
    # Evaluation thresholds
    # --------------------------------------------------------

    RECALL_THRESHOLD = 0.5
    PRECISION_THRESHOLD = 0.5
    FAITHFULNESS_THRESHOLD = 0.5
    RELEVANCY_THRESHOLD = 0.5


    # --------------------------------------------------------
    # Failure analysis
    # --------------------------------------------------------

    retrieval_failures = 0
    generation_failures = 0
    completely_failed = 0

    for result in results:

        recall = result.get("context_recall")

        precision = result.get("context_precision")

        faithful = result.get("faithfulness")

        relevancy = result.get("answer_relevancy")

         # --------------------------------------------
        # Retrieval failure
        # --------------------------------------------

        if (
            recall is not None
                and 
            recall < RECALL_THRESHOLD
            ):
            retrieval_failures += 1

        # --------------------------------------------
        # Generation failure
        # --------------------------------------------

        if (
            (
                faithful is not None
                    and
                faithful < FAITHFULNESS_THRESHOLD
            )
            
            or
            (   
                relevancy is not None
                    and 
                relevancy < RELEVANCY_THRESHOLD
            )
        ):
            generation_failures += 1

        # --------------------------------------------
        # Complete failure
        # --------------------------------------------

        if (
            recall is not None
            and precision is not None
            and faithful is not None
            and recall < RECALL_THRESHOLD
            and precision < PRECISION_THRESHOLD
            and faithful < FAITHFULNESS_THRESHOLD
        ):
            completely_failed += 1

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {

        "total_questions": total_questions,

        "metrics": {

            "context_recall": round(
                safe_mean(context_recall),
                4
            ),

        "context_precision": round(
                safe_mean(context_precision),
                4
            ),

            "faithfulness": round(
                safe_mean(faithfulness),
                4
            ),

            "answer_relevancy": round(
                safe_mean(answer_relevancy),
                4
            )
        },

         "failures": {

            "retrieval_failures": retrieval_failures,

            "generation_failures": generation_failures,

            "completely_failed": completely_failed
        }
    }

    return summary


# ============================================================
# Find Failed Questions
# ============================================================

def find_failures(results):
    """
    Identify questions where one or more
    evaluation metrics performed poorly.
    """

    failures = []

    for result in results:

        recall = result.get(
            "context_recall",
            0
        )

        precision = result.get(
            "context_precision",
            0
        )

        faithful = result.get("faithfulness")
        relevancy = result.get("answer_relevancy")

        problems = []

        if recall is not None and recall < 0.5:
            problems.append(
                "low_context_recall"
            )

        if precision is not None and precision < 0.5:
            problems.append(
                "low_context_precision"
            )

        if faithful is not None and faithful < 0.5:
            problems.append(
                "low_faithfulness"
            )

        if relevancy is not None and relevancy < 0.5:
            problems.append(
                "low_answer_relevancy"
            )

        if problems:

            failures.append({

                "question": result.get(
                    "question"
                ),

                "ground_truth": result.get(
                    "ground_truth"
                ),

                "answer": result.get(
                    "answer"
                ),

                "problems": problems,

                "context_recall": recall,

                "context_precision": precision,

                "faithfulness": faithful,

                "answer_relevancy": relevancy
            })

    return failures


# ============================================================
# Save Complete JSON Report
# ============================================================

def save_json_report(results):
    """
    Save complete per-question evaluation results.
    """

    output_path = (
        REPORT_DIR /
        "evaluation_report.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False
        )

    return output_path


# ============================================================
# Save Summary JSON
# ============================================================

def save_summary_report(
    results,
    summary,
    failures
):
    """
    Save high-level summary and failure analysis.
    """

    report = {

        "summary": summary,
        "failures": failures
    }

    output_path = (
        REPORT_DIR /
        "evaluation_summary.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
             indent=4,
            ensure_ascii=False
        )

    return output_path


# ============================================================
# Save CSV
# ============================================================

def save_csv_report(results):
    """
    Save evaluation results in CSV format.

    CSV is useful for:
    - Excel
    - Power BI
    - dashboards
    - data analysis
    """

    output_path = (
        REPORT_DIR /
        "evaluation_results.csv"
    )

    fieldnames = [

        "question",

        "ground_truth",

        "answer",

        "context_recall",

        "context_precision",

        "faithfulness",

        "answer_relevancy",

        "source"
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for result in results:

            writer.writerow({

                "question":
                    result.get(
                        "question",
                        ""
                    ),

                "ground_truth":
                    result.get(
                        "ground_truth",
                        ""
                    ),

                "answer":
                    result.get(
                        "answer",
                        ""
                    ),
                 "context_recall":
                    result.get(
                        "context_recall",
                        0
                    ),

                "context_precision":
                    result.get(
                        "context_precision",
                        0
                    ),

                "faithfulness":
                    result.get(
                        "faithfulness",
                        0
                    ),

                "answer_relevancy":
                    result.get(
                        "answer_relevancy",
                        0
                    ),

                "source":
                    result.get(
                        "source",
                        "unknown"
                    )
            })

    return output_path


# ============================================================
# Console Report
# ============================================================

def print_report(summary, failures):
    """
    Print human-readable evaluation summary.
    """

    metrics = summary["metrics"]
    failure_stats = summary["failures"]

    print("\n")
    print("=" * 70)
    print("RAG EVALUATION SUMMARY")
    print("=" * 70)

    print(
        f"\nDataset Size: "
        f"{summary['total_questions']}"
    )

    print("\nRetrieval Metrics")
    print("-" * 30)

    print(
        f"Context Recall:     "
        f"{metrics['context_recall']:.2f}"
    )

    print(
        f"Context Precision:  "
        f"{metrics['context_precision']:.2f}"
    )

    print("\nGeneration Metrics")
    print("-" * 30)

    print(
        f"Faithfulness:       "
        f"{metrics['faithfulness']:.2f}"
    )

    print(
        f"Answer Relevancy:   "
        f"{metrics['answer_relevancy']:.2f}"
    )

    print("\nFailure Analysis")
    print("-" * 30)

    print(
        f"Retrieval Failures: "
        f"{failure_stats['retrieval_failures']}"
    )

    print(
        f"Generation Failures:"
        f" {failure_stats['generation_failures']}"
    )

    print(
        f"Completely Failed:   "
        f"{failure_stats['completely_failed']}"
    )

    if failures:

        print("\nFailed Questions")
        print("-" * 70)

        for i, failure in enumerate(
            failures,
            start=1
        ):

            print(
                f"\n{i}. "
                f"{failure['question']}"
            )

            print(
                f"Problems: "
                f"{', '.join(failure['problems'])}"
            )

            print(
                f"Recall: " +
                format_score(
                    failure["context_recall"]
                )
            )

            print(
                f"Precision: "+
                format_score(
                    failure["context_precision"]
                )
            )

            print(
                f"Faithfulness: "+
                format_score(
                    failure["faithfulness"]
                )
            )

            print(
                f"Relevancy: "+
                format_score(
                    failure["answer_relevancy"]
                )
            )

    print("\n")
    print("=" * 70)