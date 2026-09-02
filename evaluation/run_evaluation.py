import json

from evaluation.evaluator import (
    context_recall,
    context_precision,
    faithfulness,
    answer_relevancy
)

from evaluation.ragas_evaluator import (
    run_ragas_evaluation
)

from rag_pipeline import run_rag
from rag_setup import initialize_rag

from evaluation.reporter import (
    build_summary,
    find_failures,
    save_json_report,
    save_summary_report,
    save_csv_report,
    print_report
)

# =========================
# Load Synthetic Dataset
# =========================

def load_evaluation_dataset(
    file_path="evaluation/synthetic_dataset.json"
):

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# =========================
# Evaluate RAG
# =========================

def evaluate(
    vectorstore,
    bm25,
    documents,
    generation_chain,
    evaluation_dataset
):

    results = []

    # =====================================
    # STEP 1: Run RAG on every question
    # =====================================

    for i, item in enumerate(
        evaluation_dataset,
        start=1
    ):

        question = item["question"]

        ground_truth = item["ground_truth"]

        print(
            f"\nEvaluating "
            f"{i}/{len(evaluation_dataset)}"
        )

        print(
            f"Question: {question}"
        )

        # -------------------------
        # Run RAG
        # -------------------------

        result = run_rag(
            vectorstore,
            bm25,
            documents,
            generation_chain,
            question
        )

        # -------------------------
        # Context Recall
        # -------------------------

        recall = context_recall(
            ground_truth,
            result["retrieved_context"]
        )

        # -------------------------
        # Context Precision
        # -------------------------

        precision = context_precision(
            question,
            result["retrieved_context"],
            ground_truth
        )

        # -------------------------
        # Faithfulness
        # -------------------------

        faithfulness_result = faithfulness(
            result["answer"],
            result["compressed_context"]
        )

        # -------------------------
        # Answer Relevancy
        # -------------------------

        relevancy_result = answer_relevancy(
            question,
            result["answer"]
        )

        # -------------------------
        # Store Result
        # -------------------------

        results.append({

            "question": question,

            "ground_truth": ground_truth,

            "retrieved_context":
                result["retrieved_context"],

            "compressed_context":
                result["compressed_context"],

            "answer":
                result["answer"],

            "context_recall":
                recall,

            "context_precision":
                precision,

            "faithfulness":
                faithfulness_result["score"],

            "faithfulness_claims":
                faithfulness_result["claims"],

            "answer_relevancy":
                relevancy_result["score"],

            "answer_relevancy_reason":
                relevancy_result["reason"],

            "source":
                item.get(
                    "source",
                    "unknown"
                )
        })

    # =====================================
    # STEP 2: All RAG results collected
    # =====================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "RAG EVALUATION DATA COLLECTED"
    )

    print(
        "=" * 70
    )

    print(
        f"Evaluation results: {len(results)}"
    )

    # =====================================
    # STEP 3: Run RAGAS ONCE
    # =====================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "RAGAS EVALUATION"
    )

    print(
        "=" * 70
    )

    ragas_results = run_ragas_evaluation(
        results
    )

    print(
        "\nRAGAS evaluation completed."
    )

    print(
        ragas_results
    )

    # =====================================
    # Return both results
    # =====================================

    return results, ragas_results


# =========================
# Main
# =========================

if __name__ == "__main__":

    # -------------------------
    # Load Dataset
    # -------------------------

    evaluation_dataset = (
        load_evaluation_dataset()
    )

    print(
        f"Loaded "
        f"{len(evaluation_dataset)} "
        f"evaluation samples."
    )

    # -------------------------
    # Initialize RAG
    # -------------------------

    (
        vectorstore,
        bm25,
        documents,
        generation_chain
    ) = initialize_rag()

    # -------------------------
    # Run Evaluation
    # -------------------------

    results, ragas_results = evaluate(

        vectorstore,

        bm25,

        documents,

        generation_chain,

        evaluation_dataset
    )

    # =====================================
    # Print Local Evaluation Results
    # =====================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "RAG EVALUATION RESULTS"
    )

    print(
        "=" * 70
    )

    for result in results:

        print(
            "\nQuestion:"
        )

        print(
            result["question"]
        )

        print(
            f"\nContext Recall: "
            f"{result['context_recall']:.2f}"
        )

        print(
            f"Context Precision: "
            f"{result['context_precision']:.2f}"
        )

        print(
            f"Faithfulness: "
            f"{result['faithfulness']:.2f}"
        )

        # -------------------------
        # Faithfulness Claims
        # -------------------------

        print(
            "\nFaithfulness Claims:"
        )

        for claim in result[
            "faithfulness_claims"
        ]:

            status = (
                "SUPPORTED"
                if claim["supported"]
                else "NOT SUPPORTED"
            )

            print(
                f"- [{status}] "
                f"{claim['claim']}"
            )

        # -------------------------
        # Answer Relevancy
        # -------------------------

        print(
            f"\nAnswer Relevancy: "
            f"{result['answer_relevancy']:.2f}"
        )

        print(
            f"Reason: "
            f"{result['answer_relevancy_reason']}"
        )

        # -------------------------
        # Ground Truth
        # -------------------------

        print(
            "\nGround Truth:"
        )

        print(
            result["ground_truth"]
        )

        # -------------------------
        # Answer
        # -------------------------

        print(
            "\nAnswer:"
        )

        print(
            result["answer"]
        )

        print(
            "-" * 70
        )

    # =====================================
    # Print RAGAS Results
    # =====================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "RAGAS FINAL RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        ragas_results
    )

     # ========================================================
    # Build Evaluation Summary
    # ========================================================

    summary = build_summary(
        results
    )

    # ========================================================
    # Find Failure Cases
    # ========================================================

    failures = find_failures(
        results
    )

    # ========================================================
    # Save Reports
    # ========================================================

    json_path = save_json_report(
        results
    )

    summary_path = save_summary_report(
        results,
        summary,
        failures
    )

    csv_path = save_csv_report(
        results
    )

    # ========================================================
    # Print Report
    # ========================================================

    print_report(
        summary,
        failures
    )

    # ========================================================
    # Report Locations
    # ========================================================

    print(
        "\nReports generated:"
    )

    print(
        f"Full JSON: "
        f"{json_path}"
    )

    print(
        f"Summary:   "
        f"{summary_path}"
    )

    print(
        f"CSV:       "
        f"{csv_path}"
    )