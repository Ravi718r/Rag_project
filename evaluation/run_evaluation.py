import json

from evaluation.evaluator import context_recall
from rag_pipeline import run_rag
from rag_setup import initialize_rag


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

            "source":
                item.get(
                    "source",
                    "unknown"
                )
        })

    return results


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

    results = evaluate(

        vectorstore,

        bm25,

        documents,

        generation_chain,

        evaluation_dataset
    )

    # -------------------------
    # Print Results
    # -------------------------

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
            "\nGround Truth:"
        )

        print(
            result["ground_truth"]
        )

        print(
            "\nAnswer:"
        )

        print(
            result["answer"]
        )

        print(
            "-" * 70
        )