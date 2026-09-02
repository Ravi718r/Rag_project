from ragas import evaluate

from ragas.dataset_schema import (
    EvaluationDataset,
    SingleTurnSample
)

from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    ContextPrecision,
    ContextRecall
)


# ============================================================
# Convert our evaluation results -> RAGAS EvaluationDataset
# ============================================================

def create_ragas_dataset(evaluation_results):

    samples = []

    for result in evaluation_results:

        # ----------------------------------------------------
        # Get retrieved context
        # ----------------------------------------------------

        retrieved_contexts = result["retrieved_context"]

        # ----------------------------------------------------
        # Make sure retrieved_contexts is list[str]
        # ----------------------------------------------------

        if isinstance(retrieved_contexts, str):

            retrieved_contexts = [
                retrieved_contexts
            ]

        elif isinstance(retrieved_contexts, list):

            flattened_contexts = []

            for context in retrieved_contexts:

                # If accidentally nested
                if isinstance(context, list):

                    flattened_contexts.extend(
                        context
                    )

                else:

                    flattened_contexts.append(
                        str(context)
                    )

            retrieved_contexts = flattened_contexts

        else:

            retrieved_contexts = [
                str(retrieved_contexts)
            ]

        # ----------------------------------------------------
        # Create RAGAS sample
        # ----------------------------------------------------

        sample = SingleTurnSample(

            user_input=str(
                result["question"]
            ),

            retrieved_contexts=retrieved_contexts,

            response=str(
                result["answer"]
            ),

            reference=str(
                result["ground_truth"]
            )
        )

        samples.append(sample)

    print(
        f"RAGAS samples created: {len(samples)}"
    )

    return EvaluationDataset(
        samples=samples
    )


# ============================================================
# Run RAGAS Evaluation
# ============================================================

def run_ragas_evaluation(
    evaluation_results
):

    # --------------------------------------------------------
    # Create dataset
    # --------------------------------------------------------

    dataset = create_ragas_dataset(
        evaluation_results
    )

    print(
        f"RAGAS DATASET SIZE: "
        f"{len(dataset.samples)}"
    )

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if len(dataset.samples) == 0:

        raise ValueError(
            "RAGAS dataset is empty."
        )

    # --------------------------------------------------------
    # Run RAGAS
    # --------------------------------------------------------

    result = evaluate(

        dataset=dataset,

        metrics=[

            Faithfulness(),

            ResponseRelevancy(),

            ContextPrecision(),

            ContextRecall()
        ]
    )

    return result