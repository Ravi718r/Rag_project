def context_recall(
    ground_truth,
    retrieved_context
):
    """
    Simple Context Recall implementation.

    Checks how much of the ground-truth
    information appears in the retrieved context.
    
    """

    ground_truth_words = set(
        ground_truth.lower().split()
    )

    retrieved_text = " ".join(
        retrieved_context
    ).lower()

    matched_words = [
        word
        for word in ground_truth_words
        if word in retrieved_text
    ]

    if not ground_truth_words:
        return 0.0

    return (
        len(matched_words)
        / len(ground_truth_words)
    )


def context_precision(
    question,
    retrieved_context,
    ground_truth
):
    """
    Simple Context Precision implementation.

    Measures how many retrieved chunks
    contain information relevant to the
    ground-truth answer.
    """

    if not retrieved_context:
        return 0.0

    ground_truth_words = set(
        ground_truth.lower().split()
    )

    relevant_chunks = 0

    for chunk in retrieved_context:

        chunk_words = set(
            chunk.lower().split()
        )

        overlap = (
            ground_truth_words
            & chunk_words
        )

        if overlap:
            relevant_chunks += 1

    return (
        relevant_chunks
        / len(retrieved_context)
    )