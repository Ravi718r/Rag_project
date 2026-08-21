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