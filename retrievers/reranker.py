from sentence_transformers import CrossEncoder

from config import FINAL_K, RERANK_THRESHOLD


# ============================
# Load Cross Encoder
# ============================
reranker = CrossEncoder(
    "BAAI/bge-reranker-base"
)


# ============================
# Re-rank Documents
# ============================
def rerank_documents(
    query,
    documents
):
    """
    Re-rank retrieved documents
    using a Cross Encoder.
    """

    if not documents:
        return []

    pairs = [
        (
            query,
            doc.page_content
        )
        for doc in documents
    ]

    scores = reranker.predict(
        pairs
    )


    doc_scores = sorted(
        zip(
            documents,
            scores
        ),
        key=lambda x: x[1],
        reverse=True
    )

    filtered_docs = [
        doc
        for doc, score in doc_scores
        if score >= RERANK_THRESHOLD
    ]

    
    # # ============================
    # # DEBUG
    # # ============================

    # print("\n===== CROSS ENCODER SCORES =====")

    # for rank, (doc, score) in enumerate(
    #     doc_scores,
    #     start=1
    # ):
    #     print(
    #         f"\nRank: {rank}"
    #         f"\nScore: {score:.4f}"
    #         f"\nContent: {doc.page_content[:200]}"
    #     )

    
    # ============================
    # Return Top K
    # ============================
    return filtered_docs[:FINAL_K]