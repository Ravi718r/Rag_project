from sentence_transformers import CrossEncoder

from config import FINAL_K


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

    # print("\n===== Cross Encoder Scores =====")

    # for doc, score in zip(documents, scores):
    #     print(f"\nScore : {score:.4f}")
    #     print(doc.page_content[:150])

    doc_scores = list(
        zip(
            documents,
            scores
        )
    )

    doc_scores = sorted(
        doc_scores,
        key=lambda x: x[1],
        reverse=True
    )

    # print("\n===== After Re-ranking =====")

    # for doc, score in doc_scores:
    #     print(f"\nScore : {score:.4f}")
    #     print(doc.page_content[:150])

    return [
        doc
        for doc, score in doc_scores[:FINAL_K]
    ]