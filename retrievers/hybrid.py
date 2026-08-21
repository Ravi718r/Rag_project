from rank_bm25 import BM25Okapi

from config import RETRIEVAL_K

def create_bm25_retriever(documents):
    """
    Create a BM25 index for documents.
    """

    tokenized_docs =[
        doc.page_content.lower().split()
        for doc in documents
    ]

    bm25 = BM25Okapi(
        tokenized_docs
    )

    return bm25


def bm25_search(
        bm25,
        documents,
        query
):
    """
    Retrieve top documents
    using BM25
    """

    tokenized_query = query.lower().split()

    scores = bm25.get_scores(
        tokenized_query
    )

    ranked_indices = sorted(
        range(len(scores)),
        key= lambda i: scores[i],
        reverse= True
    )

    top_indices = ranked_indices[:RETRIEVAL_K]

    return [
        documents[i]
        for i in top_indices
    ]



def hybrid_search(
    vectorstore,
    bm25,
    documents,
    query,
    source = None
):
    """
    Perform Hybrid Search using
    Vector Search + BM25
    with Reciprocal Rank Fusion (RRF).
    """

    if source:

        filtered_documets = [
            doc 
            for doc in documents
            if doc.metadata.get("source") == source
        ]

    else:

        filtered_documets = documents

    if not filtered_documets:
        return []

    if source:

        # ==========================
        # Vector Search
        # ==========================
        vector_docs = vectorstore.similarity_search(
            query=query,
            k=RETRIEVAL_K,
            filter={
                "source": source
            }
        )

    else:

        vector_docs = vectorstore.similarity_search(
            query=query,
            k=RETRIEVAL_K
        )

    


    # ==========================
    # BM25 Search
    # ==========================

    filtered_bm25 = create_bm25_retriever(
        filtered_documets
    )

    bm25_docs = bm25_search(
        filtered_bm25,
        filtered_documets,
        query
    )

    # ==========================
    # RRF Initialization
    # ==========================
    rrf_scores = {}

    doc_lookup = {}

    RRF_K = 60

    # ==========================
    # Process Vector Results
    # ==========================
    for rank, doc in enumerate(
        vector_docs,
        start=1
    ):

        content = doc.page_content

        if content not in rrf_scores:
            rrf_scores[content] = 0

        rrf_scores[content] = (
            rrf_scores.get(content, 0)
            + 1 / (RRF_K + rank)
        )

        doc_lookup[content] = doc

    # ==========================
    # Process BM25 Results
    # ==========================
    for rank, doc in enumerate(
        bm25_docs,
        start=1
    ):

        content = doc.page_content

        if content not in rrf_scores:
            rrf_scores[content] = 0

        rrf_scores[content] = (
            rrf_scores.get(content, 0)
            + 1 / (RRF_K + rank)
        )

        doc_lookup[content] = doc

    # ==========================
    # Sort by RRF Score
    # ==========================
    ranked_docs = sorted(
        rrf_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    # ==========================
    # Convert Back to Documents
    # ==========================
    final_docs = []

    for content, score in ranked_docs:

        final_docs.append(
            doc_lookup[content]
        )

    return final_docs[:RETRIEVAL_K]