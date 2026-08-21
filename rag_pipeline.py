from utils.formatter import format_docs
from retrievers.metadata import retrieve_documents

def run_rag(
    vectorstore,
    bm25,
    documents,
    generation_chain,
    query
):
    # =========================
    # Retrieve + rerank + compression
    # =========================
    result = retrieve_documents(
        vectorstore,
        bm25,
        documents,
        query
    )

    if result is None:
        return {
            "question": query,
            "retrieved_context": [],
            "compressed_context": [],
            "answer": "I don't know."
        }

    reranked_docs = result["reranked_docs"]
    compressed_docs = result["compressed_docs"]

    # =========================
    # Context
    # =========================

    context = format_docs(
        compressed_docs
    )

    # =========================
    # Generation
    # =========================

    answer = generation_chain.invoke(
        {
            "context": context,
            "question": query
        }
    )

    # =========================
    # Return Everything
    # =========================

    return {

        "question": query,

        "retrieved_context": [
            doc.page_content
            for doc in reranked_docs
        ],

        "compressed_context": [
            doc.page_content
            for doc in compressed_docs
        ],
        
        "answer": answer
    }