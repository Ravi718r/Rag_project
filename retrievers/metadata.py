from config import TOP_K
from routers.rule_router import route_question
from retrievers.similarity import create_similarity_retriever
from retrievers.hybrid import hybrid_search
from retrievers.reranker import rerank_documents
from retrievers.compression import compress_documents 


#=============================
# Creating Metadata retriever
#=============================
def create_metadata_retriever(
        vectorstore,
        source
):
    return vectorstore.as_retriever(
        search_kwargs={
            "k": TOP_K,
            "filter": {
                "source": source
            }
        }
    )



# ============================
# Retrieve Document
# ============================
def retrieve_documents(
    vectorstore,
    bm25,
    documents,
    query
):

    # -------------------------
    # Route Query
    # -------------------------

    source = route_question(query)

    print(f"\nRouter selected: {source}")

    # -------------------------
    # Filter Documents
    # -------------------------

    if source is not None:

        filtered_documents = [
            doc 
            for doc in documents
            if doc.metadata.get("source") == source
        ]

        if not filtered_documents:
            return None

    else:

        filtered_documents = documents

    docs = hybrid_search(
        vectorstore,
        bm25,
        filtered_documents,
        query,
        source
    )

    reranked_docs = rerank_documents(
        query,
        docs
    )
    
    
    # =========================
    # Compression
    # =========================
    # compressed_docs = compress_documents(
    #     query,
    #     reranked_docs
    # )

    

    # if not compressed_docs:
    #     return None
        
    return {
        "reranked_docs": reranked_docs,
        "compressed_docs": rerank_documents
    } 




# def retrieve_documents(
#     vectorstore,
#     query
# ):
#     """
#     Retrieve documents based on
#     the routed metadata.
#     """

#     source = route_question(query)

#     if source is not None:

#         retriever = create_metadata_retriever(
#             vectorstore,
#             source
#         )

#     else:

#         retriever = create_similarity_retriever(
#             vectorstore
#         )

#     docs = retriever.invoke(query)

#     if not docs:
#         return None

#     return docs