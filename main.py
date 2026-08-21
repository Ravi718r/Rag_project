from chat import chat
from rag_setup import initialize_rag




def main():

    print("=" * 70)
    print("RAG Retrieval Pipeline")
    print("=" * 70)


    # # ----------------------------
    # # Load Documents
    # # ----------------------------
    # documents = load_document()


    # # ----------------------------
    # # Split Documents into Chunks
    # # ----------------------------
    # documents = split_documents(
    #     documents
    # )

    
    # # ----------------------------
    # # Load Embedding Model
    # # ----------------------------
    # embedding_model = load_embedding_model()



    # # ----------------------------
    # # Load Existing Chroma Database
    # # ----------------------------
    # vectorstore = load_vector_store(
    #     embedding_model
    # )



    # # ----------------------------
    # # Create BM25 Index
    # # ----------------------------
    # bm25 = create_bm25_retriever(
    #     documents
    # )


    (
        vectorstore,
        bm25,
        documents,
        generation_chain

    ) = initialize_rag()



    # # ----------------------------
    # # Generation Chain
    # # ----------------------------
    # generation_chain = create_generation_chain()



    # ----------------------------
    # Start Chat
    # ----------------------------
    chat(
        vectorstore,
        bm25,
        documents,
        generation_chain
    )


if __name__ == "__main__":
    main()