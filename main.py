from chat import chat
from rag_setup import initialize_rag


def main():

    print("=" * 70)
    print("RAG Retrieval Pipeline")
    print("=" * 70)

    (
        vectorstore,
        bm25,
        documents,
        generation_chain, 
        embedding_model

    ) = initialize_rag()

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