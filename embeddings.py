from langchain_huggingface import HuggingFaceEmbeddings


# ============================
# Load Embedding Model
# ============================
def load_embedding_model():
    """
    Load the SAME embedding model that was used
    while creating the vector database.
    """

    print("Loading embedding model...")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embedding_model
