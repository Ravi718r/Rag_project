from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL

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
        model_name= EMBEDDING_MODEL
    )

    return embedding_model
