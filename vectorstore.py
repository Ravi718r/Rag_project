from langchain_chroma import Chroma

from config import PERSIST_DIRECTORY

# ============================
# Load Existing Chroma Database
# ============================
def load_vector_store(
    embedding_model,
    persist_directory=PERSIST_DIRECTORY,
):
    """
    Load the existing Chroma vector database.
    """

    print("Loading Chroma Vector Database...")

    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY ,
        embedding_function=embedding_model
    )

    print("Vector Database Loaded Successfully!\n")

    return vectorstore
