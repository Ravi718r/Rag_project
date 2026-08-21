from ingestion_pipeline import load_document
from ingestion_pipeline import split_documents

from embeddings import load_embedding_model
from vectorstore import load_vector_store

from retrievers.hybrid import create_bm25_retriever

from generation import create_generation_chain


def initialize_rag():

    print("Loading documents...")

    documents = load_document()

    print("Splitting documents...")

    documents = split_documents(
        documents
    )

    print("Loading embedding model...")

    embedding_model = load_embedding_model()

    print("Loading vector database...")

    vectorstore = load_vector_store(
        embedding_model
    )

    print("Creating BM25 index...")

    bm25 = create_bm25_retriever(
        documents
    )

    generation_chain = create_generation_chain()

    return (
        vectorstore,
        bm25,
        documents,
        generation_chain
    )