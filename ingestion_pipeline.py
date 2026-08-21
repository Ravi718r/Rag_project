import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings




load_dotenv()

def load_document(docs_path="docs"):
    """Load all text files from the docs directory"""
    print(f"Laoding documents form {docs_path}...")

    # Check if directory docs exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} doesn't exist. Please create it and add your company files.")


    # Load all .txt files form the docs directory 
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={
            "encoding": "utf-8"
        }
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your compnay documents")

    # print(documents[0].metadata)
    # print(documents[0].page_content[:200])
    # print("XXXX")

    # for i, doc in enumerate(documents[:2]):
    #     print(f"\nDocument {i+1}:")
    #     print(f" Source: {doc.metadata['source']}")
    #     print(f" Content length: {len(doc.page_content)} characters")
    #     print(f" Content preview: {doc.page_content[:100]}... ")
    #     print(f" metadata: {doc.metadata}")

    return documents



def split_documents(documents, chunk_size=400, chunk_overlap=80):
    """Split documents into smaller chunks with overlap"""
    print("Splitting documents into chunks....")

    text_splitter= RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap= chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            " ",
            ""
        ]
    )

    chunks = text_splitter.split_documents(documents)

    # if chunks:

    #     for i, chunk in enumerate(chunks[:5]):
    #         print(f"\n--- Chunk {i+1} ---")
    #         print(f"Source: {chunk.metadata['source']}")
    #         print(f"Length: {len(chunk.page_content)} characters")
    #         print(f"Content:")
    #         print(chunk.page_content)
    #         print("-" * 50)

    #     if len(chunks) > 5:
    #         print(f"\n... and {len(chunks)-5} more chunks")
    
    return chunks


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDb vector store"""
    print("Creating embedding and storing in ChromaDB...")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create ChromaDB vectore store
    print("--- Creating Vector Store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print("--- Finished creating vectore store ---")

    print("Vector store created and saved to {persist_directory}")
    return vectorstore

def main():

    print(" Main Function ")

    documents = load_document(docs_path="docs")

    chunks = split_documents(documents)

    vectorstore = create_vector_store(chunks)

    print("Database created successfully!")


if __name__ == "__main__":
    main()