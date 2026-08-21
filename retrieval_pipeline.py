from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnablePassthrough
from config import (
    MODEL_NAME,
    TOP_K,
    PERSIST_DIRECTORY,
    EMBEDDING_MODEL,
    SIMILARITY_THRESHOLD,
    llm,
    PROMPT,
    parser,
)
from retrievers.retrieval import (
    load_embedding_model,
    load_vector_store,
    format_docs,
    retrieve_documents,
)

from generation import create_generation_chain

# MODEL_NAME = "llama3.2:3b"
# TOP_K = 3
# PERSIST_DIRECTORY = "db/chroma_db"
# EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# SIMILARITY_THRESHOLD = 0.45

# llm = ChatOllama(
#     model =MODEL_NAME,
#     temperature=0
# )

# PROMPT = ChatPromptTemplate.from_template(
# """
# You are a helpful AI assistant.

# Answer ONLY using the provided context.

# If the answer is not present in the context,
# say "I don't know."

# Context:
# {context}

# Question:
# {question}
# """
# )



# parser = StrOutputParser()


# # ============================
# # Load Embedding Model
# # ============================
# def load_embedding_model():
#     """
#     Load the SAME embedding model that was used
#     while creating the vector database.
#     """

#     print("Loading embedding model...")

#     embedding_model = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2"
#     )

#     return embedding_model



# # ============================
# # Load Existing Chroma Database
# # ============================
# def load_vector_store(
#     embedding_model,
#     persist_directory=PERSIST_DIRECTORY,
# ):
#     """
#     Load the existing Chroma vector database.
#     """

#     print("Loading Chroma Vector Database...")

#     vectorstore = Chroma(
#         persist_directory=PERSIST_DIRECTORY ,
#         embedding_function=embedding_model
#     )

#     print("Vector Database Loaded Successfully!\n")

#     return vectorstore



# ============================
# Format Docs
# ============================
def format_docs(
    docs
):
    """
    Convert retrieved documents
    into one context string.
    """

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )

document_formatter = RunnableLambda(
    format_docs
)




# # Seprate Retrieval Strategy 
# def create_similarity_retriever(vectorstore):
#     return vectorstore.as_retriever(
#         search_type = "similarity",
#         search_kwargs={
#             "k":3
#         }
#     )

# # MMR Retriever  
# def create_mmr_retriever(vectorstore):
#     return vectorstore.as_retriever(
#         search_type = "mmr",
#         search_kwargs={
#             "k":3,
#             "fetch_k": 10
#         }
#     )

# compare them -> Similarity & MMR
def compare_retrievers(
    similarity_retriever,
    mmr_retriever
):
    query = input("Question: ")

    print("\n========== Similarity ==========")

    docs = similarity_retriever.invoke(query)

    for i, doc in enumerate(docs, start=1):
        print(f"\nChunk {i}")
        print(doc.page_content[:300])

    print("\n========== MMR ==========")

    docs = mmr_retriever.invoke(query)

    for i, doc in enumerate(docs, start=1):
        print(f"\nChunk {i}")
        print(doc.page_content[:300])


# Create Retrieval Chain
def create_retrieval_chain(
        vectorstore
):
    """
    Create the complete RAG retrieval chain
    """

    def threshold_retriever(query):
    
            results = vectorstore.similarity_search_with_score(
                query=query,
                k = TOP_K
            )
    
            relevant_docs = []
    
            for doc, score in results:
    
                if score <= SIMILARITY_THRESHOLD:
                    relevant_docs.append(doc)
    
            return relevant_docs
    threshold_runnable = RunnableLambda(
        threshold_retriever
    )

    retrieval_chain = (
        {
            "context":
                threshold_runnable
                | document_formatter,

            "question":
                RunnablePassthrough()
        }
        | PROMPT
        | llm
        | parser
    )

    return retrieval_chain


def debug_retrieval(vectorstore):
    """
    Debug retrieval by displaying similarity scores
    for the retrieved documents.
    """

    print("\n=== Retrieval Debug Mode ===")

    query = input("\nAsk your question: ")

    results = vectorstore.similarity_search_with_score(
        query=query,
        k=3
    )

    print("\n" + "=" * 80)
    print(f"Question: {query}")
    print("=" * 80)

    for rank, (doc, score) in enumerate(results, start=1):

        print(f"\nRank : {rank}")
        print(f"Similarity Score : {score:.4f}")
        print(f"Source : {doc.metadata.get('source', 'Unknown')}")

        print("\nRetrieved Chunk:\n")

        print(doc.page_content)

        print("-" * 80)


# #=============================
# # Creating Metadata retriever
# #=============================
# def create_metadata_retriever(
#         vectorstore,
#         source
# ):
#     return vectorstore.as_retriever(
#         search_kwargs={
#             "k": TOP_K,
#             "filter": {
#                 "source": source
#             }
#         }
#     )


# #=============================
# # Router 
# #=============================
# def route_question(query):

#     query = query.lower()

#     if "google" in query:
#         return "docs\\Google.txt"
    
#     elif "tesla" in query:
#         return "docs\\Tesla.txt"
    
#     elif "spacex" in query:
#         return "docs\\SpaceX.txt"
    
#     elif "microsoft" in query:
#         return "docs\\Microsoft.txt"
    
#     elif "nvidia" in query:
#         return "docs\\Nvidia.txt"


# # ============================
# # Retrieve Document
# # ============================
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



# # ============================
# # Creating a Generation Chain
# # ============================

# def create_generation_chain():
#     """
#     Chain responsible only for answer generation.
#     """

#     generation_chain = (
#         PROMPT
#         | llm
#         | parser
#     )

#     return generation_chain


# =========================
# Chat
# =========================
def chat(
    vectorstore,
    generation_chain
):

    while True:

        query = input(
            "\nYou: "
        ).strip()

        if query.lower() in [
            "/bye",
            "bye",
            "exit",
            "quit"
        ]:
            print("\nGoodbye!")
            break

        
        docs = retrieve_documents(vectorstore, query)

        if docs is None:
            print("Assistant:")
            print("I couldn't find any relevant information.")
            continue

        context = format_docs(
            docs
        )

        answer = generation_chain.invoke(
            {
                "context": context,
                "question": query
            }
        ) 

        print("\nAssistant:")
        print(answer)
    
# ============================
# Main Function
# ============================
def main():

    print("=" * 70)
    print("RAG Retrieval Pipeline")
    print("=" * 70)

    embedding_model = load_embedding_model()

    vectorstore = load_vector_store(
        embedding_model
    )

    generation_chain = create_generation_chain()

    # retriever = create_metadata_retriever(
    #     vectorstore,
    #     source= rou)
    # )

    # docs = retriever.invoke(
    #     "When was Tesla founded?"
    # )

    # for doc in docs:
    #     print(doc.metadata)
    #     print(doc.page_content)

    
    # similarity_retriever = create_similarity_retriever(
    #     vectorstore
    # )


    # mmr_retriever = create_mmr_retriever(
    #     vectorstore
    # )

    # compare_retrievers(
    #     similarity_retriever,
    #     mmr_retriever
    # )

     

    chat(
        vectorstore,
        generation_chain
    )


if __name__ == "__main__":
    main()