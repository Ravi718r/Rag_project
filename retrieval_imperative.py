from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


SIMILARITY_THRESHOLD = 0.45

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

PROMPT = ChatPromptTemplate.from_template(
"""
You are a helpful AI assistant.

Answer ONLY using the provided context.

If the answer is not present in the context,
say "I don't know."

Context:
{context}

Question:
{question}
"""
)

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


# ============================
# Load Existing Chroma Database
# ============================

def load_vector_store(
    embedding_model,
    persist_directory="db/chroma_db"
):
    """
    Load the existing Chroma vector database.
    """

    print("Loading Chroma Vector Database...")

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )

    print("Vector Database Loaded Successfully!\n")

    return vectorstore


# ============================
# Retrieve Context with threshold
# ============================
def retrieve_context_with_threshold(
        vectorestore,
        query
):
    
    """
    Retrieve documents only if they pass
    the similarity threshold.

    """

    results = vectorestore.similarity_search_with_score(
        query=query,
        k=3
    )

    relevant_docs = []

    for doc, score in results:

        print(f"Score : {score:.4f}")

        if score < SIMILARITY_THRESHOLD:
            relevant_docs.append(doc)

    if not relevant_docs:
        return None

    return relevant_docs


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


# ============================
# Generate Answer
# ============================
def generate_answer(
        context,
        query
):

    """
    Generate Using LCEL chain
    """

    final_prompt = PROMPT.invoke(
        {
            "context": context,
            "question": query
        }
    )

    response = llm.invoke(final_prompt)

    # response = chain.invoke(
    #     {
    #         "context": context,
    #         "question": query
    #     }
    # )

    return response.content


# ============================
# Answer Question
# ============================
def answer_question(
        vectorestore,
        # retriever,
        query
):
    docs = retrieve_context_with_threshold(
        vectorestore,
        query
    )

    if docs is None:
        return "❌ I couldn't find any relevant information."

    context = format_docs(docs)

    if context is None:
        return "❌ I couldn't find any relevant information in the knowledge base."


    answer = generate_answer(
        context,
        query
    )

    return answer



# =========================
# Chat
# =========================
def chat(
    vectorstore
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

        answer = answer_question(
            vectorstore,
            query
        )

        # answer = retrieval_chain.invoke(
        #     query
        # ) 
               
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
    
    chat(
        vectorstore
    )

if __name__ == "__main__":
    main()