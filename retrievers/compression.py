from langchain_ollama import ChatOllama

from config import MODEL_NAME, COMPRESSION_PROMPT


compression_llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0
)

# ============================
# Compress One Document
# ============================

def compress_document(
    query,
    document
):
    """
    Extract only the parts of a document
    relevant to the query.
    """

    prompt = COMPRESSION_PROMPT.format(
        question=query,
        document=document.page_content
    )

    response = compression_llm.invoke(
        prompt
    )

    content = response.content.strip()

    if content.upper() == "EMPTY":
        return ""

    return content

# ============================
# Compress Documents
# ============================

def compress_documents(
    query,
    documents
):
    """
    Compress all retrieved documents
    based on the query.
    """

    compressed_documents = []

    for document in documents:

        compressed_text = compress_document(
            query,
            document
        )

        if compressed_text:

            compressed_document = document.model_copy(
                update={
                    "page_content": compressed_text
                }
            )

            compressed_documents.append(
                compressed_document
            )

    return compressed_documents