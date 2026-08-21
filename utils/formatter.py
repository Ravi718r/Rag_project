from langchain_core.runnables import RunnableLambda


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

# document_formatter = RunnableLambda(
#     format_docs
# )