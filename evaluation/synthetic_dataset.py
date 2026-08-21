import json

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


MODEL_NAME = "llama3.2:3b"


llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0
)


PROMPT = ChatPromptTemplate.from_template(
    """
You are creating an evaluation dataset for a RAG system.

Read the document below and generate ONE question
that can be answered ONLY using the information
contained in the document.

Rules:

1. The question must be answerable from the document.
2. Do not use outside knowledge.
3. The answer must be explicitly supported by the document.
4. Make the question specific and useful for RAG evaluation.
5. Do not create a yes/no question.
6. Return ONLY valid JSON.
7. Do not use markdown.

Required JSON format:

{{
    "question": "...",
    "ground_truth": "..."
}}

Document:
{document}
"""
)


parser = StrOutputParser()


generation_chain = (
    PROMPT
    | llm
    | parser
)


def generate_question(document):

    response = generation_chain.invoke(
        {
            "document": document.page_content
        }
    )

    return response


def parse_response(response):

    try:

        data = json.loads(response)

        return {
            "question": data["question"],
            "ground_truth": data["ground_truth"]
        }

    except (
        json.JSONDecodeError,
        KeyError
    ):

        return None


def create_synthetic_dataset(
    documents,
    output_file="evaluation/synthetic_dataset.json"
):

    dataset = []

    for i, document in enumerate(documents):

        print(
            f"Generating question "
            f"{i + 1}/{len(documents)}..."
        )

        response = generate_question(
            document
        )

        data = parse_response(
            response
        )

        if data is None:

            print(
                "Invalid LLM response. "
                "Skipping..."
            )

            continue

        data["source"] = document.metadata.get(
            "source",
            "unknown"
        )

        dataset.append(data)

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            dataset,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(
        f"\nGenerated {len(dataset)} "
        f"evaluation samples."
    )

    print(
        f"Saved to: {output_file}"
    )

    return dataset

# =========================
# Run Dataset Generation
# ========================

if __name__ == "__main__":

    from rag_setup import initialize_rag

    (
        vectorstore,
        bm25,
        documents,
        generation_chain
    ) = initialize_rag()

    create_synthetic_dataset(
        documents[:20]
    )