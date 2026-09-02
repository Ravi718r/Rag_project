def context_recall(
    ground_truth,
    retrieved_context
):
    """
    Simple Context Recall implementation.

    Checks how much of the ground-truth
    information appears in the retrieved context.
    
    """

    ground_truth_words = set(
        ground_truth.lower().split()
    )

    retrieved_text = " ".join(
        retrieved_context
    ).lower()

    matched_words = [
        word
        for word in ground_truth_words
        if word in retrieved_text
    ]

    if not ground_truth_words:
        return 0.0

    return (
        len(matched_words)
        / len(ground_truth_words)
    )


def context_precision(
    question,
    retrieved_context,
    ground_truth
):
    """
    Simple Context Precision implementation.

    Measures how many retrieved chunks
    contain information relevant to the
    ground-truth answer.
    """

    if not retrieved_context:
        return 0.0

    ground_truth_words = set(
        ground_truth.lower().split()
    )

    relevant_chunks = 0

    for chunk in retrieved_context:

        chunk_words = set(
            chunk.lower().split()
        )

        overlap = (
            ground_truth_words
            & chunk_words
        )

        if overlap:
            relevant_chunks += 1

    return (
        relevant_chunks
        / len(retrieved_context)
    )

import json

from langchain_ollama import ChatOllama 
from langchain_core.prompts import ChatPromptTemplate
from config import MODEL_NAME

evaluation_llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0
)

faithfulness_prompt = ChatPromptTemplate.from_template(
    """
You are evaluating the factual faithfulness of a RAG answer.

Your job is to determine whether each factual claim
in the answer is supported by the provided context.

Context:
{context}

Answer:
{answer}

Instructions:

1. Identify every factual claim in the answer.
2. For each claim, determine whether it is directly
   supported by the context.
3. Do NOT use outside knowledge.
4. A claim is supported only if the context provides
   enough information to justify it.
5. Ignore statements such as:
   "I don't know."
   "I cannot answer."
6. Return ONLY valid JSON.

Required format:

{{
    "claims": [
        {{
            "claim": "...",
            "supported": true
        }}
    ]
    }}
"""
)


def faithfulness(
        answer,
        compressed_context
): 

    if not answer or not compressed_context:
        return {
            "score": 0.0,
            "claims": []
        }

    context = "\n\n".join(
        compressed_context
    )

    response = evaluation_llm.invoke(
        faithfulness_prompt.format(
            context=context,
            answer=answer
        )
    )

    raw_response = response.content.strip()

    try:

        data = json.loads(
            raw_response
        )

        claims = data.get(
            "claims",
            []
        )

        if not claims:

            return {
                "score": 1.0,
                "claims": []
            }
        supported_claims = sum(
            1
            for claim in claims
            if claim.get(
                "supported",
                False
            )
        )

        score = (
            supported_claims
            / len(claims)
        )

        return {
            "score": score,
            "claims": claims
        }
    except (
        json.JSONDecodeError,
        TypeError
    ):

        return {
            "score": 0.0,
            "claims": []
        }



#=======================
# Answer Relevancy
#=======================
answer_relevancy_prompt = ChatPromptTemplate.from_template(
    """
You are evaluating the relevance of a RAG answer.

Question:
{question}

Answer:
{answer}

Determine whether the answer directly addresses
the user's question.

Rules:

1. Give 1.0 if the answer directly answers the question.
2. Give 0.5 if the answer partially addresses the question.
3. Give 0.0 if the answer does not answer the question.
4. "I don't know" should receive 0.0 when the question
   expects information that could be answered.
5. Do not judge whether the answer is factually correct.
6. Return ONLY valid JSON.

Required format:

{{
    "score": 0.0,
    "reason": "..."
}}
"""
)


def answer_relevancy(
    question,
    answer
):

    if not answer:
        return {
            "score": 0.0,
            "reason": "No answer was generated."
        }

    response = evaluation_llm.invoke(
        answer_relevancy_prompt.format(
            question=question,
            answer=answer
        )
    )

    raw_response = response.content.strip()

    try:

        data = json.loads(
            raw_response
        )

        score = float(
            data.get("score", 0.0)
        )

        score = max(
            0.0,
            min(1.0, score)
        )

        return {
            "score": score,
            "reason": data.get(
                "reason",
                ""
            )
        }

    except (
        json.JSONDecodeError,
        TypeError,
        ValueError
    ):

        return {
            "score": 0.0,
            "reason": "Invalid evaluator response."
        }