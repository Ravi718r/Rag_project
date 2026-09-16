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

import re
from typing import Any

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from config import MODEL_NAME


# ============================================================
# EVALUATION LLM
# ============================================================

evaluation_llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0,
)


# ============================================================
# TEXT NORMALIZATION HELPERS
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize text for comparison.

    Example:
        "Google, LLC." -> "google llc"
        "14%" -> "14"
    """

    if not text:
        return ""

    text = str(text).lower()

    # Normalize curly apostrophes
    text = text.replace("’", "'")

    # Remove punctuation
    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


def normalize_number_words(text: str) -> str:
    """
    Convert common number words into digits.

    Example:
        fourteen percent -> 14 percent
        twenty -> 20
    """

    if not text:
        return ""

    replacements = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
        "eleven": "11",
        "twelve": "12",
        "thirteen": "13",
        "fourteen": "14",
        "fifteen": "15",
        "sixteen": "16",
        "seventeen": "17",
        "eighteen": "18",
        "nineteen": "19",
        "twenty": "20",
    }

    text = normalize_text(text)

    for word, number in replacements.items():

        text = re.sub(
            rf"\b{word}\b",
            number,
            text,
        )

    return text


def normalized_tokens(text: str) -> set[str]:
    """
    Return normalized tokens.
    """

    text = normalize_number_words(text)

    if not text:
        return set()

    return set(text.split())


# ============================================================
# ROBUST JSON PARSER
# ============================================================

def parse_json_response(
    raw_response: Any,
):
    """
    Parse JSON returned by the evaluation LLM.

    Handles:
        - normal JSON
        - ```json ... ```
        - extra text surrounding JSON
    """

    if raw_response is None:
        return None

    if not isinstance(
        raw_response,
        str,
    ):
        raw_response = str(
            raw_response
        )

    raw_response = raw_response.strip()

    if not raw_response:
        return None

    # --------------------------------------------------------
    # Direct JSON
    # --------------------------------------------------------

    try:

        return json.loads(
            raw_response
        )

    except json.JSONDecodeError:
        pass

    # --------------------------------------------------------
    # Remove markdown code fences
    # --------------------------------------------------------

    cleaned = re.sub(
        r"```(?:json)?",
        "",
        raw_response,
        flags=re.IGNORECASE,
    )

    cleaned = cleaned.replace(
        "```",
        "",
    ).strip()

    try:

        return json.loads(
            cleaned
        )

    except json.JSONDecodeError:
        pass

    # --------------------------------------------------------
    # Extract JSON object from surrounding text
    # --------------------------------------------------------

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start != -1 and end != -1:

        candidate = cleaned[
            start:end + 1
        ]

        try:

            return json.loads(
                candidate
            )

        except json.JSONDecodeError:
            pass

    return None



faithfulness_prompt = ChatPromptTemplate.from_template(
    """
You are a strict evaluator of factual faithfulness in a RAG system.

Your job has TWO steps:

STEP 1:
Extract factual claims from the ANSWER.

STEP 2:
For each extracted claim, determine whether the CONTEXT directly supports it.

CRITICAL RULE:

A claim MUST NOT contain information that was not explicitly stated
in the ANSWER.

NEVER expand, complete, infer, or rewrite a short answer.

Examples:

ANSWER:
2006.

Correct claim:
"2006"

Incorrect claims:
"Scott Hassan founded Willow Garage in 2006."
"Scott Hassan founded Willow Garage."
"Hassan founded Willow Garage in 2006."

The incorrect claims are invalid because the ANSWER only says "2006".

Another example:

ANSWER:
Scott Hassan.

Correct claim:
"Scott Hassan"

Do NOT transform it into:
"Scott Hassan was the original lead programmer of Google Search."

Another example:

ANSWER:
14%

Correct claim:
"14%"

Do NOT transform it into:
"Larry Page and Sergey Brin own 14% of Google's publicly listed shares."

Another example:

ANSWER:
I don't know.

Correct result:
{{"claims": []}}

CONTEXT:
{context}

ANSWER:
{answer}

Rules:

1. Extract claims ONLY from the ANSWER.

2. Every claim must be explicitly represented in the ANSWER.

3. Never add information from the CONTEXT into a claim.

4. Never expand a short answer.

5. Never infer what the answer means.

6. Never use the ground truth.

7. Never use outside knowledge.

8. A claim is supported only when the CONTEXT directly supports
   the information expressed by that claim.

9. If the answer is "I don't know", "unknown", "not sure",
   or another non-answer, return an empty claims list.

10. Return ONLY valid JSON.

Required format:

{{
    "claims": [
        {{
            "claim": "exact information expressed by the answer",
            "supported": true
        }}
    ]
}}
"""
)


def faithfulness(
    answer,
    compressed_context,
):
    """
    Evaluate factual faithfulness of the generated answer.

    Important:
    Short answers are treated as atomic claims so that the
    evaluator cannot expand them into invented statements.

    Returns:
        score: float | None
        claims: list
        status: str
        reason: str
    """

    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    if not answer or not answer.strip():

        return {
            "score": None,
            "claims": [],
            "status": "evaluation_error",
            "reason": "Answer is empty.",
        }

    if not compressed_context:

        return {
            "score": None,
            "claims": [],
            "status": "evaluation_error",
            "reason": "No context available for evaluation.",
        }

    # ========================================================
    # NORMALIZE ANSWER
    # ========================================================

    normalized_answer = normalize_text(
        answer
    )

    # ========================================================
    # EXPLICIT NON-ANSWER
    # ========================================================

    non_answer_patterns = {
        "i dont know",
        "i do not know",
        "dont know",
        "unknown",
        "not sure",
        "cannot determine",
        "cant determine",
        "unable to determine",
        "no information available",
        "no information",
    }

    if normalized_answer in non_answer_patterns:

        return {
            "score": None,
            "claims": [],
            "status": "not_applicable",
            "reason": "Answer contains no factual claim.",
        }

    # ========================================================
    # SHORT ANSWER HANDLING
    # ========================================================
    #
    # This is the important fix.
    #
    # Examples:
    #
    # 2006.
    # 14%.
    # Nest.
    # X.
    # Scott Hassan.
    #
    # We don't ask the small local LLM to interpret these.
    # The complete answer is treated as one atomic claim.
    #
    # ========================================================

    cleaned_answer = answer.strip()

    normalized_cleaned_answer = normalize_text(
        cleaned_answer
    )

    # Remove trailing punctuation for simple atomic answers
    atomic_answer = re.sub(
        r"[.!?,:;]+$",
        "",
        cleaned_answer,
    ).strip()

    atomic_normalized = normalize_text(
        atomic_answer
    )

    # Detect simple short factual answers.
    #
    # We intentionally keep this conservative.
    words = atomic_normalized.split()

    is_short_answer = (
        len(words) <= 3
        and len(atomic_answer) <= 80
    )

    if is_short_answer:

        context = "\n\n".join(
            str(chunk)
            for chunk in compressed_context
        )

        # ----------------------------------------------------
        # Determine whether the exact answer is represented
        # in the context.
        #
        # Number-word normalization handles:
        #
        #     fourteen
        #     14
        #
        # ----------------------------------------------------

        answer_for_matching = normalize_number_words(
            atomic_answer
        )

        context_for_matching = normalize_number_words(
            context
        )

        supported = (
            answer_for_matching.lower()
            in context_for_matching.lower()
        )

        return {
            "score": 1.0 if supported else 0.0,
            "claims": [
                {
                    "claim": atomic_answer,
                    "supported": supported,
                }
            ],
            "status": "success",
            "reason": "",
        }

    # ========================================================
    # PREPARE CONTEXT
    # ========================================================

    context = "\n\n".join(
        str(chunk)
        for chunk in compressed_context
    )

    # ========================================================
    # CALL EVALUATOR FOR LONGER ANSWERS
    # ========================================================

    try:

        response = evaluation_llm.invoke(
            faithfulness_prompt.format(
                context=context,
                answer=answer,
            )
        )

    except Exception as e:

        return {
            "score": None,
            "claims": [],
            "status": "evaluation_error",
            "reason": (
                f"Evaluator invocation failed: {str(e)}"
            ),
        }

    # ========================================================
    # PARSE JSON
    # ========================================================

    data = parse_json_response(
        response.content
    )

    if not isinstance(data, dict):

        return {
            "score": None,
            "claims": [],
            "status": "evaluation_error",
            "reason": "Evaluator returned invalid JSON.",
        }

    claims = data.get(
        "claims"
    )

    if not isinstance(
        claims,
        list,
    ):

        return {
            "score": None,
            "claims": [],
            "status": "evaluation_error",
            "reason": "Invalid claims structure.",
        }

    # ========================================================
    # NO CLAIMS
    # ========================================================

    if not claims:

        return {
            "score": None,
            "claims": [],
            "status": "not_applicable",
            "reason": "Answer contains no factual claims.",
        }

    # ========================================================
    # VALIDATE CLAIMS
    # ========================================================

    valid_claims = []

    normalized_answer_for_validation = normalize_number_words(
        answer
    )

    for claim in claims:

        if not isinstance(
            claim,
            dict,
        ):
            continue

        claim_text = claim.get(
            "claim"
        )

        supported = claim.get(
            "supported"
        )

        if not isinstance(
            claim_text,
            str,
        ):
            continue

        claim_text = claim_text.strip()

        if not claim_text:
            continue

        if not isinstance(
            supported,
            bool,
        ):
            continue

        # ----------------------------------------------------
        # Normalize claim and answer.
        # ----------------------------------------------------

        normalized_claim = normalize_number_words(
            claim_text
        )

        normalized_claim = normalize_text(
            normalized_claim
        )

        normalized_answer_text = normalize_text(
            normalized_answer_for_validation
        )

        # ----------------------------------------------------
        # Strict containment check.
        #
        # The evaluator is NOT allowed to invent a claim.
        # ----------------------------------------------------

        if normalized_claim not in normalized_answer_text:
            continue

        valid_claims.append(
            {
                "claim": claim_text,
                "supported": supported,
            }
        )

    # ========================================================
    # INVALID / INVENTED CLAIMS
    # ========================================================

    if not valid_claims:

        return {
            "score": None,
            "claims": [],
            "status": "evaluation_error",
            "reason": (
                "Evaluator produced claims that were not "
                "explicitly present in the answer."
            ),
        }

    # ========================================================
    # CALCULATE SCORE
    # ========================================================

    supported_claims = sum(
        1
        for claim in valid_claims
        if claim["supported"]
    )

    score = (
        supported_claims
        / len(valid_claims)
    )

    return {
        "score": round(score, 4),
        "claims": valid_claims,
        "status": "success",
        "reason": "",
    }



answer_relevancy_prompt = ChatPromptTemplate.from_template(
    """
You are an evaluation model for a Retrieval-Augmented Generation (RAG) system.

Your task is to evaluate ONLY whether the ANSWER directly addresses
the QUESTION.

Question:
{question}

Answer:
{answer}

Evaluation rules:

1. Give 1.0 when the answer provides the information requested
   by the question.

2. Short answers can receive 1.0.
   For example:

   Question: Who was the original lead programmer?
   Answer: Scott Hassan.

   Score: 1.0

3. A direct name, number, date, location, or short phrase can be
   a completely relevant answer when that is what the question asks for.

4. Give 0.5 only when the answer partially addresses the question
   but does not fully provide the requested information.

5. Give 0.0 when the answer does not address the question.

6. "I don't know" should receive 0.0 when the question expects
   an answer.

7. Do NOT evaluate factual correctness.
   Only evaluate whether the answer addresses the question.

8. Do NOT use outside knowledge.

9. Do NOT require the answer to repeat the wording of the question.

10. Do NOT penalize an answer for being short.

11. If the answer directly provides the requested entity,
    number, date, location, or concept, it should normally receive 1.0.

12. Return ONLY valid JSON.

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

    """
    Evaluate whether the generated answer
    directly addresses the question.

    Returns:
        score: float | None
        reason: str
        status: str
    """

    if not answer:
        return {
            "score": 0.0,
            "reason": "No answer was generated.",
            "status": "evaluation_error"
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
            data.get("score")
        )

        if score not in (0.0, 0.5, 1.0):
            return {
                "score": None,
                "reason": f"Invalid score returned: {score}",
                "status": "evaluation_error"
            }
        return {
            "score": score,
            "reason": data.get("reason", ""),
            "status": "success"
        }

    except (
        json.JSONDecodeError,
        TypeError,
        ValueError
    ):

        return {
            "score": 0.0,
            "reason": "Invalid evaluator response.",
            "status": "evaluation_error"
        }