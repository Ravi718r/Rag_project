import logging
import time

from fastapi import FastAPI, HTTPException, status

from app.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse
)

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.logging_config import setup_logging

from rag_setup import initialize_rag
from rag_pipeline import run_rag


# ============================================================
# Logging
# ============================================================

setup_logging()

logger = logging.getLogger(__name__)


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="RAG Question Answering API",

    description=(
        "A Retrieval-Augmented Generation API using "
        "hybrid retrieval, BM25, vector search, "
        "reranking and LLM generation."
    ),

    version="1.0.0"
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request,
    exc: RequestValidationError
):

    logger.warning(
        "Request validation failed | "
        "method=%s | path=%s | errors=%s",
        request.method,
        request.url.path,
        exc.errors()
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Request validation failed",
            "details": exc.errors()
        }
    )
# ============================================================
# Initialize RAG once
# ============================================================

logger.info("Initializing RAG system...")

try:

    (
        vectorstore,
        bm25,
        documents,
        generation_chain,
        embedding_model
    ) = initialize_rag()

    logger.info(
        "RAG system initialized successfully."
    )

except Exception:

    logger.exception(
        "Failed to initialize RAG system."
    )

    raise


logger.info("RAG system ready!")


# ============================================================
# Health Check
# ============================================================

@app.get(
    "/health",
    response_model=HealthResponse
)
def health_check():

    logger.info(
        "Health check requested."
    )

    return {
        "status": "healthy"
    }


# ============================================================
# Ask Question
# ============================================================

@app.post(
    "/ask",
    response_model=AskResponse
)
def ask_question(
    request: AskRequest
):

    start_time = time.perf_counter()

    logger.info(
        "RAG request received."
    )

    try:

        # ----------------------------------------------------
        # Run RAG pipeline
        # ----------------------------------------------------

        result = run_rag(
            vectorstore=vectorstore,
            bm25=bm25,
            documents=documents,
            generation_chain=generation_chain,
            query=request.question
        )

        # ----------------------------------------------------
        # Calculate latency
        # ----------------------------------------------------

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        logger.info(
            "RAG request completed successfully "
            "in %.2f ms.",
            latency_ms
        )

        # ----------------------------------------------------
        # Return API response
        # ----------------------------------------------------

        return AskResponse(
            question=result["question"],
            answer=result["answer"],
            latency_ms=round(
                latency_ms,
                2
            )
        )

    # ========================================================
    # Expected HTTP errors
    # ========================================================

    except HTTPException:

        raise

    # ========================================================
    # Unexpected errors
    # ========================================================

    except Exception:

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        logger.exception(
            "RAG request failed after %.2f ms.",
            latency_ms
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,

            detail=(
                "An internal error occurred while "
                "processing the question."
            )
        )