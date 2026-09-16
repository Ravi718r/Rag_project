import os

from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ============================================================
# Load Environment Variables
# ============================================================

load_dotenv()


# ============================================================
# RAG Configuration
# ============================================================

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "qwen2.5:3b"
)

TOP_K = int(
    os.getenv(
        "TOP_K",
        "3"
    )
)

PERSIST_DIRECTORY = os.getenv(
    "PERSIST_DIRECTORY",
    "db/chroma_db"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

SIMILARITY_THRESHOLD = float(
    os.getenv(
        "SIMILARITY_THRESHOLD",
        "0.45"
    )
)

RETRIEVAL_K = int(
    os.getenv(
        "RETRIEVAL_K",
        "10"
    )
)

FINAL_K = int(
    os.getenv(
        "FINAL_K",
        "3"
    )
)

RERANK_THRESHOLD = float(
    os.getenv(
        "RERANK_THRESHOLD",
        "0.20"
    )
)


# ============================================================
# LLM
# ============================================================

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0
)


# ============================================================
# Generation Prompt
# ============================================================

PROMPT = ChatPromptTemplate.from_template(
"""
You are a helpful AI assistant answering questions using
retrieved documents.

Answer ONLY using information explicitly supported by
the provided context.

Rules:

1. Do not use outside knowledge.
2. Do not invent facts.
3. Do not infer unsupported relationships.
4. Do not combine unrelated numbers or facts.
5. Preserve the exact relationship between entities,
   values, dates, and percentages.
6. If the context does not contain enough information
   to answer the question, say:
   "I don't know."

Context:
{context}

Question:
{question}

Answer:
"""
)


# ============================================================
# Compression Prompt
# ============================================================

COMPRESSION_PROMPT = """
You are a context extraction system.

Extract the complete factual statements from the document
that are relevant to answering the question.

IMPORTANT:

1. Use ONLY information explicitly present in the document.
2. Do NOT invent or infer facts.
3. Preserve relationships between people, organizations,
   numbers, dates, percentages, and other entities.
4. Never return isolated numbers or fragments.
5. Preserve the original meaning.
6. Do NOT answer the question.
7. Do NOT explain your reasoning.
8. If no relevant information exists, return exactly EMPTY.

Question:
{question}

Document:
{document}

Relevant factual statements:
"""


# ============================================================
# Evaluation Dataset Prompt
# ============================================================

DATASET_PROMPT = ChatPromptTemplate.from_template(
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


# ============================================================
# Output Parser
# ============================================================

parser = StrOutputParser()

# from langchain_ollama import ChatOllama
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser

# MODEL_NAME = "qwen2.5:3b"
# TOP_K = 3
# PERSIST_DIRECTORY = "db/chroma_db"
# EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# SIMILARITY_THRESHOLD = 0.45
# RETRIEVAL_K = 10
# FINAL_K = 3

# RERANK_THRESHOLD= 0.20

# llm = ChatOllama(
#     model=MODEL_NAME,
#     temperature=0
# )


# PROMPT = ChatPromptTemplate.from_template(
# """
# You are a helpful AI assistant answering questions using
# retrieved documents.

# Answer ONLY using information explicitly supported by
# the provided context.

# Rules:

# 1. Do not use outside knowledge.
# 2. Do not invent facts.
# 3. Do not infer unsupported relationships.
# 4. Do not combine unrelated numbers or facts.
# 5. Preserve the exact relationship between entities,
#    values, dates, and percentages.
# 6. If the context does not contain enough information
#    to answer the question, say:
#    "I don't know."

# Context:
# {context}

# Question:
# {question}

# Answer:
# """
# )


# COMPRESSION_PROMPT = """
# You are a context extraction system.

# Extract the complete factual statements from the document
# that are relevant to answering the question.

# IMPORTANT:

# 1. Use ONLY information explicitly present in the document.
# 2. Do NOT invent or infer facts.
# 3. Preserve relationships between people, organizations,
#    numbers, dates, percentages, and other entities.
# 4. Never return isolated numbers or fragments.
# 5. Preserve the original meaning.
# 6. Do NOT answer the question.
# 7. Do NOT explain your reasoning.
# 8. If no relevant information exists, return exactly EMPTY.

# Question:
# {question}

# Document:
# {document}

# Relevant factual statements:
# """

# DATASET_PROMPT = ChatPromptTemplate.from_template(
#     """
# You are creating an evaluation dataset for a RAG system.

# Read the document below and generate ONE question
# that can be answered ONLY using the information
# contained in the document.

# Rules:

# 1. The question must be answerable from the document.
# 2. Do not use outside knowledge.
# 3. The answer must be explicitly supported by the document.
# 4. Make the question specific and useful for RAG evaluation.
# 5. Do not create a yes/no question.
# 6. Return ONLY valid JSON.
# 7. Do not use markdown.

# Required JSON format:

# {{
#     "question": "...",
#     "ground_truth": "..."
# }}

# Document:
# {document}
# """
# )

# parser = StrOutputParser()