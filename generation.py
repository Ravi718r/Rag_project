from config import (
    PROMPT,
    llm,
    parser
)


# ============================
# Creating a Generation Chain
# ============================

def create_generation_chain():
    """
    Chain responsible only for answer generation.
    """

    generation_chain = (
        PROMPT
        | llm
        | parser
    )

    return generation_chain