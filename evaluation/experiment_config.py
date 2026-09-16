from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    """
    Configuration for a RAG experiment.
    """

    name: str

    use_compression: bool = False
    use_reranking: bool = False
    use_mmr: bool = False

    top_k: int = 5