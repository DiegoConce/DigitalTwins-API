from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings and configuration."""

    # Application metadata
    APP_NAME: str = "Catalog Search API"
    APP_DESCRIPTION: str = "Digital Twins API for searching and retrieving catalog items."

    # Search settings
    DEFAULT_TOP_K: int = 3 # Number of top results to return

    # Hugging Face API settings
    HUGGINGFACE_API_TOKEN: str = "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"

    # Model settings
    LANGUAGE_MODEL_NAME: str = "meta-llama/Llama-3.2-3B" #"meta-llama/Llama-3.1-8B-Instruct"
    EMBEDDING_MODEL_NAME: str = "jinaai/jina-embeddings-v3"

    # Device settings
    DEVICE: Literal["cpu", "cuda", "auto"] = "cpu"
    EMBEDDING_DEVICE: Literal["cpu", "cuda", "auto"] = "cpu"
    LLM_DEVICE: Literal["cpu", "cuda", "auto"] = "cpu"


    # Prompt settings
    PROMPT_TEMPLATE: str = """
    As an intelligent language model, your role is to accurately determine whether the provided data is relevant to the user's query.
    Answer ONLY with 'Yes' or 'No'
    """

    # Data paths
    DATASETS_CSV_PATH: str = "data/datasets_hg_embeddings_sm.csv"
    MODELS_CSV_PATH: str = "data/models_hg_embeddings_sm.csv"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
