from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings and configuration."""

    # Application metadata
    APP_NAME: str = "DATHA Marketplace"
    APP_DESCRIPTION: str = "DATHA Marketplace API for searching and retrieving catalog items."

    # Search settings
    DEFAULT_TOP_K: int = 3 # Number of top results to return
    USE_LLM_FILTER: bool = False  # Whether to use LLM-based relevance filtering

    # Hugging Face API settings
    HUGGINGFACE_API_TOKEN: str = "YOUR_TOKEN"

    # Model settings
    LANGUAGE_MODEL_NAME: str = "meta-llama/Llama-3.2-3B"    #"meta-llama/Llama-3.1-8B-Instruct"
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

    #MinIo settings
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_SECURE: bool = False                      # Disable HTTPs for local testing
    MINIO_DATASETS_BUCKET: str = "datasets"
    MINIO_MODELS_BUCKET: str = "models"

    # Qdrant VectorDB settings
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_USE_HTTPS: bool = False
    QDRANT_DATASETS_COLLECTION: str = "datasets"
    QDRANT_MODELS_COLLECTION: str = "models"
    VECTOR_SIZE: int = 1024  # Jina embeddings v3 dimension

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
