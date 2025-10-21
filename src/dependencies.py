from functools import lru_cache
from src.services.rag import RAGService
from src.services.storage import StorageService
from src.services.vectordb import QdrantService
from src.services.datasets import DatasetService
from src.services.models import ModelService


@lru_cache()
def get_rag_service() -> RAGService:
    return RAGService()


@lru_cache()
def get_storage_service() -> StorageService:
    return StorageService()


@lru_cache()
def get_qdrant_service() -> QdrantService:
    return QdrantService()


def get_dataset_service(
    rag_service: RAGService = None,
    storage_service: StorageService = None,
    qdrant_service: QdrantService = None
) -> DatasetService:
    if rag_service is None:
        rag_service = get_rag_service()
    if storage_service is None:
        storage_service = get_storage_service()
    if qdrant_service is None:
        qdrant_service = get_qdrant_service()
    return DatasetService(rag_service, storage_service, qdrant_service)


def get_model_service(
    rag_service: RAGService = None,
    storage_service: StorageService = None,
    qdrant_service: QdrantService = None
) -> ModelService:
    if rag_service is None:
        rag_service = get_rag_service()
    if storage_service is None:
        storage_service = get_storage_service()
    if qdrant_service is None:
        qdrant_service = get_qdrant_service()
    return ModelService(rag_service, storage_service, qdrant_service)
