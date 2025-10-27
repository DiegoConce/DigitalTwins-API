from typing import List, Dict, Optional, Literal
import numpy as np
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, MatchValue
)
from src.config import settings


def make_qdrant_id(dataset_id: str) -> int:
    uid = uuid.uuid5(uuid.NAMESPACE_URL, dataset_id)
    return uid.int % (2 ** 63)


class QdrantService:
    """Service for managing vector storage with Qdrant."""

    def __init__(self):
        self.client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            https=settings.QDRANT_USE_HTTPS
        )
        self._ensure_collections_exist()

    def _ensure_collections_exist(self):
        """Create collections if they don't exist."""

        collections = [
            settings.QDRANT_MODELS_COLLECTION,
            settings.QDRANT_DATASETS_COLLECTION
        ]

        for collection_name in collections:
            if not self.client.collection_exists(collection_name):
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=settings.VECTOR_SIZE,
                        distance=Distance.DOT
                    )
                )

    def upsert_dataset(self, dataset_id: str, embedding: np.ndarray):
        """Insert or update a dataset with its embedding. The dataset_id is the constructed as "author/dataset_id """
        point = PointStruct(
            id=make_qdrant_id(dataset_id),  # Convert to positive int
            vector=embedding.tolist(),
            payload={"dataset_id": dataset_id}
        )

        self.client.upsert(
            collection_name=settings.QDRANT_DATASETS_COLLECTION,
            points=[point]
        )

    def get_dataset_by_id(self, dataset_id: str) -> Optional[Dict]:
        """Retrieve a dataset by its ID."""
        result = self.client.scroll(
            collection_name=settings.QDRANT_DATASETS_COLLECTION,
            scroll_filter=Filter(
                must=[FieldCondition(key="dataset_id", match=MatchValue(value=dataset_id))]

            ),
            limit=1
        )
        if result[0]:
            return result[0][0].payload
        return None

    def delete_dataset_by_id(self, dataset_id: str) -> bool:
        """Delete a dataset from the vector database."""
        try:
            self.client.delete(
                collection_name=settings.QDRANT_DATASETS_COLLECTION,
                points_selector=[make_qdrant_id(dataset_id)]
            )
            return True
        except Exception:
            return False

    def search_datasets(self, query_vector: np.ndarray, top_k) -> List[Dict]:
        """Search for similar datasets by vector similarity."""
        search_result = self.client.query_points(
            collection_name=settings.QDRANT_DATASETS_COLLECTION,
            query=query_vector.tolist(),
            limit=top_k,
            with_payload=True,  # include dataset_id in the results
            with_vectors=False  # we don't need full vectors back
        )
        print("Qdrant search_datasets results:", search_result)
        return [
            {
                "dataset_id": hit.payload.get("dataset_id"),
                "score": hit.score
            }
            for hit in search_result.points
        ]

    def upsert_model(self, model_id: str, embedding: np.ndarray):
        """Insert or update a model with its embedding. The model_id is the constructed as "author/model_id """
        point = PointStruct(
            id=make_qdrant_id(model_id),  # Convert to positive int
            vector=embedding.tolist(),
            payload={"model_id": model_id}
        )

        self.client.upsert(
            collection_name=settings.QDRANT_MODELS_COLLECTION,
            points=[point]
        )

    def get_model_by_id(self, model_id: str) -> Optional[Dict]:
        """Retrieve a model by its ID."""
        result = self.client.scroll(
            collection_name=settings.QDRANT_MODELS_COLLECTION,
            scroll_filter=Filter(
                must=[FieldCondition(key="model_id", match=MatchValue(value=model_id))]

            ),
            limit=1
        )
        if result[0]:
            return result[0][0].payload
        return None

    def delete_model_by_id(self, model_id: str) -> bool:
        """Delete a model from the vector database."""
        try:
            self.client.delete(
                collection_name=settings.QDRANT_MODELS_COLLECTION,
                points_selector=[make_qdrant_id(model_id)]
            )
            return True
        except Exception:
            return False

    def search_models(self, query_vector: np.ndarray, top_k) -> List[Dict]:
        """Search for similar models by vector similarity."""
        search_result = self.client.query_points(
            collection_name=settings.QDRANT_MODELS_COLLECTION,
            query=query_vector.tolist(),
            limit=top_k,
            with_payload=True,  # include model_id in the results
            with_vectors=False  # we don't need full vectors back
        )
        print("Qdrant search_models results:", search_result)
        return [
            {
                "model_id": hit.payload.get("model_id"),
                "score": hit.score
            }
            for hit in search_result.points
        ]
