import ast
import numpy as np
from src.models.schemas import ModelItem
from typing import List, Optional
from src.services.storage import StorageService
from src.services.vectordb import QdrantService
from src.services import rag


class ModelService:

    def __init__(self, rag_service: rag.RAGService, storage_service: StorageService, qdrant_service: QdrantService):
        self.rag_service = rag_service
        self.storage_service = storage_service
        self.qdrant_service = qdrant_service

    def get_sample(self) -> List[ModelItem]:
        sample = self.storage_service.get_model_sample(10)
        return [self.dict_to_model_item(res) for res in sample]

    def search(self, query: str) -> List[ModelItem]:
        """Search models using RAG service."""
        results = self.rag_service.search(
            qdrant_service=self.qdrant_service,
            storage_service=self.storage_service,
            query=query,
            mode="model"
        )

        return [self.dict_to_model_item(metadata) for metadata in results]

    def add_model(self, item: ModelItem, embedding: np.array, weights: Optional[List[bytes]] = None, csv_sample: Optional[List[bytes]] = None) -> None:
        """Add a new model to the catalog."""
        model_id = item.author + "/" + item.model_id

        if self.storage_service.get_model_by_id(model_id):
            raise ValueError(f"Model with ID {model_id} already exists.")

        if self.qdrant_service.get_model_by_id(model_id):
            raise ValueError(f"Model with ID {model_id} already exists in vector DB.")

        metadata = {
            'model_id': item.model_id,
            'base_model': item.base_model,
            'author': item.author,
            'readme_file': item.readme_file,
            'license': item.license,
            'language': item.language,
            'downloads': item.downloads,
            'likes': item.likes,
            'tags': item.tags,
            'pipeline_tag': item.pipeline_tag,
            'library_name': item.library_name,
            'created_at': item.created_at,
        }

        # Store metadata in MinIO
        self.storage_service.store_model(model_id, metadata, weight_files=weights, csv_sample=csv_sample)

        # Add model to Qdrant vector DB
        self.qdrant_service.upsert_model(model_id, embedding)

    def delete_model(self, model_id: str) -> bool:
        """Delete model from both storage and vector DB."""
        storage_deleted = self.storage_service.delete_dataset_by_id(model_id)
        vector_deleted = self.qdrant_service.delete_model_by_id(model_id)
        return storage_deleted and vector_deleted

    @staticmethod
    def dict_to_model_item(metadata: dict) -> ModelItem:
        """Convert a metadata dictionary into a ModelItem object."""

        def ensure_list(value):
            """Helper to ensure the value is a list, handling NaN and None."""
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return []
            if isinstance(value, str):
                if not value.strip():
                    return []
                try:
                    parsed = ast.literal_eval(value)
                    return parsed if isinstance(parsed, list) else [str(value)]
                except Exception:
                    return [value]
            if isinstance(value, list):
                return [str(item) for item in value if not (isinstance(item, float) and np.isnan(item))]
            return [str(value)]

        def safe_str(value):
            """Convert to string, handling NaN."""
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return ""
            return str(value)

        def safe_int(value):
            """Convert to int, handling NaN and None."""
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return 0
            try:
                return int(value)
            except (ValueError, TypeError):
                return 0

        return ModelItem(
            model_id=safe_str(metadata.get('model_id', '')),
            base_model=safe_str(metadata.get('base_model', '')),
            author=safe_str(metadata.get('author', '')),
            readme_file=safe_str(metadata.get('readme_file', '')),
            license=safe_str(metadata.get('license', '')),
            language=ensure_list(metadata.get('language')),
            downloads=safe_int(metadata.get('downloads', 0)),
            likes=safe_int(metadata.get('likes', 0)),
            tags=ensure_list(metadata.get('tags')),
            pipeline_tag=safe_str(metadata.get('pipeline_tag', '')),
            library_name=safe_str(metadata.get('library_name', '')),
            created_at=safe_str(metadata.get('created_at', ''))
        )
