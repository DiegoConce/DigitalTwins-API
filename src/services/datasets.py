import ast
import numpy as np
from src.models.schemas import DatasetItem
from src.services import rag
from src.services.storage import StorageService
from src.services.vectordb import QdrantService
from typing import List, Dict, Optional


class DatasetService:
    """Service for managing dataset catalog."""

    def __init__(self, rag_service: rag.RAGService, storage_service: StorageService, qdrant_service: QdrantService):
        self.rag_service = rag_service
        self.storage_service = storage_service
        self.qdrant_service = qdrant_service

    def get_sample(self) -> List[DatasetItem]:
        sample = self.storage_service.get_dataset_sample(10)
        return [self.dict_to_dataset_item(res) for res in sample]

    def search(self, query: str) -> List[DatasetItem]:
        """Search datasets using RAG service."""
        results = self.rag_service.search(
            qdrant_service=self.qdrant_service,
            storage_service=self.storage_service,
            query=query,
            mode="dataset"
        )

        return [self.dict_to_dataset_item(metadata) for metadata in results]

    def add_dataset(self, item: DatasetItem, embedding: np.ndarray, data: Optional[List[bytes]] = None,
                    csv_sample: Optional[List[bytes]] = None) -> None:
        """Add a new dataset to the catalog."""
        dataset_id = item.author + "/" + item.dataset_id

        if self.storage_service.get_dataset_by_id(dataset_id):
            raise ValueError(f"Dataset with ID {dataset_id} already exists.")

        if self.qdrant_service.get_dataset_by_id(dataset_id):
            raise ValueError(f"Dataset with ID {dataset_id} already exists in vector DB.")

        metadata = {
            'dataset_id': item.dataset_id,
            'author': item.author,
            'created_at': item.created_at,
            'readme_file': item.readme_file,
            'downloads': item.downloads,
            'likes': item.likes,
            'tags': item.tags,
            'language': item.language,
            'license': item.license,
            'multilinguality': item.multilinguality,
            'size_categories': item.size_categories,
            'task_categories': item.task_categories,
            'has_csv': getattr(item, 'has_csv', False)
        }

        # Store metadata in MinIO
        self.storage_service.store_dataset(dataset_id, metadata, sample_files=data, csv_sample=csv_sample)

        # Store embedding in Qdrant
        self.qdrant_service.upsert_dataset(dataset_id, embedding)

    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete dataset from both storage and vector DB."""
        storage_deleted = self.storage_service.delete_dataset_by_id(dataset_id)
        vector_deleted = self.qdrant_service.delete_dataset_by_id(dataset_id)
        return storage_deleted and vector_deleted

    @staticmethod
    def dict_to_dataset_item(metadata: Dict) -> DatasetItem:
        """Convert a metadata dictionary into a DatasetItem object."""

        def ensure_list(value):
            """Helper to ensure the value is a list, handling NaN and None."""
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return []
            if isinstance(value, str):
                # Handle empty strings
                if not value.strip():
                    return []
                try:
                    parsed = ast.literal_eval(value)
                    return parsed if isinstance(parsed, list) else [str(value)]
                except Exception:
                    return [value]
            if isinstance(value, list):
                # Filter out NaN values from lists
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

        return DatasetItem(
            dataset_id=safe_str(metadata.get("dataset_id", "")),
            author=safe_str(metadata.get("author", "")),
            created_at=safe_str(metadata.get("created_at", "")),
            readme_file=safe_str(metadata.get("readme_file", "")),
            downloads=safe_int(metadata.get("downloads", 0)),
            likes=safe_int(metadata.get("likes", 0)),
            tags=ensure_list(metadata.get("tags")),
            language=ensure_list(metadata.get("language")),
            license=safe_str(metadata.get("license", "")),
            multilinguality=ensure_list(metadata.get("multilinguality")),
            size_categories=ensure_list(metadata.get("size_categories")),
            task_categories=ensure_list(metadata.get("task_categories")),
            has_csv=metadata.get("has_csv", False)
        )
