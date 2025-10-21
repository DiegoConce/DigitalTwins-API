import ast
import random
import numpy as np
from src.config import settings
from src.models.schemas import DatasetItem
from src.services import rag
from src.services.storage import StorageService
from typing import List, Optional, Dict
import pandas as pd


def load_data() -> pd.DataFrame:
    return pd.read_csv(settings.DATASETS_CSV_PATH)


class DatasetService:
    """Service for managing dataset catalog."""

    def __init__(self, rag_service: rag.RAGService, storage_service: StorageService):
        self.data = load_data() # nn servira piu
        self.rag_service = rag_service
        self.storage_service = storage_service

    def get_sample(self) -> List[DatasetItem]:
        sample = self.storage_service.get_dataset_sample(10)
        return [self.dict_to_dataset_item(res) for res in sample]

    def search(self, query: str, top_k: int = settings.DEFAULT_TOP_K) -> List[DatasetItem]:
        results = self.rag_service.search(self.data, query, mode="dataset", top_k=top_k)
        return [self._row_to_item(row) for _, row in results.iterrows()]

    def add_dataset(self, item: DatasetItem, embedding: np.ndarray) -> None:
        """Add a new dataset to the catalog."""
        dataset_id = item.author + "/" + item.dataset_id

        if self.storage_service.get_dataset_by_id(dataset_id):
            raise ValueError(f"Dataset with ID {dataset_id} already exists.")

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
        }

        self.storage_service.store_dataset(dataset_id, metadata, embedding)

    @staticmethod
    def _row_to_item(row: pd.Series) -> DatasetItem:
        def parse_list_field(value) -> List[str]:
            """Convert CSV string representation to list, handling NaN values."""
            if pd.isna(value):
                return []
            if isinstance(value, list):
                return value
            if isinstance(value, str):
                try:
                    # Use ast.literal_eval to safely parse string representation of list
                    parsed = ast.literal_eval(value)
                    return parsed if isinstance(parsed, list) else []
                except (ValueError, SyntaxError):
                    return []
            return []

        return DatasetItem(
            dataset_id=str(row.get('dataset_id', '')),
            author=str(row.get('author', '')),
            created_at=str(row.get('created_at', '')),
            readme_file=str(row.get('readme_file', '')),
            downloads=int(row.get('downloads', 0)) if pd.notna(row.get('downloads')) else 0,
            likes=int(row.get('likes', 0)) if pd.notna(row.get('likes')) else 0,
            tags=parse_list_field(row.get('tags')),
            language=parse_list_field(row.get('language')),
            license=str(row.get('license', '')) if pd.notna(row.get('license')) else '',
            multilinguality=parse_list_field(row.get('multilinguality')),
            size_categories=parse_list_field(row.get('size_categories')),
            task_categories=parse_list_field(row.get('task_categories')),
            # No embeddings field here, not needed in the frontend
        )

    @staticmethod
    def dict_to_dataset_item(metadata: Dict) -> DatasetItem:
        """Convert a metadata dictionary into a DatasetItem object."""

        def ensure_list(value):
            """Helper to ensure the value is a list."""
            if value is None:
                return []
            if isinstance(value, str):
                try:
                    parsed = ast.literal_eval(value)
                    return parsed if isinstance(parsed, list) else [value]
                except Exception:
                    return [value]
            if isinstance(value, list):
                return value
            return [value]

        return DatasetItem(
            dataset_id=str(metadata.get("dataset_id", "")),
            author=str(metadata.get("author", "")),
            created_at=str(metadata.get("created_at", "")),
            readme_file=str(metadata.get("readme_file", "")),
            downloads=int(metadata.get("downloads", 0)),
            likes=int(metadata.get("likes", 0)),
            tags=ensure_list(metadata.get("tags")),
            language=ensure_list(metadata.get("language")),
            license=str(metadata.get("license", "")),
            multilinguality=ensure_list(metadata.get("multilinguality")),
            size_categories=ensure_list(metadata.get("size_categories")),
            task_categories=ensure_list(metadata.get("task_categories"))
        )
