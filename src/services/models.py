import ast
import numpy as np
from src.models.schemas import ModelItem
from typing import List, Optional
import pandas as pd
import random
from src.config import settings
from src.services.storage import StorageService
from src.services import rag


def load_data() -> pd.DataFrame:
    return pd.read_csv(settings.MODELS_CSV_PATH)


class ModelService:

    def __init__(self, rag_service: rag.RAGService, storage_service: StorageService):
        self.data = load_data()
        self.rag_service = rag_service
        self.storage_service = storage_service

    def get_sample(self, n: int = 10) -> List[ModelItem]:
        sample = self.storage_service.get_model_sample(10)
        return [self.dict_to_model_item(res) for res in sample]

    def search(self, query: str, top_k: int = settings.DEFAULT_TOP_K) -> List[ModelItem]:
        results = self.rag_service.search(self.data, query, mode="dataset", top_k=top_k)
        return [self._row_to_item(row) for _, row in results.iterrows()]

    def add_model(self, item: ModelItem, embedding: np.array) -> None:
        """Add a new model to the catalog."""
        model_id = item.author + "/" + item.model_id

        if self.storage_service.get_model_by_id(model_id):
            raise ValueError(f"Model with ID {model_id} already exists.")

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

        self.storage_service.store_model(model_id, metadata, embedding)

    @staticmethod
    def _row_to_item(row: pd.Series) -> ModelItem:
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

        return ModelItem(
            model_id=str(row.get('model_id', '')),
            base_model=str(row.get('base_model', '')),
            author=str(row.get('author', '')),
            readme_file=str(row.get('readme_file', '')),
            license=str(row.get('license', '')),
            language=parse_list_field(row.get('language')),
            downloads=int(row.get('downloads', 0)),
            likes=int(row.get('likes', 0)),
            tags=parse_list_field(row.get('tags')),
            pipeline_tag=str(row.get('pipeline_tag', '')),
            library_name=str(row.get('library_name', '')),
            created_at=str(row.get('created_at', '')),
            # No embeddings field here, not needed in the frontend
        )

    @staticmethod
    def dict_to_model_item(metadata: dict) -> ModelItem:
        """Convert a metadata dictionary into a ModelItem object."""

        def ensure_list(value):
            """Helper to ensure the value is a list."""
            if value is None:
                return []
            if isinstance(value, list):
                return value
            if isinstance(value, str):
                try:
                    parsed = ast.literal_eval(value)
                    return parsed if isinstance(parsed, list) else []
                except (ValueError, SyntaxError):
                    return []
            return []

        return ModelItem(
            model_id=str(metadata.get('model_id', '')),
            base_model=str(metadata.get('base_model', '')),
            author=str(metadata.get('author', '')),
            readme_file=str(metadata.get('readme_file', '')),
            license=str(metadata.get('license', '')),
            language=ensure_list(metadata.get('language')),
            downloads=int(metadata.get('downloads', 0)),
            likes=int(metadata.get('likes', 0)),
            tags=ensure_list(metadata.get('tags')),
            pipeline_tag=str(metadata.get('pipeline_tag', '')),
            library_name=str(metadata.get('library_name', '')),
            created_at=str(metadata.get('created_at', '')),
        )
