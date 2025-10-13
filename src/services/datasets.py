import ast
import random

import numpy as np

from src.config import settings
from src.models.schemas import DatasetItem
from src.services import rag
from typing import List, Optional
import pandas as pd


def load_data() -> pd.DataFrame:
    return pd.read_csv(settings.DATASETS_CSV_PATH)


class DatasetService:
    """Service for managing dataset catalog."""

    def __init__(self, rag_service: rag.RAGService):
        self.data = load_data()
        self.rag_service = rag_service

    def get_sample(self, n: int = 10) -> List[DatasetItem]:
        sample_df = self.data.sample(n=min(n, len(self.data)))
        return [self._row_to_item(row) for _, row in sample_df.iterrows()]

    def search(self, query: str, top_k: int = settings.DEFAULT_TOP_K) -> List[DatasetItem]:
        results = self.rag_service.search(self.data, query, mode="dataset", top_k=top_k)
        return [self._row_to_item(row) for _, row in results.iterrows()]

    def add_dataset(self, item: DatasetItem, embedding: np.ndarray = None) -> None:
        """Add a new dataset to the catalog."""
        if item.dataset_id in self.data['dataset_id'].values:
            raise ValueError(f"Dataset with ID {item.dataset_id} already exists.")

        new_row = {
            'dataset_id': item.dataset_id,
            'author': item.author,
            'created_at': item.created_at,
            'readme_file': item.readme_file,
            'downloads': item.downloads,
            'likes': item.likes,
            'tags': str(item.tags),
            'language': str(item.language),
            'license': item.license,
            'multilinguality': str(item.multilinguality),
            'size_categories': str(item.size_categories),
            'task_categories': str(item.task_categories),
            'embeddings': str(embedding.tolist()) if embedding is not None else '[]'
        }

        print("Adding new dataset:", new_row)
        self.data = pd.concat([self.data, pd.DataFrame([new_row])], ignore_index=True)
        self.data.to_csv(settings.DATASETS_CSV_PATH, index=False)

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
