import ast

from src.models.schemas import ModelItem
from typing import List, Optional
import pandas as pd
import random
from src.config import settings
from src.services import rag


def load_data() -> pd.DataFrame:
    return pd.read_csv(settings.MODELS_CSV_PATH)


class ModelService:

    def __init__(self, rag_service: rag.RAGService):
        self.data = load_data()
        self.rag_service = rag_service

    def get_sample(self, n: int = 10) -> List[ModelItem]:
        sample_df = self.data.sample(n=min(n, len(self.data)))
        return [self._row_to_item(row) for _, row in sample_df.iterrows()]

    def search(self, query: str, top_k: int = settings.DEFAULT_TOP_K) -> List[ModelItem]:
        results = self.rag_service.search(self.data, query, mode="dataset", top_k=top_k)
        return [self._row_to_item(row) for _, row in results.iterrows()]


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
        )
