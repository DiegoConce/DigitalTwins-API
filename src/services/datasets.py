import ast
import random
from src.models.schemas import DatasetItem
from typing import List, Optional
import pandas as pd


class DatasetService:
    """Service for managing dataset catalog."""

    def __init__(self):
        self.datasets = self._load_datasets()


    def _load_datasets(self) -> List[DatasetItem]:
        """Load datasets from CSV file.
        !!potrebbe nn servire
        """
        df = pd.read_csv('data/datasets_hg_embeddings_sm.csv')
        df = df.where(pd.notna(df), None)

        # Helper function to safely parse string lists
        def safe_parse_list(value):
            if value is None or value == '':
                return []
            try:
                return ast.literal_eval(value) if isinstance(value, str) else []
            except:
                return []

        datasets = []
        for _, row in df.iterrows():
            dataset = DatasetItem(
                dataset_id=str(row['dataset_id']) if row['dataset_id'] is not None else '',
                author=str(row['author']) if row['author'] is not None else '',
                created_at=str(row['created_at']) if row['created_at'] is not None else '',
                readme_file=str(row['readme_file']) if row['readme_file'] is not None else '',
                downloads=int(row['downloads']) if pd.notna(row['downloads']) else 0,
                likes=int(row['likes']) if pd.notna(row['likes']) else 0,
                tags=safe_parse_list(row.get('tags')),
                language=safe_parse_list(row.get('language')),
                license=str(row['license']) if row['license'] is not None else '',
                multilinguality=safe_parse_list(row.get('multilinguality')),
                size_categories=safe_parse_list(row.get('size_categories')),
                task_categories=safe_parse_list(row.get('task_categories')),
                embeddings=safe_parse_list(row.get('embeddings'))
            )
            datasets.append(dataset)

        return datasets

    def get_sample(self) -> List[DatasetItem]:
        """Return a random sample of 10 datasets."""
        return random.sample(self.datasets, min(10, len(self.datasets)))

    def get_by_id(self, dataset_id: str) -> Optional[DatasetItem]:
        """Get specific dataset by ID."""
        for dataset in self.datasets:
            if dataset.dataset_id == dataset_id:
                return dataset
        return None

    def search(self, description: str, top_k: int = 10) -> str:
        """Search datasets based on description."""
        # Implement your search logic here

        return "Search functionality not implemented yet. " + description + " top_k: " + str(top_k)
