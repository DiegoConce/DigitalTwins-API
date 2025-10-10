from typing import List, Optional
from pydantic import BaseModel

# Pydantic models: they define the shape of data for requests and responses.

"""
        return {
            "dataset_id": dataset_id,
            "author": getattr(dataset_info, 'author', None),
            "created_at": getattr(dataset_info, 'created_at', None),
            "readme_file": readme_text,
            "downloads": getattr(dataset_info, 'downloads', 0),
            "likes": getattr(dataset_info, 'likes', 0),
            "tags": getattr(dataset_info, 'tags', None),
            "language": getattr(card_data, 'language', None),
            "license": getattr(card_data, 'license', None),
            "multilinguality": getattr(card_data, 'multilinguality', None),
            "size_categories": getattr(card_data, 'size_categories', None),
            "task-categories": getattr(card_data, 'task_categories', None),
        }
        """


class DatasetItem(BaseModel):
    dataset_id: str
    author: Optional[str] = ""
    created_at: Optional[str] = ""
    readme_file: Optional[str] = ""
    downloads: int = 0
    likes: int = 0
    tags: List[str] = []
    language: List[str] = []
    license: Optional[str] = ""
    multilinguality: List[str] = []
    size_categories: List[str] = []
    task_categories: List[str] = []
    embeddings: List[float] = []


"""
        return {
            'model_id': model_id,
            'base_model': getattr(card_data, 'base_model', None),
            'author': getattr(info, 'author', None),
            'readme_file': readme,
            'license': getattr(card_data, 'license', None),
            'language': getattr(card_data, 'language', None),
            'downloads': getattr(info, 'downloads', 0),
            'likes': getattr(info, 'likes', 0),
            'tags': ', '.join(info.tags) if hasattr(info, 'tags') and info.tags else '',
            'pipeline_tag': getattr(info, 'pipeline_tag', None),
            'library_name': getattr(info, 'library_name', None),
            'created_at': getattr(info, 'created_at', None),
        }"""


class ModelItem(BaseModel):
    model_id: str
    base_model: Optional[str] = ""
    author: Optional[str] = ""
    readme_file: Optional[str] = ""
    license: Optional[str] = ""
    language: List[str] = []
    downloads: int = 0
    likes: int = 0
    tags: List[str] = []
    pipeline_tag: Optional[str] = ""
    library_name: Optional[str] = ""
    created_at: Optional[str] = ""
    embeddings: List[float] = []


class SearchRequest(BaseModel):
    """Request body for search queries."""
    description: str


class SearchResult(BaseModel):
    """A single search result item."""
    item: DatasetItem or ModelItem


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult] = []
    total_results: int
