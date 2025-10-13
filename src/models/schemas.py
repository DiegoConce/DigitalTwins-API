from typing import List, Optional
from pydantic import BaseModel


# Pydantic models: they define the shape of data for requests and responses.

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
    embeddings: List[str] = []


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
    embeddings: List[str] = []
