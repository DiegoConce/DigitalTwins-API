import ast
from typing import List, Literal
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from src.config import settings
from src.models.schemas import DatasetItem, ModelItem


def get_device(device_config: Literal["cpu", "cuda", "auto"] = "auto") -> str:
    if device_config == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if device_config == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU.")
        return "cpu"

    return device_config


def get_torch_dtype(device: str):
    return torch.float16 if device == "cuda" else torch.float32


class EmbeddingService:
    """Gestisce la generazione di embeddings."""

    def __init__(self):
        self.model = None
        self.device = get_device(settings.EMBEDDING_DEVICE)
        print("Using device for embeddings:", self.device)

    def _load_model(self):
        if self.model is None:
            self.model = AutoModel.from_pretrained(
                settings.EMBEDDING_MODEL_NAME,
                trust_remote_code=True
            ).to(self.device)

    def encode(self, text: str) -> np.ndarray:
        self._load_model()
        embedding = self.model.encode(text, task="text-matching")

        if self.device == "cuda":
            torch.cuda.empty_cache()

        return embedding

    def encode_dataset_item(self, item: DatasetItem) -> np.ndarray:
        full_text = "\n".join([
            item.dataset_id,
            item.author,
            item.created_at,
            item.readme_file,
            str(item.downloads),
            str(item.likes),
            str(item.tags),
            str(item.language),
            item.license,
            str(item.multilinguality),
            str(item.size_categories),
            str(item.task_categories)
        ])
        return self.encode(full_text)

    def encode_model_item(self, item: ModelItem) -> np.ndarray:
        full_text = "\n".join([
            item.model_id,
            item.base_model,
            item.author,
            item.readme_file,
            item.license,
            str(item.language),
            str(item.downloads),
            str(item.likes),
            str(item.tags),
            item.pipeline_tag,
            item.library_name,
            item.created_at
        ])
        return self.encode(full_text)


class SimilarityCalculator:
    """Calcola la similarità tra embeddings."""

    @staticmethod
    def compute_score(embeddings: np.ndarray, query_embedding: np.ndarray) -> float:
        return np.dot(embeddings, query_embedding)

    @staticmethod
    def filter_by_score(data: pd.DataFrame,
                        query_embedding: np.ndarray,
                        top_k: int = settings.DEFAULT_TOP_K) -> pd.DataFrame:
        data = data.copy()
        data['embeddings'] = data['embeddings'].apply(
            lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else x
        )
        data['score'] = data['embeddings'].apply(
            lambda x: SimilarityCalculator.compute_score(x, query_embedding)
        )
        return data.nlargest(top_k, 'score')


class ContentBuilder:
    """Costruisce il contenuto per il prompt LLM."""

    @staticmethod
    def build(data: pd.Series, mode: Literal["model", "dataset"]) -> str:
        if mode == "model":
            return "\n".join([
                str(data.get('model_id', '')),
                str(data.get('base_model', '')),
                str(data.get('author', '')),
                str(data.get('readme_file', '')),
                str(data.get('license', '')),
                str(data.get('language', '')),
                str(data.get('tags', '')),
                str(data.get('pipeline_tag', '')),
                str(data.get('library_name', ''))
            ])

        return "\n".join([
            str(data.get('dataset_id', '')),
            str(data.get('author', '')),
            str(data.get('readme_file', '')),
            str(data.get('tags', '')),
            str(data.get('language', '')),
            str(data.get('license', '')),
            str(data.get('multilinguality', '')),
            str(data.get('size_categories', '')),
            str(data.get('task_categories', ''))
        ])


class LLMFilter:
    """Filtra risultati usando un LLM."""

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.device = get_device(settings.LLM_DEVICE)
        self.dtype = get_torch_dtype(self.device)
        print("Using device for LLM:", self.device, "with dtype:", self.dtype)

    def _load_model(self):
        if self.model is None:
            login(settings.HUGGINGFACE_API_TOKEN)

            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.LANGUAGE_MODEL_NAME,
                trust_remote_code=True
            )

            device_map = 'auto' if self.device == "cuda" else None

            self.model = AutoModelForCausalLM.from_pretrained(
                settings.LANGUAGE_MODEL_NAME,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map=device_map
            )

            if device_map is None:
                self.model = self.model.to(self.device)

    def is_relevant(self, content: str) -> bool:
        self._load_model()

        messages = [
            {"role": "system", "content": settings.PROMPT_TEMPLATE},
            {"role": "user", "content": content}
        ]

        tokenized = self.tokenizer.apply_chat_template(messages, tokenize=False)
        tokenized = self.tokenizer(tokenized, return_tensors='pt').to(self.device)
        generated_ids = self.model.generate(**tokenized, max_new_tokens=100)
        output = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        return "yes" in output[-10:].lower()


class RAGService:
    """Servizio principale per la ricerca RAG."""

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.similarity_calculator = SimilarityCalculator()
        self.content_builder = ContentBuilder()
        self.llm_filter = LLMFilter()

    def search(
            self,
            data: pd.DataFrame,
            query: str,
            mode: Literal["model", "dataset"] = "model",
            top_k: int = settings.DEFAULT_TOP_K
    ) -> pd.DataFrame:
        # Step 1: Embedding similarity
        query_embedding = self.embedding_service.encode(query)
        filtered_data = self.similarity_calculator.filter_by_score(
            data,
            query_embedding,
            top_k
        )

        print("Filtered data after similarity:", filtered_data)

        # Step 2: LLM filtering
        """
        indices_to_keep = []
        for idx, row in filtered_data.iterrows():
            content = self.content_builder.build(row, mode)
            if self.llm_filter.is_relevant(content):
                indices_to_keep.append(idx) 
        """

        # return filtered_data.loc[indices_to_keep].reset_index(drop=True)

        return filtered_data.reset_index(drop=True)
