import io, json
import random
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict
from minio import Minio
from minio.error import S3Error
from src.config import settings


class StorageService:
    def __init__(self):
        self.client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE
        )
        self.datasets_bucket = settings.MINIO_DATASETS_BUCKET
        self.models_bucket = settings.MINIO_MODELS_BUCKET
        self._ensure_buckets_exist()

    def _ensure_buckets_exist(self):
        """ Create buckets if they do not exist. """
        for bucket in [self.datasets_bucket, self.models_bucket]:
            if not self.client.bucket_exists(bucket):
                self.client.make_bucket(bucket)

    def _put_json(self, bucket: str, object_name: str, data: dict) -> None:
        """Helper to store JSON data"""
        json_data = json.dumps(data, indent=2).encode('utf-8')
        self.client.put_object(
            bucket,
            object_name,
            io.BytesIO(json_data),
            len(json_data),
            content_type="application/json"
        )

    def _get_json(self, bucket: str, object_name: str) -> Optional[dict]:
        """Helper to retrieve JSON data"""
        try:
            response = self.client.get_object(bucket, object_name)
            return json.loads(response.read())
        except S3Error:
            return None

    def _put_numpy_array(self, bucket: str, object_name: str, array: np.ndarray) -> None:
        """Helper to store numpy array"""
        with io.BytesIO() as buffer:
            np.save(buffer, array)
            buffer.seek(0)
            self.client.put_object(
                bucket,
                object_name,
                buffer,
                buffer.getbuffer().nbytes,
                content_type="application/octet-stream"
            )

    def _get_numpy_array(self, bucket: str, object_name: str) -> Optional[np.ndarray]:
        """Helper to retrieve numpy array"""
        try:
            response = self.client.get_object(bucket, object_name)
            return np.load(io.BytesIO(response.read()))
        except S3Error:
            return None

    # Dataset Operations
    def store_dataset(self, dataset_id: str, metadata: dict, embedding: np.ndarray,
                      sample_files: Optional[List[bytes]] = None) -> None:

        """Store complete dataset information"""
        metadata_with_timestamp = {
            **metadata,
            "stored_at": datetime.utcnow().isoformat(),
            "storage_version": "1.0"
        }

        self._put_json(
            self.datasets_bucket,
            f"{dataset_id}/metadata.json",
            metadata_with_timestamp
        )

        self._put_numpy_array(
            self.datasets_bucket,
            f"{dataset_id}/embedding.npy",
            np.array(embedding)
        )

        # Optional sample files in data/
        if sample_files:
            for idx, file_bytes in enumerate(sample_files):
                self.client.put_object(
                    self.datasets_bucket,
                    f"{dataset_id}/data/sample_{idx}.txt",
                    io.BytesIO(file_bytes),
                    len(file_bytes),
                    content_type="text/plain"
                )

    def list_datasets(self) -> list:
        """
        List all dataset IDs stored in MinIO in the format author/dataset_id.
        Assumes each dataset has a metadata.json file at: <author>/<dataset_id>/metadata.json
        """
        dataset_paths = set()

        # List all objects recursively
        for obj in self.client.list_objects(self.datasets_bucket, recursive=True):
            # Only consider metadata.json files
            if obj.object_name.endswith("metadata.json"):
                # Extract author/dataset_id from path
                parts = obj.object_name.split("/")
                if len(parts) >= 2:
                    dataset_path = f"{parts[0]}/{parts[1]}"
                    dataset_paths.add(dataset_path)

        return list(dataset_paths)

    def get_dataset_sample(self, n: int) -> List[Dict]:
        """
        Retrieve a random sample of `n` dataset items from storage.
        Returns a list of metadata dicts.
        """
        dataset_ids = self.list_datasets()
        if not dataset_ids:
            return []

        sample_ids = random.sample(dataset_ids, min(n, len(dataset_ids)))

        sample_items = []
        for dataset_id in sample_ids:
            metadata = self._get_json(self.datasets_bucket, f"{dataset_id}/metadata.json")
            if metadata:
                sample_items.append(metadata)
        return sample_items

    def get_dataset_by_id(self, dataset_id: str) -> Optional[Dict]:
        """
        Retrieve a dataset's metadata by ID if it exists.
        Returns None if not found.
        """
        metadata = self._get_json(
            self.datasets_bucket,
            f"{dataset_id}/metadata.json"
        )
        return metadata

    def store_model(self, model_id: str, metadata: dict, embedding: np.ndarray,
                    weight_files: Optional[List[bytes]] = None) -> None:

        """Store complete model information"""
        metadata_with_timestamp = {
            **metadata,
            "stored_at": datetime.utcnow().isoformat(),
            "storage_version": "1.0"
        }

        self._put_json(
            self.models_bucket,
            f"{model_id}/metadata.json",
            metadata_with_timestamp
        )

        self._put_numpy_array(
            self.models_bucket,
            f"{model_id}/embedding.npy",
            np.array(embedding)
        )

        # Optional weights in weights/
        if weight_files:
            for idx, file_bytes in enumerate(weight_files):
                self.client.put_object(
                    self.models_bucket,
                    f"{model_id}/weights/weight_{idx}.txt",
                    io.BytesIO(file_bytes),
                    len(file_bytes),
                    content_type="text/plain"
                )

    def list_models(self) -> list:
        """
        List all model IDs stored in MinIO in the format author/dataset_id.
        """
        model_paths = set()

        # List all objects recursively
        for obj in self.client.list_objects(self.models_bucket, recursive=True):
            # Only consider metadata.json files
            if obj.object_name.endswith("metadata.json"):
                # Extract author/model_id from path
                parts = obj.object_name.split("/")
                if len(parts) >= 2:
                    model_path = f"{parts[0]}/{parts[1]}"
                    model_paths.add(model_path)

        return list(model_paths)

    def get_model_sample(self, n: int) -> List[Dict]:
        """
        Retrieve a random sample of `n` model items from storage.
        Returns a list of metadata dicts.
        """
        model_ids = self.list_models()
        if not model_ids:
            return []

        sample_ids = random.sample(model_ids, min(n, len(model_ids)))

        sample_items = []
        for model_id in sample_ids:
            metadata = self._get_json(self.models_bucket, f"{model_id}/metadata.json")
            if metadata:
                sample_items.append(metadata)
        return sample_items

    def get_model_by_id(self, model_id: str) -> Optional[Dict]:
        """
        Retrieve a model's metadata by ID if it exists.
        Returns None if not found.
        """
        metadata = self._get_json(
            self.models_bucket,
            f"{model_id}/metadata.json"
        )
        return metadata


### Testing the class
if __name__ == "__main__":
    """
            import pandas as pd
            from tqdm import tqdm
            import numpy as np
        
        
            def bulk_upload_datasets_from_csv():
                
    df = pd.read_csv(settings.DATASETS_CSV_PATH)
    
    print(f"Uploading {len(df)} datasets to MinIO...")
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Uploading datasets"):
        try:
            dataset_id = row["dataset_id"]
    
            # Convert row to dict for metadata
            metadata = row.to_dict()
    
            # Generate a mock embedding
            embedding = np.random.rand(768).tolist()
    
            storage_service.store_dataset(
                dataset_id=dataset_id,
                metadata=metadata,
                embedding=embedding,
                sample_files=None  # optional
            )
        except Exception as e:
            print(f"❌ Error uploading dataset {row.get('dataset_id', 'unknown')}: {e}")
    
    print("✅ Bulk upload of datasets complete.")
    
    
    def bulk_upload_models_from_csv():
    
    # Load CSV
    df = pd.read_csv(settings.MODELS_CSV_PATH)
    # df = df.fillna("")
    print(f"Uploading {len(df)} models to MinIO...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Uploading models"):
        try:
            model_id = row["model_id"]
    
            # Convert row to dict for metadata
            metadata = row.to_dict()
    
            # Generate a mock embedding
            embedding = np.random.rand(768).tolist()
    
            storage_service.store_model(
                model_id=model_id,
                metadata=metadata,
                embedding=embedding,
                weight_files=None  # optional
            )
        except Exception as e:
            print(f"❌ Error uploading model {row.get('model_id', 'unknown')}: {e}")
    
    print("✅ Bulk upload of models complete.")
    
    
    # Ricorda per ora minio è containerizzato, lo storage è montato su docker
    storage_service = StorageService()
    # rag_service = RAGService()
    # dataset_service = DatasetService(rag_service)
    print("StorageService initialized and buckets ensured.")
    
    # Esegui l'upload di massa dei dataset dal CSV
    # bulk_upload_datasets_from_csv()
    # bulk_upload_models_from_csv()
    # print("Available datasets in storage:", storage_service.list_datasets())
    
    # print("Sample datasets:", storage_service.get_dataset_sample(n=5))
    
    # print(storage_service._get_json(storage_service.datasets_bucket, "kashif/App_Flow/metadata.json"))
    
    # print("Getting dataset by ID:", storage_service.get_dataset_by_id("kashif/App_Flow"))
    
    # print("Available models in storage:", storage_service.list_models())
    """
