import pandas as pd
import numpy as np
from tqdm import tqdm
from src.dependencies import get_qdrant_service, get_storage_service, get_rag_service


def bulk_populate_datasets(csv_path: str):
    """Populate both MinIO and Qdrant with datasets from CSV."""
    qdrant_service = get_qdrant_service()
    storage_service = get_storage_service()
    rag_service = get_rag_service()

    df = pd.read_csv(csv_path)
    print(f"Loading {len(df)} datasets from CSV...")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Populating datasets"):
        try:
            dataset_id = row['dataset_id']
            metadata = row.to_dict()

            # Parse embedding from CSV
            embedding = np.fromstring(row['embeddings'][1:-1], sep=',').astype(np.float32)

            # Store in MinIO
            storage_service.store_dataset(dataset_id, metadata, embedding)

            # Store in Qdrant
            qdrant_service.upsert_dataset(dataset_id, embedding)

        except Exception as e:
            print(f"Error with dataset {row.get('dataset_id', 'unknown')}: {e}")

    print("Datasets population complete.")


def bulk_populate_models(csv_path: str):
    """Populate both MinIO and Qdrant with models from CSV."""
    qdrant_service = get_qdrant_service()
    storage_service = get_storage_service()
    rag_service = get_rag_service()

    df = pd.read_csv(csv_path)
    print(f"Loading {len(df)} models from CSV...")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Populating models"):
        try:
            model_id = row['model_id']
            metadata = row.to_dict()

            # Parse embedding from CSV
            embedding = np.fromstring(row['embeddings'][1:-1], sep=',').astype(np.float32)

            # Store in MinIO
            storage_service.store_model(model_id, metadata, embedding)

            # Store in Qdrant
            qdrant_service.upsert_model(model_id, embedding)

        except Exception as e:
            print(f"Error with model {row.get('model_id', 'unknown')}: {e}")

    print("Models population complete.")


if __name__ == "__main__":
    # Update paths as needed
    DATASETS_CSV = "/home/diego/Documenti/Workspace/DigitalTwins-API/data/datasets_hg_embeddings_sm.csv"
    MODELS_CSV = "/home/diego/Documenti/Workspace/DigitalTwins-API/data/models_hg_embeddings_sm.csv"

    bulk_populate_datasets(DATASETS_CSV)
    bulk_populate_models(MODELS_CSV)
