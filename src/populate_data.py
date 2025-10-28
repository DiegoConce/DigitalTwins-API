import pandas as pd
import numpy as np
from tqdm import tqdm
from src.dependencies import get_qdrant_service, get_storage_service, get_dataset_service, get_model_service
from src.models.schemas import DatasetItem, ModelItem
from src.config import settings


def bulk_populate_datasets(csv_path: str):
    """Populate both MinIO and Qdrant with datasets from CSV."""
    qdrant_service = get_qdrant_service()
    storage_service = get_storage_service()

    df = pd.read_csv(csv_path)
    df = df.sample(20)
    print(f"Loading {len(df)} datasets from CSV...")

    placeholder_file = (b"This is a placeholder file. "
                        b"It is used to represent a dataset file or a weight file for a model.")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Populating datasets"):
        try:
            dataset_id = row['dataset_id']
            metadata = row.to_dict()

            # Parse embedding from CSV
            embedding = np.fromstring(row['embeddings'][1:-1], sep=',').astype(np.float32)

            # Store in MinIO
            storage_service.store_dataset(dataset_id, metadata, sample_files=[placeholder_file])

            # Store in Qdrant
            qdrant_service.upsert_dataset(dataset_id, embedding)

        except Exception as e:
            print(f"Error with dataset {row.get('dataset_id', 'unknown')}: {e}")

    print("Datasets population complete.")


def bulk_populate_models(csv_path: str):
    """Populate both MinIO and Qdrant with models from CSV."""
    qdrant_service = get_qdrant_service()
    storage_service = get_storage_service()

    df = pd.read_csv(csv_path)
    df = df.sample(20)
    print(f"Loading {len(df)} models from CSV...")

    placeholder_file = (b"This is a placeholder file. "
                        b"It is used to represent a dataset file or a weight file for a model.")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Populating models"):
        try:
            model_id = row['model_id']
            metadata = row.to_dict()

            # Parse embedding from CSV
            embedding = np.fromstring(row['embeddings'][1:-1], sep=',').astype(np.float32)

            # Store in MinIO
            storage_service.store_model(model_id, metadata, weight_files=[placeholder_file])

            # Store in Qdrant
            qdrant_service.upsert_model(model_id, embedding)

        except Exception as e:
            print(f"Error with model {row.get('model_id', 'unknown')}: {e}")

    print("Models population complete.")


def populate_mock_data():
    """Add 6 personalized datasets and 2 personalized models as mock data."""
    dataset_service = get_dataset_service()
    model_service = get_model_service()

    placeholder_file = (b"This is a placeholder file. "
                        b"It is used to represent a dataset file or a weight file for a model.")

    # Datasets
    mock_datasets = [
        DatasetItem(
            dataset_id="SISMA-3DPrinter-Logs",
            author="BIREX-CompetenceCenter",
            created_at="2024-01-15",
            readme_file="System logs from SISMA metal 3D printers with process parameters and sensor readings for anomaly detection and predictive maintenance.",
            downloads=500,
            likes=25,
            tags=["industrial", "additive-manufacturing", "logs", "predictive-maintenance", "machine-learning",
                  "anomaly-detection", "time-series"],
            language=["none"],
            license="cc-by-nc-4.0",
            multilinguality=["nonlinguistic"],
            size_categories=["100K<n<1M"],
            task_categories=["time-series-analysis"]
        ),
        DatasetItem(
            dataset_id="SISMA-3DPrinter-Images",
            author="BIREX-CompetenceCenter",
            created_at="2024-01-15",
            readme_file="Layer-by-layer optical images from SISMA 3D printers for defect detection and automated quality control.",
            downloads=750,
            likes=40,
            tags=["industrial", "additive-manufacturing", "image-processing", "computer-vision", "quality-inspection"],
            language=["none"],
            license="cc-by-nc-4.0",
            multilinguality=["nonlinguistic"],
            size_categories=["10K<n<100K"],
            task_categories=["image-classification", "object-detection"]
        ),
        DatasetItem(
            dataset_id="SISMA-3DPrinter-EnergeticConsumption",
            author="BIREX-CompetenceCenter",
            created_at="2024-01-15",
            readme_file="Energy consumption profiles of SISMA 3D printers for efficiency and sustainability analysis.",
            downloads=400,
            likes=18,
            tags=["energy-monitoring", "time-series", "industrial", "sustainability", "additive-manufacturing"],
            language=["none"],
            license="cc-by-nc-4.0",
            multilinguality=["nonlinguistic"],
            size_categories=["10K<n<100K"],
            task_categories=["time-series-analysis"]
        ),
        DatasetItem(
            dataset_id="SLM-NIKON-3DPrinter-Logs",
            author="BIREX-CompetenceCenter",
            created_at="2024-01-15",
            readme_file="Operational logs from NIKON SLM metal 3D printers including real-time process parameters and diagnostic data for fault detection.",
            downloads=520,
            likes=27,
            tags=["industrial", "logs", "additive-manufacturing", "nikon", "predictive-maintenance",
                  "anomaly-detection", "time-series"],
            language=["none"],
            license="cc-by-nc-4.0",
            multilinguality=["nonlinguistic"],
            size_categories=["100K<n<1M"],
            task_categories=["time-series-analysis"]
        ),
        DatasetItem(
            dataset_id="SLM-NIKON-3DPrinter-Images",
            author="BIREX-CompetenceCenter",
            created_at="2024-01-15",
            readme_file="Optical images from NIKON SLM metal 3D printers annotated for surface defects and irregularities in additive manufacturing processes.",
            downloads=850,
            likes=42,
            tags=["industrial", "computer-vision", "image-analysis", "additive-manufacturing", "defect-detection"],
            language=["none"],
            license="cc-by-nc-4.0",
            multilinguality=["nonlinguistic"],
            size_categories=["10K<n<100K"],
            task_categories=["image-classification", "object-detection"]
        ),
        DatasetItem(
            dataset_id="SLM-NIKON-3DPrinter-EnergeticConsumption",
            author="BIREX-CompetenceCenter",
            created_at="2024-01-15",
            readme_file="Energy usage time series collected from NIKON SLM metal 3D printers for energy efficiency and sustainability modeling.",
            downloads=420,
            likes=19,
            tags=["energy-monitoring", "industrial", "additive-manufacturing", "time-series", "sustainability"],
            language=["none"],
            license="cc-by-nc-4.0",
            multilinguality=["nonlinguistic"],
            size_categories=["10K<n<100K"],
            task_categories=["time-series-analysis"]
        )
    ]

    # 2 Personalized Models
    mock_models = [
        ModelItem(
            model_id="SLM-NIKON-3DPrinter-DefectDetector-YOLO",
            author="BIREX-CompetenceCenter",
            base_model="yolov5-cnn",
            readme_file="YOLO-based CNN model for defect and anomaly detection on images captured from NIKON SLM metal 3D printers.",
            license="apache-2.0",
            language=["none"],
            downloads=980,
            likes=60,
            tags=["vision", "object-detection", "anomaly-detection", "yolo", "cnn", "industrial",
                  "additive-manufacturing", "nikon"],
            pipeline_tag="object-detection",
            library_name="ultralytics",
            created_at="2024-03-01"
        ),
        ModelItem(
            model_id="SISMA-3DPrinter-DefectDetector-YOLO",
            author="BIREX-CompetenceCenter",
            base_model="yolov5-cnn",
            readme_file="YOLO-based CNN model for detecting defects and anomalies in images from SISMA metal 3D printers.",
            license="apache-2.0",
            language=["none"],
            downloads=910,
            likes=55,
            tags=["vision", "object-detection", "anomaly-detection", "yolo", "cnn", "industrial",
                  "additive-manufacturing", "sisma"],
            pipeline_tag="object-detection",
            library_name="ultralytics",
            created_at="2024-03-01"
        )
    ]

    print("Adding personalized mock datasets...")
    for dataset in tqdm(mock_datasets, desc="Mock datasets"):
        try:
            # Generate random embedding
            embedding = np.random.randn(1024).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)

            dataset_service.add_dataset(dataset, embedding, data=[placeholder_file])
        except Exception as e:
            print(f"Error adding dataset {dataset.dataset_id}: {e}")

    print("Adding personalized mock models...")
    for model in tqdm(mock_models, desc="Mock models"):
        try:
            # Generate random embedding
            embedding = np.random.randn(1024).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)

            model_service.add_model(model, embedding, weights=[placeholder_file])
        except Exception as e:
            print(f"Error adding model {model.model_id}: {e}")

    print("Mock personalized data population complete.")


def data_already_present():
    """Check if there are already datasets or models in Qdrant or MinIO."""

    qdrant_service = get_qdrant_service()
    storage_service = get_storage_service()

    # Check Qdrant collections
    qdrant_has_data = False
    try:
        datasets_count = qdrant_service.client.count(
            collection_name=settings.QDRANT_DATASETS_COLLECTION
        )
        models_count = qdrant_service.client.count(
            collection_name=settings.QDRANT_MODELS_COLLECTION
        )
        qdrant_has_data = datasets_count.count > 0 or models_count.count > 0
    except Exception:
        qdrant_has_data = False

    # Check MinIO storage
    storage_has_data = False
    try:
        datasets_list = storage_service.list_datasets()
        models_list = storage_service.list_models()
        storage_has_data = len(datasets_list) > 0 or len(models_list) > 0
    except Exception:
        storage_has_data = False

    return qdrant_has_data or storage_has_data


if __name__ == "__main__":
    # Update paths as needed
    DATASETS_CSV = "data/datasets_hg_embeddings_sm.csv"
    MODELS_CSV = "data/models_hg_embeddings_sm.csv"

    if data_already_present():
        print("Data already present in storage or vector DB. Skipping population.")
    else:
        bulk_populate_datasets(DATASETS_CSV)
        bulk_populate_models(MODELS_CSV)
        populate_mock_data()
        qdrant_service = get_qdrant_service()
        datasets_count = qdrant_service.client.count(collection_name=settings.QDRANT_DATASETS_COLLECTION)
        models_count = qdrant_service.client.count(collection_name=settings.QDRANT_MODELS_COLLECTION)
        print(f"After CSV: Datasets={datasets_count.count}, Models={models_count.count}")

