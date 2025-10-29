import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, Request, HTTPException, UploadFile
from fastapi.params import Form, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from minio import S3Error
from starlette.responses import RedirectResponse, JSONResponse, StreamingResponse
from src.dependencies import get_dataset_service, get_model_service
from src.models.schemas import DatasetItem, ModelItem
from src.config import settings

templates = Jinja2Templates(directory="templates")

dataset_service = None
model_service = None
search_results_cache = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global dataset_service, model_service
    # Initialize AFTER populate_data completes
    dataset_service = get_dataset_service()
    model_service = get_model_service()
    yield


app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    lifespan=lifespan
)


@app.get("/", response_class=HTMLResponse)
async def root():
    return templates.TemplateResponse("index.html", {"request": {}})


@app.get("/datasets", response_class=HTMLResponse)
async def view_datasets(request: Request):
    datasets = dataset_service.get_sample()
    return templates.TemplateResponse("datasets.html", {
        "request": request,
        "datasets": datasets
    })


@app.get("/api/datasets")
async def api_sample_datasets():
    results = dataset_service.get_sample()
    return JSONResponse(content={
        "count": len(results),
        "results": [
            {
                "dataset_id": result.dataset_id,
                "author": result.author,
                "tags": result.tags,
                "readme_file": result.readme_file,
            }
            for result in results
        ]
    })

@app.get("/api/models")
async def api_sample_models():
    results = model_service.get_sample()
    return JSONResponse(content={
        "count": len(results),
        "results": [
            {
                "model_id": result.model_id,
                "author": result.author,
                "pipeline_tag": result.pipeline_tag,
                "readme_file": result.readme_file,
            }
            for result in results
        ]
    })

@app.get("/datasets/add", response_class=HTMLResponse)
async def show_add_dataset_form(request: Request):
    return templates.TemplateResponse("add_dataset.html", {"request": request})


@app.post("/datasets/add")
async def add_dataset(
        dataset_id: str = Form(...),
        readme_file: str = Form(...),
        tags: str = Form(...),
        language: str = Form(...),
        file: UploadFile = File(...)
):
    # Parse comma-separated values
    tags_list = [tag.strip() for tag in tags.split(",")]
    language_list = [lang.strip() for lang in language.split(",")]

    # Read file content
    file_content = await file.read()

    # Split the dataset_id
    name, author = dataset_id.split("/")

    dataset_item = DatasetItem(
        dataset_id=name,
        author=author,
        created_at="2024-01-01",
        readme_file=readme_file,
        downloads=0,
        likes=0,
        tags=tags_list,
        language=language_list,
        license="mit",
        multilinguality=["monolingual"],
        size_categories=["1K<n<10K"],
        task_categories=["text-classification"]
    )

    try:
        embedding = dataset_service.rag_service.embedding_service.encode_dataset_item(dataset_item)
        dataset_service.add_dataset(dataset_item, embedding, data=[file_content])
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    return JSONResponse(content={"message": "Dataset added successfully", "dataset_id": dataset_id})


@app.post("/search/datasets")
async def search_datasets(description: str = Form(...)):
    results = dataset_service.search(description)

    search_id = str(uuid.uuid4())
    search_results_cache[search_id] = results

    return RedirectResponse(url=f"/search/datasets/{search_id}", status_code=303)


@app.get("/search/datasets/{search_id}", response_class=HTMLResponse)
async def get_search_results(search_id: str, request: Request):
    if search_id not in search_results_cache:
        raise HTTPException(status_code=404, detail="Search ID not found")

    results = search_results_cache[search_id]

    return templates.TemplateResponse("results_dataset.html", {
        "request": request,
        "results": results
    })


@app.get("/datasets/{dataset_id:path}/data")
async def download_dataset_data(dataset_id: str):
    bucket = dataset_service.storage_service.datasets_bucket
    object_path = f"{dataset_id}/data/placeholder.txt"

    try:
        response = dataset_service.storage_service.client.get_object(bucket, object_path)
    except S3Error:
        raise HTTPException(status_code=404, detail="Dataset file not found")

    filename = f"{dataset_id.replace('/', '-')}_data.txt"

    return StreamingResponse(
        response,
        media_type="text/plain",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/models", response_class=HTMLResponse)
async def view_models(request: Request):
    models = model_service.get_sample()
    return templates.TemplateResponse("models.html", {
        "request": request,
        "models": models
    })


@app.get("/models/add", response_class=HTMLResponse)
async def show_add_model_form(request: Request):
    return templates.TemplateResponse("add_model.html", {"request": request})


@app.post("/models/add")
async def add_model(
        model_id: str = Form(...),
        readme_file: str = Form(...),
        tags: str = Form(...),
        language: str = Form(...),
        file: UploadFile = File(...)
):
    # Parse comma-separated values
    tags_list = [tag.strip() for tag in tags.split(",")]
    language_list = [lang.strip() for lang in language.split(",")]

    # Read file content
    file_content = await file.read()

    # Split the model_id
    name, author = model_id.split("/")

    model_item = ModelItem(
        model_id=name,
        base_model="gpt-3",
        author=author,
        readme_file=readme_file,
        license="mit",
        language=language_list,
        downloads=0,
        likes=0,
        tags=tags_list,
        pipeline_tag="text-generation",
        library_name="transformers",
        created_at="2024-01-01"
    )

    try:
        embedding = model_service.rag_service.embedding_service.encode_model_item(model_item)
        model_service.add_model(model_item, embedding, weights=[file_content])
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    return JSONResponse(content={"message": "Model added successfully", "model_id": model_id})


@app.post("/search/models")
async def search_models(description: str = Form(...)):
    results = model_service.search(description)

    search_id = str(uuid.uuid4())
    search_results_cache[search_id] = results

    return RedirectResponse(url=f"/search/models/{search_id}", status_code=303)


@app.get("/search/models/{search_id}", response_class=HTMLResponse)
async def get_model_search_results(search_id: str, request: Request):
    if search_id not in search_results_cache:
        raise HTTPException(status_code=404, detail="Search ID not found")

    results = search_results_cache[search_id]

    return templates.TemplateResponse("results_model.html", {
        "request": request,
        "results": results
    })


@app.get("/models/{model_id:path}/weights")
async def download_model_weights(model_id: str):
    bucket = model_service.storage_service.models_bucket
    object_path = f"{model_id}/weights/placeholder.txt"

    try:
        response = model_service.storage_service.client.get_object(bucket, object_path)
    except S3Error:
        raise HTTPException(status_code=404, detail="Model weights file not found")

    filename = f"{model_id.replace('/', '-')}_weights.txt"

    return StreamingResponse(
        response,
        media_type="text/plain",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/search/datasets")
async def api_search_datasets(description: str = Form(...)):
    """
    API endpoint for searching datasets that returns JSON results directly.
    Useful for programmatic access and testing the download pipeline.
    """
    results = dataset_service.search(description)

    return JSONResponse(content={
        "query": description,
        "count": len(results),
        "results": [
            {
                "dataset_id": result.dataset_id,
                "author": result.author,
                "tags": result.tags,
                "task_categories": result.task_categories,
            }
            for result in results
        ]
    })


@app.post("/api/search/models")
async def api_search_models(description: str = Form(...)):
    """
    API endpoint for searching models that returns JSON results directly.
    Useful for programmatic access and testing the download pipeline.
    """
    results = model_service.search(description)

    return JSONResponse(content={
        "query": description,
        "count": len(results),
        "results": [
            {
                "model_id": result.model_id,
                "author": result.author,
                "pipeline_tag": result.pipeline_tag,
                "library_name": result.library_name,
            }
            for result in results
        ]
    })


@app.delete("/api/datasets/{dataset_id:path}")
async def delete_dataset(dataset_id: str):
    """Delete a dataset by its ID."""
    if not dataset_service.storage_service.get_dataset_by_id(dataset_id):
        raise HTTPException(status_code=404, detail="Dataset not found")

    success = dataset_service.delete_dataset(dataset_id)

    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete dataset")

    return JSONResponse(content={"message": "Dataset deleted successfully", "dataset_id": dataset_id})


@app.delete("/api/models/{model_id:path}")
async def delete_model(model_id: str):
    """Delete a model by its ID."""
    if not model_service.storage_service.get_model_by_id(model_id):
        raise HTTPException(status_code=404, detail="Model not found")

    success = model_service.delete_model(model_id)

    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete model")

    return JSONResponse(content={"message": "Model deleted successfully", "model_id": model_id})
