from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.models.schemas import SearchRequest
from src.services.models import ModelService
from src.services.datasets import DatasetService

templates = Jinja2Templates(directory="templates")

app = FastAPI(
    title="Catalog Search API",
    description="Digital Twins API for searching and retrieving catalog items.",
)

dataset_service = DatasetService()
model_service = ModelService()


@app.get("/", response_class=HTMLResponse)
async def root():
    return templates.TemplateResponse("index.html", {"request": {}})


@app.get("/models")
async def get_models():
    return {"models": model_service.get_all()}


@app.get("/datasets", response_class=HTMLResponse)
async def view_datasets(request: Request):
    datasets = dataset_service.get_sample()
    return templates.TemplateResponse("datasets.html", {
        "request": request,
        "datasets": datasets
    })


@app.post("/search/datasets")
async def search_datasets(search_request: SearchRequest):
    results = dataset_service.search(search_request.description)
    return {
        "query": search_request.description,
        "results": results,
    }


