import uuid
from fastapi import FastAPI, Request, HTTPException
from fastapi.params import Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from src.services.models import ModelService
from src.services.datasets import DatasetService
from src.services.rag import RAGService
from src.config import settings

templates = Jinja2Templates(directory="templates")

app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
)

rag_service = RAGService()
dataset_service = DatasetService(rag_service)
model_service = ModelService(rag_service)

search_results_cache = {}


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


@app.post("/search/datasets")
async def search_datasets(description: str = Form(...)):
    results = dataset_service.search(description)

    search_id = str(uuid.uuid4())
    search_results_cache[search_id] = results

    return RedirectResponse(url=f"/search/datasets/{search_id}", status_code=303)


@app.get("/search/datasets/{search_id}")
async def get_search_results(search_id: str):
    if search_id not in search_results_cache:
        raise HTTPException(status_code=404, detail="Search ID not found")

    results = search_results_cache[search_id]

    return {"results": results}

@app.get("/models", response_class=HTMLResponse)
async def view_models(request: Request):
    models = model_service.get_sample()
    return templates.TemplateResponse("models.html", {
        "request": request,
        "models": models
    })


@app.post("/search/models")
async def search_models(description: str = Form(...)):
    results = model_service.search(description)

    search_id = str(uuid.uuid4())
    search_results_cache[search_id] = results

    return RedirectResponse(url=f"/search/models/{search_id}", status_code=303)


@app.get("/search/models/{search_id}")
async def get_model_search_results(search_id: str):
    if search_id not in search_results_cache:
        raise HTTPException(status_code=404, detail="Search ID not found")

    results = search_results_cache[search_id]

    return {"results": results}