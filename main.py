"""AutoMLOps Platform - FastAPI application entrypoint.



Wires together dataset management, the AutoML pipeline orchestrator,

prediction serving, and drift monitoring into a single API surface.

"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from contextlib import asynccontextmanager



from fastapi import FastAPI, HTTPException

from fastapi.middleware.cors import CORSMiddleware

from fastapi.openapi.docs import get_swagger_ui_html

from fastapi.responses import FileResponse, RedirectResponse

from fastapi.staticfiles import StaticFiles



from api import routes_agent, routes_chat, routes_dataset, routes_mle, routes_monitor, routes_multimodal, routes_pipeline, routes_predict

from core.config import BASE_DIR, get_settings

from core.logging_config import configure_logging, get_logger



STATIC_DIR = BASE_DIR / "static"

SWAGGER_DIR = STATIC_DIR / "swagger"

UI_DIR = STATIC_DIR / "ui"

UI_INDEX = UI_DIR / "index.html"

DOCS_DIR = BASE_DIR / "docs"



configure_logging()

logger = get_logger(__name__)





@asynccontextmanager

async def lifespan(_: FastAPI):

    settings = get_settings()

    logger.info("AutoMLOps Platform starting up")

    logger.info("MLflow tracking URI: %s", settings.mlflow_tracking_uri)

    logger.info("Upload dir: %s | Artifacts dir: %s", settings.upload_dir, settings.artifacts_dir)

    if UI_INDEX.exists():

        logger.info("Web UI enabled at /ui/")

    else:

        logger.warning("Web UI not found at %s", UI_INDEX)

    yield





app = FastAPI(

    title="AutoMLOps Platform",

    description=(

        "Upload a dataset and let the platform profile it, clean it, handle outliers, "

        "engineer features, select & tune a model with Optuna (10+ algorithms), track experiments in MLflow, "

        "explain predictions with SHAP, monitor drift with Evidently, get AI recommendations via LangGraph + RAG, "

        "and serve the result as an API."

    ),

    version="1.0.0",

    lifespan=lifespan,

    docs_url=None,

    redoc_url=None,

)



if SWAGGER_DIR.exists():

    app.mount("/static/swagger", StaticFiles(directory=SWAGGER_DIR), name="swagger-ui")



if UI_DIR.exists():

    app.mount("/ui/css", StaticFiles(directory=UI_DIR / "css"), name="ui-css")

    app.mount("/ui/js", StaticFiles(directory=UI_DIR / "js"), name="ui-js")



if DOCS_DIR.exists():

    app.mount("/assets/docs", StaticFiles(directory=DOCS_DIR), name="project-docs")



app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],

)



app.include_router(routes_dataset.router)

app.include_router(routes_pipeline.router)

app.include_router(routes_predict.router)

app.include_router(routes_monitor.router)

app.include_router(routes_agent.router)
app.include_router(routes_chat.router)
app.include_router(routes_mle.router)
app.include_router(routes_multimodal.router)





@app.get("/docs", include_in_schema=False)

async def swagger_ui():

    return get_swagger_ui_html(

        openapi_url=app.openapi_url,

        title=f"{app.title} - Swagger UI",

        swagger_js_url="/static/swagger/swagger-ui-bundle.js",

        swagger_css_url="/static/swagger/swagger-ui.css",

    )





@app.get("/ui", include_in_schema=False)

async def ui_redirect():

    return RedirectResponse(url="/ui/")





@app.get("/ui/", include_in_schema=False)

async def serve_ui():

    if not UI_INDEX.exists():

        raise HTTPException(status_code=404, detail="UI not found. Ensure static/ui/index.html exists.")

    return FileResponse(UI_INDEX)





@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/ui/")


@app.get("/api", tags=["health"])
async def api_info() -> dict[str, str]:
    return {
        "service": "AutoMLOps Platform",
        "status": "ok",
        "ui": "/ui/",
        "api_docs": "/docs",
        "workflow_diagram": "/assets/docs/workflow_complete.png",
        "agent": "/agent/analyze",
        "chatbot": "/chat/status",
    }


@app.get("/health", tags=["health"])

async def health() -> dict[str, str]:

    return {"status": "healthy"}


