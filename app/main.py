"""
FastAPI application entry point.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.model_loader import load_all_models
from app.routers import predict, data
from app.routers.data import load_sample_data
from app.keep_alive import keep_alive_loop
from app.models import HealthResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load models + sample data. Shutdown: cleanup."""
    logger.info("=" * 60)
    logger.info("  Starting ToxGuard Content Flagger API")
    logger.info("=" * 60)

    # Load ML models into memory
    load_all_models()

    # Load sample data for the randomize feature
    load_sample_data()

    # Start keep-alive background task
    keep_alive_task = asyncio.create_task(keep_alive_loop())

    logger.info("=" * 60)
    logger.info("  ✅ API Ready — all models loaded")
    logger.info("=" * 60)

    yield

    # Shutdown
    keep_alive_task.cancel()
    logger.info("API shutting down.")


app = FastAPI(
    title="ToxGuard Content Flagger API",
    description="Multi-Label Content Moderation API powered by Dual BiLSTM pipelines.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(predict.router)
app.include_router(data.router)

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    from app.model_loader import _models
    return HealthResponse(
        status="healthy",
        models_loaded=len(_models) > 0,
        models_count=len(_models)
    )

@app.get("/", tags=["System"])
async def root():
    """API root"""
    return {
        "name": "ToxGuard Content Flagger API",
        "docs": "/docs",
        "health": "/health"
    }
