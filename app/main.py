from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings
from app.logger import get_logger

logger = get_logger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "A Computer Vision backend that analyzes uploaded ID document images "
            "for image quality, document readiness, OCR text extraction, and risk assessment.\n\n"
            "> **Disclaimer**: This is a portfolio/educational project. "
            "It does **not** perform legal identity verification."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/v1")

    @app.get("/health", tags=["Health"], summary="Health check")
    async def health() -> dict:
        return {"status": "ok", "version": settings.app_version}

    logger.info(f"Started '{settings.app_name}' v{settings.app_version}")
    return app


app = create_app()
