# src/api/main.py
"""
Main FastAPI application with router registration and middleware.
Replaces the old app.py with proper dependency injection and error handling.
"""

import os
from pathlib import Path
from typing import Optional
from datetime import datetime
import logging

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware

from src.agent.model_runner import ModelRunner
from src.agent.diagnostics import DiagnosticTools
from src.storage import get_db
from src.api.schemas import ErrorDetail
from src.api.routers import health, score, review, audit, metrics
# Note: explain router imported later if needed due to optional LLM dependencies

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration from environment
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models/lgbm_final"))
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "data/vectorstore/faiss")
LLM_MODE = os.getenv("LLM_MODE", "none")
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "models/llm/Phi-3-mini-4k-instruct-q4.gguf")
ENABLE_RETRIEVAL = os.getenv("ENABLE_RETRIEVAL", "false").lower() == "true"

# Global state (initialized at startup)
_model_runner: Optional[ModelRunner] = None
_diag: Optional[DiagnosticTools] = None


# ============================================================================
# FastAPI App Creation
# ============================================================================

app = FastAPI(
    title="AML Risk Decision Engine",
    description="Production-grade fraud/AML transaction monitoring with explainability, review workflow, and audit trail",
    version="0.1.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Dependency Injection
# ============================================================================

def get_model_runner() -> ModelRunner:
    """Get initialized model runner."""
    global _model_runner
    if _model_runner is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return _model_runner


def get_diagnostics() -> DiagnosticTools:
    """Get initialized diagnostics tools."""
    global _diag
    if _diag is None:
        raise HTTPException(status_code=503, detail="Diagnostics not initialized")
    return _diag


# ============================================================================
# Startup & Shutdown
# ============================================================================

@app.on_event("startup")
def startup_event():
    """Initialize services on startup."""
    global _model_runner, _diag
    
    try:
        logger.info("Initializing model runner...")
        _model_runner = ModelRunner(MODEL_DIR)
        logger.info(f"✓ Model loaded from {MODEL_DIR}")
        
        logger.info("Initializing diagnostics...")
        _diag = DiagnosticTools()
        logger.info("✓ Diagnostics initialized")
        
        logger.info("Initializing database...")
        db = get_db()
        logger.info("✓ Database initialized")
        
        logger.info("✓ All services initialized successfully")
    
    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        raise


@app.on_event("shutdown")
def shutdown_event():
    """Clean up on shutdown."""
    logger.info("Shutting down...")
    db = get_db()
    db.close()
    logger.info("✓ Database closed")


# ============================================================================
# Router Registration
# ============================================================================

# Include all routers
app.include_router(health.router)
app.include_router(score.router)
app.include_router(review.router)
app.include_router(audit.router)
app.include_router(metrics.router)

# Include explain router if LLM dependencies are available
try:
    from src.api.routers import explain
    app.include_router(explain.router)
except ImportError as e:
    logger.warning(f"Explain router not available (missing LLM dependencies): {e}")


# ============================================================================
# Exception Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with structured response."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error_code": f"HTTP_{exc.status_code}",
            "message": exc.detail,
            "details": None,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error_code": "INTERNAL_ERROR",
            "message": "An unexpected error occurred",
            "details": None,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )


# ============================================================================
# Root Routes
# ============================================================================

@app.get("/")
def root():
    """API root."""
    return {
        "title": "AML Risk Decision Engine",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)