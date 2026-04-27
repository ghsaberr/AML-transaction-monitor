# src/api/routers/explain.py
"""Explanation endpoint with RAG + local LLM."""

from fastapi import APIRouter, HTTPException
from typing import Optional
from src.api.schemas import ExplainRequest, ExplainResponse
from src.api.service import ExplanationService
import os
import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["explain"])

# Config
LLM_MODE = os.getenv("LLM_MODE", "none")
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "models/llm/Phi-3-mini-4k-instruct-q4.gguf")
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "data/vectorstore/faiss")

# Global explanation service singleton
_explanation_service: Optional[ExplanationService] = None


def get_explanation_service(model_runner) -> ExplanationService:
    """Get or create explanation service with agent."""
    global _explanation_service
    if _explanation_service is None:
        _explanation_service = ExplanationService(model_runner)
    return _explanation_service


@router.post(
    "/explain",
    response_model=ExplainResponse,
    summary="Explain Score",
    description="Generate explanation for a transaction score using RAG + local LLM",
    tags=["explain"]
)
def explain(req: ExplainRequest):
    """
    Explain a transaction score using RAG (Retrieval-Augmented Generation).
    
    Features:
    - Retrieves similar historical AML cases from FAISS
    - Generates explanation using local Phi-3 LLM (if available)
    - Falls back to feature importance if LLM unavailable
    - Cites document IDs in the rationale
    
    Args:
        req: ExplainRequest with case_id, tx_features, and optional tx_text
    
    Returns:
        ExplainResponse with decision, rationale, and cited documents
    
    Raises:
        HTTPException 400: If invalid input
        HTTPException 500: If explanation generation fails
    """
    try:
        # Import here to avoid circular imports during tests
        from src.api.main import get_model_runner
        
        model_runner = get_model_runner()
        service = get_explanation_service(model_runner)
        
        # Get score from request or default
        score = getattr(req, "score", 0.5)  # Could be in request or DB
        
        # Generate explanation with RAG + LLM
        result = service.explain_score(
            case_id=req.case_id,
            score=score,
            tx_features=req.tx_features,
            tx_text=req.tx_text if hasattr(req, "tx_text") else None,
        )
        
        return result
    
    except ValueError as e:
        logger.warning(f"Invalid input: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Explanation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")