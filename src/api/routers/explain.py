# src/api/routers/explain.py
"""Explanation endpoint with dual-mode (agent + fallback)."""

from fastapi import APIRouter, Depends, HTTPException
from typing import Optional
from src.api.schemas import ExplainRequest, ExplainResponse
from src.api.service import ExplanationService
from src.agent.model_runner import ModelRunner
from src.agent.aml_agent import AMLAgent, AgentConfig
from src.agent.diagnostics import DiagnosticTools
import os
import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["explain"])

# Config
LLM_MODE = os.getenv("LLM_MODE", "none")
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "models/llm/Phi-3-mini-4k-instruct-q4.gguf")
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "data/vectorstore/faiss")
ENABLE_RETRIEVAL = os.getenv("ENABLE_RETRIEVAL", "false").lower() == "true"

# Global agent instance
_agent: Optional[AMLAgent] = None


def get_explanation_service(model_runner: ModelRunner) -> ExplanationService:
    """Dependency injection for explanation service."""
    return ExplanationService(model_runner)


def _get_agent(model_runner: ModelRunner) -> Optional[AMLAgent]:
    """Get or initialize agent if retrieval enabled."""
    global _agent
    
    if not ENABLE_RETRIEVAL:
        return None
    
    if _agent is not None:
        return _agent
    
    try:
        diag = DiagnosticTools()
        cfg = AgentConfig(
            vectorstore_dir=VECTORSTORE_DIR,
            llm_mode=LLM_MODE,
            llm_model_path=LLM_MODEL_PATH,
        )
        _agent = AMLAgent(model_runner=model_runner, diagnostics=diag, config=cfg)
        return _agent
    except Exception as e:
        logger.warning(f"Failed to initialize agent: {e}")
        return None


@router.post(
    "/explain",
    response_model=ExplainResponse,
    summary="Explain Score",
    description="Generate explanation for a transaction score (agent-based or feature importance)"
)
def explain(
    req: ExplainRequest,
    service: ExplanationService = Depends(get_explanation_service),
    model_runner: ModelRunner = None,
):
    """
    Explain a transaction score.
    
    Dual-mode explanation:
    - If ENABLE_RETRIEVAL=true and agent loads: Uses LLM agent with retrieval
    - Otherwise: Falls back to feature importance (LightGBM)
    
    Args:
        req: ExplainRequest with case_id and tx_features
        service: ExplanationService instance (injected)
        model_runner: ModelRunner instance (injected)
    
    Returns:
        ExplainResponse with either agent_response or top_features
    
    Raises:
        HTTPException 404: If case not found
        HTTPException 500: If explanation generation fails
    """
    try:
        agent = None
        agent_response = None
        
        # Try to get agent-based explanation
        if model_runner:
            agent = _get_agent(model_runner)
        
        if agent:
            try:
                agent_response = agent.run(
                    tx_features=req.tx_features,
                    raw_tx=req.raw_tx or {},
                    tx_text=req.tx_text or "",
                )
            except Exception as e:
                logger.warning(f"Agent explanation failed, using fallback: {e}")
                agent_response = None
        
        # Generate explanation (with fallback to feature importance)
        result = service.explain_score(
            case_id=req.case_id,
            score=0.0,  # TODO: Get actual score from DB
            tx_features=req.tx_features,
            agent_enabled=(agent is not None),
            agent_response=agent_response,
        )
        
        return result
    
    except ValueError as e:
        logger.warning(f"Invalid input: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Explanation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")