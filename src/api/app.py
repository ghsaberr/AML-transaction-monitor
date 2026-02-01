from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.agent.model_runner import ModelRunner
from src.agent.diagnostics import DiagnosticTools
from src.agent.aml_agent import AMLAgent, AgentConfig


MODEL_DIR = Path(os.getenv("MODEL_DIR", "models/lgbm_final"))
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "data/vectorstore/faiss")

LLM_MODE = os.getenv("LLM_MODE", "none")  # none | local
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "models/llm/Phi-3-mini-4k-instruct-q4.gguf")

# NEW: retrieval toggle (so Lambda can run without HF downloads)
ENABLE_RETRIEVAL = os.getenv("ENABLE_RETRIEVAL", "false").lower() == "true"


app = FastAPI(title="AML Agent API", version="0.1.0")

_model_runner: Optional[ModelRunner] = None
_diag: Optional[DiagnosticTools] = None
_agent: Optional[AMLAgent] = None


class ScoreRequest(BaseModel):
    tx_features: Dict[str, Any]


class ExplainRequest(BaseModel):
    tx_features: Dict[str, Any]
    raw_tx: Dict[str, Any]
    tx_text: str


@app.on_event("startup")
def startup():
    global _model_runner, _diag, _agent
    _model_runner = ModelRunner(MODEL_DIR)
    _diag = DiagnosticTools()
    _agent = None  # lazy init


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score")
def score(req: ScoreRequest):
    assert _model_runner is not None
    return _model_runner.run(req.tx_features)


def _get_agent() -> AMLAgent:
    global _agent
    if _agent is not None:
        return _agent

    if not ENABLE_RETRIEVAL:
        raise HTTPException(
            status_code=503,
            detail="Explain/agent is disabled in this deployment (ENABLE_RETRIEVAL=false)."
        )

    assert _model_runner is not None
    assert _diag is not None

    cfg = AgentConfig(
        vectorstore_dir=VECTORSTORE_DIR,
        llm_mode=LLM_MODE,
        llm_model_path=LLM_MODEL_PATH,
    )
    _agent = AMLAgent(model_runner=_model_runner, diagnostics=_diag, config=cfg)
    return _agent


@app.post("/explain")
def explain(req: ExplainRequest):
    assert _diag is not None

    agent = _get_agent()
    out_text = agent.run(
        tx_features=req.tx_features,
        raw_tx=req.raw_tx,
        tx_text=req.tx_text,
    )
    return {"agent_response": out_text}
