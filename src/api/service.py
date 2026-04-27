# src/api/service.py
"""
Service layer that contains business logic, decoupled from HTTP routes.
This enables testing without FastAPI and reusability across interfaces.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from uuid import uuid4
import numpy as np
import json
import logging

from src.storage import get_db, DatabaseError

logger = logging.getLogger(__name__)


class ScoringService:
    """Orchestrates scoring, case creation, and decision logic."""
    
    def __init__(self, model_runner=None):
        """
        Initialize scoring service.
        
        Args:
            model_runner: Initialized ModelRunner instance (lazy if not provided)
        """
        self.model_runner = model_runner
        self.db = get_db()
    
    def score_and_create_case(
        self,
        tx_features: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Score a transaction and create a persistent case record.
        
        Uses strict feature contract validation from ModelRunner.
        
        Args:
            tx_features: Transaction features dictionary
        
        Returns:
            Dictionary with scoring results and case metadata
        
        Raises:
            ValueError: If feature validation fails
            Exception: If database operation fails
        """
        # Run model with feature contract validation
        # ModelRunner.run() validates features and returns versions
        try:
            model_result = self.model_runner.run(tx_features)
        except ValueError as e:
            logger.error(f"Feature validation failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            raise
        
        # Extract results
        score = model_result["score"]
        decision = model_result["decision"]
        model_version = model_result["model_version"]
        threshold_version = model_result["threshold_version"]
        feature_contract_version = model_result["feature_contract_version"]
        schema_hash = model_result["schema_hash"]
        
        # Determine review flag from decision
        review_flag = decision == "ALERT"
        threshold = float(self.model_runner.threshold)
        
        # Create identifiers
        case_id = str(uuid4())
        request_id = str(uuid4())
        
        # Create case in database
        try:
            case_record = self.db.create_case(
                case_id=case_id,
                request_id=request_id,
                model_version=model_version,
                threshold_version=threshold_version,
                feature_contract_version=feature_contract_version,
                score=score,
                raw_features=tx_features,
            )
        except DatabaseError as e:
            logger.error(f"Failed to create case: {e}")
            raise
        
        # Return structured response
        return {
            "case_id": case_id,
            "request_id": request_id,
            "score": score,
            "review_flag": review_flag,
            "decision": decision,
            "threshold_used": threshold,
            "model_version": model_version,
            "threshold_version": threshold_version,
            "feature_contract_version": feature_contract_version,
            "schema_hash": schema_hash,
            "review_status": "queued_for_review",
            "timestamp": datetime.utcnow(),
        }


class ReviewService:
    """Orchestrates case review workflow."""
    
    def __init__(self):
        """Initialize review service."""
        self.db = get_db()
    
    def submit_review(
        self,
        case_id: str,
        reviewer_id: str,
        decision: str,
        note: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Submit a manual review decision for a case.
        
        Args:
            case_id: Case identifier
            reviewer_id: ID of reviewer
            decision: APPROVED, REJECTED, or ESCALATED
            note: Optional reviewer note
        
        Returns:
            Dictionary with review record
        
        Raises:
            ValueError: If case not found or invalid decision
        """
        valid_decisions = {'APPROVED', 'REJECTED', 'ESCALATED'}
        if decision not in valid_decisions:
            raise ValueError(f"Invalid decision. Must be one of {valid_decisions}")
        
        # Verify case exists
        case = self.db.get_case(case_id)
        if case is None:
            raise ValueError(f"Case {case_id} not found")
        
        # Record review in database
        review_id = str(uuid4())
        try:
            review_record = self.db.record_review(
                review_id=review_id,
                case_id=case_id,
                reviewer_id=reviewer_id,
                decision=decision,
                note=note,
            )
        except DatabaseError as e:
            logger.error(f"Failed to record review: {e}")
            raise
        
        return review_record
    
    def get_case_audit_trail(self, case_id: str) -> List[Dict[str, Any]]:
        """Get audit trail for a case."""
        case = self.db.get_case(case_id)
        if case is None:
            raise ValueError(f"Case {case_id} not found")
        
        return self.db.get_audit_trail(case_id)


class ExplanationService:
    """Orchestrates explanation generation with RAG + LLM agent."""
    
    def __init__(self, model_runner=None):
        """Initialize explanation service."""
        self.model_runner = model_runner
        self.agent = None
        self._init_agent()
    
    def _init_agent(self):
        """Initialize RAG agent with graceful degradation."""
        try:
            from src.agent.aml_agent import AMLAgent, AgentConfig
            from src.agent.diagnostics import DiagnosticTools
            import os
            
            llm_mode = os.getenv("LLM_MODE", "none")
            llm_path = os.getenv("LLM_MODEL_PATH", "models/llm/Phi-3-mini-4k-instruct-q4.gguf")
            vectorstore_dir = os.getenv("VECTORSTORE_DIR", "data/vectorstore/faiss")
            
            config = AgentConfig(
                llm_mode=llm_mode,
                llm_model_path=llm_path,
                vectorstore_dir=vectorstore_dir,
            )
            
            diag = DiagnosticTools()
            self.agent = AMLAgent(
                model_runner=self.model_runner,
                diagnostics=diag,
                config=config
            )
            logger.info("✓ RAG agent initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize agent: {e}. Using fallback mode.")
            self.agent = None
    
    def explain_score(
        self,
        case_id: str,
        score: float,
        tx_features: Dict[str, Any],
        tx_text: str = None,
    ) -> Dict[str, Any]:
        """
        Generate explanation for a score using RAG + LLM or fallback.
        
        Args:
            case_id: Case being explained
            score: Risk score
            tx_features: Transaction features
            tx_text: Natural language description for retrieval
        
        Returns:
            Dictionary with explanation details
        """
        # Extract top features
        top_features = self._get_feature_importance(tx_features)
        
        # Try agent-based explanation if available
        if self.agent:
            try:
                agent_result = self.agent.run(
                    score=score,
                    top_features=top_features,
                    tx_text=tx_text or self._features_to_text(tx_features),
                )
                
                return {
                    "case_id": case_id,
                    "score": score,
                    "model_version": "1.0.0",
                    "explanation_type": "agent_rag",
                    "decision": agent_result.get("decision", "PASS"),
                    "rationale": agent_result.get("rationale", []),
                    "cited_docs": agent_result.get("cited_docs", []),
                    "top_features": top_features,
                    "llm_enabled": agent_result.get("llm_enabled", False),
                    "timestamp": datetime.utcnow(),
                }
            except Exception as e:
                logger.warning(f"Agent explanation failed: {e}. Falling back to feature importance.")
        
        # Fallback to feature importance
        return {
            "case_id": case_id,
            "score": score,
            "model_version": "1.0.0",
            "explanation_type": "feature_importance",
            "top_features": top_features,
            "timestamp": datetime.utcnow(),
        }
    
    def _features_to_text(self, tx_features: Dict[str, Any]) -> str:
        """Convert features dict to natural language for retrieval."""
        high_risk_features = [
            f for f, v in tx_features.items()
            if isinstance(v, (int, float)) and v > 0.5
        ]
        return f"Transaction with features: {', '.join(high_risk_features)}"
    
    def _get_feature_importance(self, tx_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract top contributing features using model feature importance.
        """
        try:
            # Get feature importance from LightGBM model
            importance = self.model_runner.model.feature_importance()
            feature_names = self.model_runner.features
            
            # Create tuples of (feature, importance)
            feature_scores = list(zip(feature_names, importance))
            
            # Sort by absolute importance (descending)
            feature_scores.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Get top 5 features
            top_features = []
            for feature_name, importance_value in feature_scores[:5]:
                feature_value = tx_features.get(feature_name, 0)
                contribution = "positive" if importance_value > 0 else "negative"
                
                top_features.append({
                    "feature_name": feature_name,
                    "importance_value": float(abs(importance_value)),
                    "contribution": contribution,
                    "feature_value": float(feature_value) if isinstance(feature_value, (int, float)) else str(feature_value),
                })
            
            return top_features
        
        except Exception as e:
            logger.warning(f"Failed to extract feature importance: {e}")
            return []


class HealthService:
    """Health check and system status."""
    
    def __init__(self, model_runner=None):
        """Initialize health service."""
        self.model_runner = model_runner
        self.db = get_db()
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive health status.
        
        Returns:
            Dictionary with health information
        """
        # Check model
        try:
            model_status = "ready" if self.model_runner.model is not None else "error"
        except Exception:
            model_status = "error"
        
        # Check database
        try:
            self.db.get_case_count_by_status()
            db_status = "ok"
        except Exception:
            db_status = "error"
        
        overall_status = "ok" if (model_status == "ready" and db_status == "ok") else "degraded"
        
        return {
            "status": overall_status,
            "version": "0.1.0",
            "model_version": "1.0.0",
            "model_status": model_status,
            "threshold_version": "1.0.0",
            "feature_contract_version": "1.0.0",
            "database_status": db_status,
            "timestamp": datetime.utcnow(),
        }