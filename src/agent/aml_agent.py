from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import os
import logging

from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    vectorstore_dir: str = "data/vectorstore/faiss"
    top_k: int = 3

    # LLM
    llm_mode: str = "none"  # "none" | "local"
    llm_model_path: str = "models/llm/Phi-3-mini-4k-instruct-q4.gguf"
    llm_temperature: float = 0.0
    llm_n_ctx: int = 2048


class AMLAgent:
    """Agent for explainable AML scoring using RAG + local LLM."""
    
    def __init__(self, model_runner, diagnostics, config: Optional[AgentConfig] = None):
        self.model_runner = model_runner
        self.diagnostics = diagnostics
        self.config = config or AgentConfig()
        self.llm = None
        self.retriever = None

        try:
            # Embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            # Vectorstore - load if exists
            if os.path.exists(self.config.vectorstore_dir):
                vectorstore = FAISS.load_local(
                    self.config.vectorstore_dir,
                    embeddings,
                    allow_dangerous_deserialization=True,
                )
                self.retriever = vectorstore.as_retriever(
                    search_kwargs={"k": self.config.top_k}
                )
                logger.info(f"✓ Loaded FAISS vectorstore from {self.config.vectorstore_dir}")
            else:
                logger.warning(f"FAISS vectorstore not found at {self.config.vectorstore_dir}")
        except Exception as e:
            logger.error(f"Failed to load vectorstore: {e}")
            self.retriever = None

        # Prompt template
        self.prompt = PromptTemplate(
            input_variables=["score", "top_features", "docs"],
            template="""You are an expert AML analyst. Analyze the following transaction:

Risk Score: {score}
Top Features: {top_features}

Retrieved Similar Cases:
{docs}

Provide:
1. Final decision (ALERT or PASS)
2. Key risk factors
3. Cite retrieved document IDs (e.g., [rule_001])
4. Confidence level (0-1)

Be concise and structured."""
        )

        # LLM initialization with graceful degradation
        if self.config.llm_mode == "local":
            self._init_llm()

    def _init_llm(self):
        """Initialize local LLM with graceful degradation."""
        try:
            # Check if model file exists
            if not os.path.exists(self.config.llm_model_path):
                logger.warning(f"LLM model not found at {self.config.llm_model_path}")
                self.llm = None
                return

            # Import and initialize llama-cpp-python
            from langchain_community.llms import LlamaCpp

            self.llm = LlamaCpp(
                model_path=self.config.llm_model_path,
                temperature=self.config.llm_temperature,
                n_ctx=self.config.llm_n_ctx,
                n_gpu_layers=0,  # CPU-only for portability
                verbose=False,
            )
            logger.info(f"✓ Loaded LLM from {self.config.llm_model_path}")
        except ImportError as e:
            logger.warning(f"llama-cpp-python not installed: {e}")
            self.llm = None
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            self.llm = None

    def run(
        self,
        score: float,
        top_features: List[Dict[str, Any]],
        tx_text: str = None,
    ) -> Dict[str, Any]:
        """
        Generate explanation for a score using RAG + LLM.
        
        Args:
            score: Risk score (0-1)
            top_features: List of top contributing features
            tx_text: Transaction description for retrieval
        
        Returns:
            Dict with decision, rationale, and cited documents
        """
        result = {
            "score": score,
            "decision": "ALERT" if score >= 0.5 else "PASS",
            "rationale": [],
            "cited_docs": [],
            "llm_enabled": self.llm is not None,
            "agent_enabled": True,
        }

        # 1) Retrieve similar cases
        retrieved_docs = []
        cited_docs = []
        if self.retriever and tx_text:
            try:
                docs = self.retriever.invoke(tx_text)
                retrieved_docs = [
                    f"[{d.metadata.get('doc_id', 'UNKNOWN')}] {d.page_content}"
                    for d in docs
                ]
                cited_docs = [d.metadata.get('doc_id', 'UNKNOWN') for d in docs]
                result["cited_docs"] = cited_docs
            except Exception as e:
                logger.warning(f"Retrieval failed: {e}")

        # 2) Format features
        features_text = ", ".join(
            [f["feature_name"] for f in top_features[:3]]
        )

        # 3) Build context
        docs_text = "\n".join(retrieved_docs) if retrieved_docs else "[No similar cases found]"

        # 4) Generate prompt
        try:
            prompt_text = self.prompt.format(
                score=f"{score:.2%}",
                top_features=features_text,
                docs=docs_text,
            )
        except Exception as e:
            logger.error(f"Prompt formatting failed: {e}")
            prompt_text = None

        # 5) LLM reasoning (if available)
        if self.llm and prompt_text:
            try:
                llm_response = self.llm.invoke(prompt_text)
                result["rationale"] = [llm_response]
                logger.info("✓ LLM generated explanation")
            except Exception as e:
                logger.warning(f"LLM inference failed: {e}")
                result["rationale"] = self._fallback_rationale(score, top_features, cited_docs)
        else:
            result["rationale"] = self._fallback_rationale(score, top_features, cited_docs)

        return result

    def _fallback_rationale(
        self,
        score: float,
        top_features: List[Dict[str, Any]],
        cited_docs: List[str],
    ) -> List[str]:
        """Fallback explanation when LLM unavailable."""
        rationale = [
            f"Risk Score: {score:.2%}",
            f"Top Risk Factors: {', '.join([f['feature_name'] for f in top_features[:3]])}",
        ]
        if cited_docs:
            rationale.append(f"Similar Cases: {', '.join(cited_docs)}")
        return rationale
