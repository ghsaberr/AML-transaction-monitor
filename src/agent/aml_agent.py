from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


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
    def __init__(self, model_runner, diagnostics, config: Optional[AgentConfig] = None):
        self.model_runner = model_runner
        self.diagnostics = diagnostics
        self.config = config or AgentConfig()

        # Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Vectorstore
        vectorstore = FAISS.load_local(
            self.config.vectorstore_dir,
            embeddings,
            allow_dangerous_deserialization=True,  # OK for controlled artifacts; see note below
        )
        self.retriever = vectorstore.as_retriever(
            search_kwargs={"k": self.config.top_k}
        )

        # Prompt
        self.prompt = PromptTemplate(
            input_variables=["model_output", "diagnostics", "docs"],
            template="""
You are an AML analyst agent.

Model output:
{model_output}

Deterministic findings:
{diagnostics}

Retrieved documents:
{docs}

Produce:
- Final decision (ALERT or PASS)
- Bullet-point rationale
- Cite document IDs explicitly (e.g. [DOC_12])
"""
        )

        # LLM is optional
        self.llm = None
        if self.config.llm_mode == "local":
            # Import only when needed (keeps cloud/lambda slimmer when llm_mode=none)
            from langchain_community.llms import LlamaCpp

            self.llm = LlamaCpp(
                model_path=self.config.llm_model_path,
                temperature=self.config.llm_temperature,
                n_ctx=self.config.llm_n_ctx,
                verbose=False,
            )

    def run(self, tx_features: dict, raw_tx: dict, tx_text: str):
        # 1) ML model
        model_out = self.model_runner.run(tx_features)

        # 2) Deterministic rules
        diagnostics_out = self.diagnostics.run(raw_tx)

        # 3) Retrieval
        docs = self.retriever.invoke(tx_text)
        docs_text = "\n".join(
            f"[{d.metadata.get('doc_id', 'UNKNOWN')}] {d.page_content}"
            for d in docs
        )

        # 4) Prompt
        prompt = self.prompt.format(
            model_output=model_out,
            diagnostics=diagnostics_out,
            docs=docs_text,
        )

        # 5) If no LLM, return structured text without generative reasoning
        if self.llm is None:
            return (
                "LLM_DISABLED\n"
                f"MODEL: {model_out}\n"
                f"DIAGNOSTICS: {diagnostics_out}\n"
                f"RETRIEVED_DOCS:\n{docs_text}\n"
            )

        # 6) LLM reasoning
        return self.llm.invoke(prompt)
