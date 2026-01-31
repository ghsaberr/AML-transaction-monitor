from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


class AMLAgent:
    def __init__(self, model_runner, diagnostics):
        self.model_runner = model_runner
        self.diagnostics = diagnostics

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = FAISS.load_local(
            "data/vectorstore/faiss",
            embeddings,
            allow_dangerous_deserialization=True
        )

        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        self.llm = LlamaCpp(
            model_path="models/llm/Phi-3-mini-4k-instruct-q4.gguf",
            temperature=0.0,
            n_ctx=2048,
            verbose=False
        )

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

    def run(self, tx_features: dict, raw_tx: dict, tx_text: str):
        # 1. ML model
        model_out = self.model_runner.run(tx_features)

        # 2. Deterministic rules
        diagnostics_out = self.diagnostics.run(raw_tx)

        # 3. Retrieval (FIX HERE 👇)
        docs = self.retriever.invoke(tx_text)

        docs_text = "\n".join(
            f"[{d.metadata.get('doc_id', 'UNKNOWN')}] {d.page_content}"
            for d in docs
        )

        # 4. Prompt
        prompt = self.prompt.format(
            model_output=model_out,
            diagnostics=diagnostics_out,
            docs=docs_text
        )

        # 5. LLM reasoning
        return self.llm.invoke(prompt)
