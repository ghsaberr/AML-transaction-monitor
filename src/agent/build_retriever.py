from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import json
from pathlib import Path

KB_DIR = Path("data/knowledge_base")

docs = []
metadatas = []

for file in KB_DIR.glob("*.jsonl"):
    with open(file) as f:
        for line in f:
            obj = json.loads(line)
            docs.append(obj["text"])
            metadatas.append({"doc_id": obj["doc_id"]})

emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_texts(docs, emb, metadatas=metadatas)
vectorstore.save_local("data/vectorstore/faiss")

print("Retriever built")
