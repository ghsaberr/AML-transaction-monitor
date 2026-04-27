#!/usr/bin/env python
"""
Setup script to build FAISS vectorstore from knowledge base.
Run once to create the index for RAG retrieval.

Usage:
    python -m src.agent.setup_rag
    python -m uv run src/agent/setup_rag.py
"""

import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_vectorstore():
    """Build FAISS vectorstore from knowledge base."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
    except ImportError as e:
        logger.error(f"Missing dependencies: {e}")
        logger.info("Install with: pip install langchain-huggingface sentence-transformers faiss-cpu")
        return False

    KB_DIR = Path("data/knowledge_base")
    VECTORSTORE_DIR = Path("data/vectorstore/faiss")

    # Create vectorstore dir
    VECTORSTORE_DIR.parent.mkdir(parents=True, exist_ok=True)

    # Load documents from knowledge base
    docs = []
    metadatas = []

    logger.info(f"Loading documents from {KB_DIR}")
    for file in KB_DIR.glob("*.jsonl"):
        logger.info(f"  Processing {file.name}")
        with open(file) as f:
            for i, line in enumerate(f):
                try:
                    obj = json.loads(line)
                    docs.append(obj.get("text", ""))
                    metadatas.append({
                        "doc_id": obj.get("doc_id", f"doc_{i}"),
                        "source": file.name,
                    })
                except json.JSONDecodeError as e:
                    logger.warning(f"  Skipping invalid JSON in line {i + 1}: {e}")

    if not docs:
        logger.error("No documents found in knowledge base!")
        return False

    logger.info(f"Loaded {len(docs)} documents")

    # Initialize embeddings
    logger.info("Initializing embeddings (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create and save vectorstore
    logger.info("Creating FAISS vectorstore...")
    vectorstore = FAISS.from_texts(
        texts=docs,
        embedding=embeddings,
        metadatas=metadatas,
    )

    logger.info(f"Saving vectorstore to {VECTORSTORE_DIR}")
    vectorstore.save_local(str(VECTORSTORE_DIR))

    logger.info("✓ FAISS vectorstore built successfully!")
    logger.info(f"  Vectorstore: {VECTORSTORE_DIR}")
    logger.info(f"  Documents: {len(docs)}")
    return True


def verify_vectorstore():
    """Verify vectorstore can be loaded."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
    except ImportError:
        return False

    VECTORSTORE_DIR = Path("data/vectorstore/faiss")

    if not VECTORSTORE_DIR.exists():
        logger.warning(f"Vectorstore not found at {VECTORSTORE_DIR}")
        return False

    try:
        logger.info("Verifying vectorstore...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = FAISS.load_local(
            str(VECTORSTORE_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info(f"✓ Vectorstore loaded successfully")
        logger.info(f"  Vectors available for retrieval")
        return True
    except Exception as e:
        logger.error(f"Failed to load vectorstore: {e}")
        return False


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("FAISS Vectorstore Setup for AML RAG")
    logger.info("=" * 60)
    
    success = build_vectorstore()
    
    if success:
        verify_vectorstore()
        logger.info("\n✓ Setup complete! The /explain endpoint can now use RAG retrieval.")
    else:
        logger.error("\n✗ Setup failed. Please check the logs above.")
        exit(1)
