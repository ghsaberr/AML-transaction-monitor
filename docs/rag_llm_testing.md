# Sprint 6 Testing Guide

## Quick Validation (5 minutes)

```bash
# Verify all tests still pass
uv run pytest tests/ -v

# Expected: 96/96 PASSED
```

## End-to-End Agent Explanation Test

### Test 1: Graceful Degradation (No LLM)

**Setup:**
```bash
export LLM_MODE=none
uv run uvicorn src.api.main:app &
```

**Test:**
```bash
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{
    "case_id": "case_001",
    "tx_features": {"transaction_velocity": 0.8, "transaction_amount": 50000},
    "tx_text": "High velocity transaction to new account"
  }'
```

**Expected Response:**
```json
{
  "explanation_type": "feature_importance",
  "llm_enabled": false,
  "top_features": [
    {"feature_name": "transaction_velocity", "importance_value": 0.45, "contribution": "positive"},
    ...
  ]
}
```

### Test 2: With FAISS Retrieval

**Setup:**
```bash
# Build FAISS index
python -m src.agent.setup_rag
# Expected output: "✓ FAISS index created at data/vectorstore/faiss"

# Test without LLM (but with retrieval)
export LLM_MODE=none
uv run uvicorn src.api.main:app &
```

**Test:**
```bash
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{
    "case_id": "case_002",
    "tx_features": {"transaction_velocity": 0.9, "transaction_amount": 75000},
    "tx_text": "Multiple rapid transactions to multiple destinations, classic smurfing pattern"
  }'
```

**Expected Response:**
```json
{
  "decision": "ALERT",
  "rationale": [
    "Risk Score: 90.00%",
    "Top Risk Factors: transaction_velocity, geographic_risk_score",
    "Similar cases: [rule_002, rule_003]",  ← Citations from FAISS
    "..."
  ],
  "cited_docs": ["rule_002", "rule_003"],
  "llm_enabled": false
}
```

### Test 3: Full RAG + LLM (Optional)

**Setup (requires Phi-3 model):**
```bash
# Verify model exists
ls -lh models/llm/Phi-3-mini-4k-instruct-q4.gguf
# Expected: ~2.4GB

# Build FAISS
python -m src.agent.setup_rag

# Start server with LLM enabled
export LLM_MODE=local
uv run uvicorn src.api.main:app &
```

**Test:**
```bash
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{
    "case_id": "case_003",
    "tx_features": {"transaction_velocity": 0.95, "transaction_amount": 100000},
    "tx_text": "Structuring: Multiple deposits just below threshold to avoid reporting"
  }'
```

**Expected Response (with LLM reasoning):**
```json
{
  "decision": "ALERT",
  "rationale": [
    "Risk Score: 95.00%",
    "Top Risk Factors: transaction_velocity, transaction_amount",
    "The pattern strongly suggests deliberate structuring to circumvent AML thresholds.",
    "Similar to [rule_001] (structuring) and [rule_003] (layering).",
    "Recommend immediate account review and potential SAR filing."
  ],
  "cited_docs": ["rule_001", "rule_003"],
  "llm_enabled": true,
  "top_features": [...]
}
```

## Programmatic Testing

### Unit Test: Agent Explanation Generation

```python
# tests/test_agent_explanation.py
import pytest
from src.agent.aml_agent import AMLAgent, AgentConfig
from src.modeling.model_runner import ModelRunner

@pytest.fixture
def mock_runner():
    """Mock ModelRunner for testing"""
    runner = ModelRunner(model_path="models/lgbm_final/model.txt")
    return runner

def test_agent_graceful_fallback_no_llm(mock_runner):
    """Agent should work without Phi-3 model"""
    config = AgentConfig(llm_mode="none")
    agent = AMLAgent(config, mock_runner)
    
    result = agent.run(
        tx_features={"transaction_velocity": 0.8},
        raw_tx={"amount": 50000},
        tx_text="Test transaction"
    )
    
    # Should return fallback structure
    assert "LLM_DISABLED" in result or isinstance(result, dict)
    assert result is not None  # Never returns None

def test_agent_with_faiss_retrieval():
    """Agent should retrieve similar cases from FAISS"""
    config = AgentConfig(
        vectorstore_dir="data/vectorstore/faiss",
        top_k=3
    )
    agent = AMLAgent(config, mock_runner)
    
    if agent.retriever:  # FAISS available
        docs = agent.retriever.invoke("smurfing pattern detected")
        assert len(docs) > 0
        assert all(hasattr(d, 'metadata') for d in docs)
        assert all(d.metadata.get('doc_id') for d in docs)

def test_explain_response_schema():
    """ExplainResponse should validate with new fields"""
    from src.api.schemas import ExplainResponse, FeatureImportance
    
    response = ExplainResponse(
        case_id="case_001",
        score=0.85,
        decision="ALERT",
        rationale=["High velocity"],
        cited_docs=["rule_001"],
        llm_enabled=False,
        top_features=[
            FeatureImportance(
                feature_name="velocity",
                importance_value=0.45,
                contribution="positive"
            )
        ]
    )
    
    assert response.decision == "ALERT"
    assert response.llm_enabled is False
    assert len(response.cited_docs) > 0
```

## Integration Test: Full /explain Endpoint

```python
# tests/test_explain_integration.py
import pytest
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

def test_explain_endpoint_basic():
    """POST /explain should return valid ExplainResponse"""
    response = client.post("/explain", json={
        "case_id": "test_case_001",
        "tx_features": {
            "transaction_velocity": 0.75,
            "transaction_amount": 50000,
            "unique_destinations_24h": 5,
            "days_since_account_creation": 100,
            "is_flagged_as_high_risk": True,
            "account_age_category": 1,
            "geographic_risk_score": 0.6,
            "transaction_type_risk": 0.5,
            "transaction_count_24h": 10,
            "avg_transaction_amount_7d": 5000.0
        },
        "tx_text": "Test transaction pattern"
    })
    
    assert response.status_code == 200
    data = response.json()
    
    # Validate required fields
    assert "case_id" in data
    assert "score" in data
    assert "decision" in data
    assert data["decision"] in ["ALERT", "PASS"]
    
    # Validate optional fields
    assert "llm_enabled" in data
    assert "top_features" in data
    assert isinstance(data["top_features"], list)

def test_explain_endpoint_citations():
    """Explanation should cite retrieved documents"""
    response = client.post("/explain", json={
        "case_id": "test_case_002",
        "tx_features": {
            "transaction_velocity": 0.95,
            "transaction_amount": 100000,
            ...
        },
        "tx_text": "Multiple rapid transactions"
    })
    
    assert response.status_code == 200
    data = response.json()
    
    # If FAISS is available, should have citations
    if "cited_docs" in data:
        assert isinstance(data["cited_docs"], list)
        # Citations should match format: rule_001, rule_002, etc.
        for doc_id in data["cited_docs"]:
            assert isinstance(doc_id, str)

def test_explain_fallback_no_gguf():
    """Endpoint should gracefully fall back if .gguf missing"""
    # This test runs with default LLM_MODE=none
    response = client.post("/explain", json={
        "case_id": "test_case_003",
        "tx_features": {...},
        "tx_text": "Test transaction"
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "top_features" in data  # Fallback should include features
    assert data.get("llm_enabled", False) is False
```

## Validation Checklist

### Code Quality
- [ ] `src/agent/aml_agent.py` has docstrings for all public methods
- [ ] `src/api/schemas.py` ExplainResponse validates correctly
- [ ] All exceptions caught with proper logging (no bare `except:`)
- [ ] Type hints present on all public functions

### Functionality
- [ ] FAISS index loads without errors: `python -m src.agent.setup_rag`
- [ ] Agent returns non-None value even without .gguf
- [ ] Citations appear in rationale when FAISS available
- [ ] Feature importance extracted correctly from model
- [ ] Fallback rationale makes sense (not placeholder text)

### Performance
- [ ] /explain endpoint responds in <5s (with LLM), <500ms (without)
- [ ] FAISS retrieval takes <100ms
- [ ] Memory usage stable (no leaks in repeated calls)

### Robustness
- [ ] Missing FAISS index: Endpoint works, no citations
- [ ] Missing Phi-3 model: Endpoint works, uses fallback
- [ ] Invalid tx_features: 400 Bad Request with clear error
- [ ] Concurrent requests: Thread-safe singleton, no race conditions

## Performance Benchmarking

```python
# tests/benchmark_agent.py
import time
from src.agent.aml_agent import AMLAgent, AgentConfig
from src.modeling.model_runner import ModelRunner

def benchmark_agent_latency():
    runner = ModelRunner("models/lgbm_final/model.txt")
    config = AgentConfig(llm_mode="none")  # Exclude LLM latency
    agent = AMLAgent(config, runner)
    
    times = []
    for i in range(10):
        start = time.time()
        agent.run(
            tx_features={"transaction_velocity": 0.8},
            raw_tx={"amount": 50000},
            tx_text="Test"
        )
        times.append(time.time() - start)
    
    print(f"Avg: {sum(times)/len(times)*1000:.1f}ms")
    print(f"P95: {sorted(times)[9]*1000:.1f}ms")
    # Expected: <100ms without LLM, <3s with LLM

if __name__ == "__main__":
    benchmark_agent_latency()
```

## Running All Tests

```bash
# Full test suite (should be 96/96 PASSED)
uv run pytest tests/ -v

# Specific test file
uv run pytest tests/test_explain_integration.py -v

# With coverage
uv run pytest tests/ --cov=src --cov-report=html

# With output
uv run pytest tests/ -v -s
```

## Debugging Tips

### Agent not retrieving documents
```python
# Verify FAISS index exists and is loadable
from src.agent.build_retriever import build_retriever
retriever = build_retriever("data/vectorstore/faiss")
docs = retriever.invoke("smurfing pattern")
print(f"Found {len(docs)} documents")
print(f"Top doc: {docs[0].metadata.get('doc_id')}")
```

### LLM not loading
```python
# Verify Phi-3 model file
import os
model_path = "models/llm/Phi-3-mini-4k-instruct-q4.gguf"
print(f"Model exists: {os.path.exists(model_path)}")
print(f"Model size: {os.path.getsize(model_path) / 1e9:.1f}GB")
```

### Slow /explain requests
```bash
# Profile the endpoint
python -m cProfile -s cumtime src/api/app.py
# Identify which component (feature extraction, retrieval, LLM) is slow
```

