# Architecture Overview: AML Transaction Monitoring System

## Executive Summary

This document describes the production-grade architecture of the AML Transaction Monitoring System—a risk decision engine that scores transactions for anti-money laundering (AML) compliance, routes cases for human review, maintains audit trails, and monitors model performance and feature drift in real-time.

**Design Principle**: Contract-first, service-oriented architecture with strict input/output validation, immutable audit trails, and integrated monitoring for governance compliance.

---

## System Architecture

### High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                 Production Transaction Stream                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │  API Layer (FastAPI Routers)       │
        │  - /score (POST)                   │
        │  - /review (POST)                  │
        │  - /audit (GET)                    │
        │  - /metrics (GET)                  │
        └────────────────────┬───────────────┘
                             │
                             ▼
        ┌────────────────────────────────────┐
        │   Pydantic Validation Layer        │
        │   (Request/Response Contracts)     │
        └────────────────────┬───────────────┘
                             │
                             ▼
        ┌────────────────────────────────────┐
        │   Feature Contract Enforcement     │
        │   - Schema validation              │
        │   - Type coercion                  │
        │   - Null handling                  │
        │   - Deterministic hashing          │
        └────────────────────┬───────────────┘
                             │
                             ▼
        ┌────────────────────────────────────┐
        │    Scoring Service                 │
        │    - ModelRunner (LightGBM)        │
        │    - Feature ordering              │
        │    - Score generation [0,1]        │
        │    - Decision logic (ALERT/PASS)   │
        └────────────────────┬───────────────┘
                             │
        ┌────────────────────┴────────────────┐
        │                                     │
        ▼                                     ▼
   ┌─────────────┐              ┌───────────────────────┐
   │  Storage    │              │  Metrics Collector    │
   │  (SQLite)   │              │  - Latency tracking   │
   │  - Cases    │              │  - Throughput         │
   │  - Reviews  │              │  - Score statistics   │
   │  - Audit    │              └───────────────────────┘
   └──────┬──────┘                         │
          │                                ▼
          │                  ┌──────────────────────────────┐
          │                  │   Monitoring & Alerting      │
          │                  │   - Drift detection (KS test)│
          │                  │   - Performance metrics      │
          │                  │   - Alerting policies        │
          │                  └──────────────────────────────┘
          │
          ▼
   ┌──────────────────┐
   │  Audit Trail     │
   │  (Immutable)     │
   │  Event Log       │
   └──────────────────┘
```

---

## Layered Architecture

### Layer 1: API Endpoints (FastAPI)
**Purpose**: HTTP request/response handling

**Components**:
- `src/api/routers/score.py`: `/score` POST endpoint
- `src/api/routers/review.py`: `/review` POST endpoint
- `src/api/routers/audit.py`: `/audit` GET endpoints
- `src/api/routers/metrics.py`: `/metrics` GET endpoint
- `src/api/routers/health.py`: `/health` GET endpoint

**Responsibility**: Parse HTTP requests, delegate to services, return HTTP responses

---

### Layer 2: Pydantic Contracts
**Purpose**: Strict validation of all external inputs/outputs

**Components**:
- `src/api/schemas.py`: 25+ validation schemas
  - `ScoreRequest/Response`
  - `ReviewRequest/Response`
  - `AuditTrailResponse`
  - `MetricsSnapshot`
  - `MonitoringStatusResponse`
  - Feature/drift/performance/alert schemas

**Responsibility**: Enforce API contracts, raise validation errors before processing

**Key Principle**: All external data must pass Pydantic validation before touching business logic

---

### Layer 3: Feature Contract Enforcement
**Purpose**: Guarantee feature schema consistency between training and production

**Components**:
- `src/features/feature_contract.py`: FeatureContract system
  - 35-feature schema definition
  - Type coercion (e.g., "5.7" → 5 for int)
  - Null handling (disallow, zero, mean, default)
  - Deterministic SHA256 hashing for schema versioning
  - Training metadata (means, mins, maxes) for validation

**Responsibility**:
1. Validate feature names match schema
2. Enforce correct dtypes (FLOAT, INT, BOOL, STRING)
3. Handle missing/null values per feature policy
4. Compute schema hash for model lineage tracking
5. Reject transactions with contract violations

**Key Principle**: No feature reaches the model without strict contract validation

---

### Layer 4: Scoring Service
**Purpose**: Decoupled business logic for transaction scoring

**Components**:
- `src/api/service.py`: ScoringService class
  - Lazy imports to prevent circular dependencies
  - Transaction scoring with feature contract integration
  - Case record creation with full metadata
  - Returns score, decision, and version information

**Process**:
1. Accept raw transaction features from router
2. Pass to ModelRunner with feature contract validation
3. Receive score + decision
4. Create case record with metadata (model version, threshold version, schema hash)
5. Log audit event
6. Return response

**Key Principle**: Services encapsulate domain logic, independent of HTTP framework

---

### Layer 5: Model Management
**Purpose**: Load, version, and lineage tracking for ML models

**Components**:
- `src/agent/model_runner.py`: ModelRunner class
  - Loads LightGBM model from `models/lgbm_final/model.txt`
  - Loads feature contract from `models/lgbm_final/feature_contract.json`
  - Loads artifact metadata from `models/lgbm_final/artifact_metadata.json`
  - Validates features against contract
  - Generates predictions [0,1]
  - Returns metadata (version, schema_hash, threshold_version)

**Artifact Metadata Tracked**:
- Model version (e.g., "1.0.0")
- Training window (dates, sample count)
- Threshold rationale (value, precision at threshold)
- MLflow tracking (experiment ID, run ID)
- Performance baseline (AUC, precision)

**Key Principle**: Model as versioned artifact with complete lineage for governance

---

### Layer 6: Storage Layer (SQLite)
**Purpose**: Persistent state management with immutable audit trails

**Components**:
- `src/storage/db.py`: Database abstraction
  - 3 tables: cases, audit_events, review_history
  - Thread-safe connections via thread-local storage
  - Append-only audit events (immutable)
  - Status transitions tracked via review_history

**Database Schema**:

```sql
-- Cases table (scored transactions)
CREATE TABLE cases (
    case_id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL,
    model_version TEXT,
    threshold_version TEXT,
    feature_contract_version TEXT,
    score REAL,
    decision TEXT,
    raw_features JSONB,
    status TEXT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

-- Audit events (immutable log)
CREATE TABLE audit_events (
    event_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL,
    event_type TEXT,
    actor TEXT,
    details JSONB,
    timestamp TIMESTAMP,
    FOREIGN KEY (case_id) REFERENCES cases(case_id)
);

-- Review history (decisions)
CREATE TABLE review_history (
    review_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL,
    reviewer_id TEXT,
    decision TEXT,
    note TEXT,
    previous_status TEXT,
    new_status TEXT,
    timestamp TIMESTAMP,
    FOREIGN KEY (case_id) REFERENCES cases(case_id)
);
```

**Key Principle**: Immutable append-only events for compliance audit trails

---

### Layer 7: Monitoring & Observability
**Purpose**: Real-time detection of model performance degradation and feature drift

**Components**:
- `src/monitoring/metrics.py`: Four monitoring classes
  - `FeatureDriftDetector`: KS-test for distribution shift, mean drift analysis
  - `ModelPerformanceCalculator`: Precision/recall/F1 from review audit trail
  - `MetricsCollector`: Event buffering, latency percentiles, aggregation
  - `AlertingPolicy`: Configurable thresholds for all alert conditions

**Monitoring Pipeline**:
```
Scoring Events → MetricsCollector → FeatureDriftDetector
                                   → ModelPerformanceCalculator
                                   → AlertingPolicy
                                   → GET /metrics endpoint
```

**Key Metrics Tracked**:
- Latency: avg, p95, p99 (milliseconds)
- Throughput: requests/sec, cases processed
- Scores: mean, median, std, min, max
- Drift: KS statistic, p-value, mean shift (std devs)
- Performance: precision, recall, F1, AUC, confusion matrix
- Alerts: drift, latency, performance, queue backlog

**Key Principle**: Proactive monitoring enables early detection of model degradation

---

### Layer 8: Agentic Explainability Layer (RAG)
**Purpose**: Provide human-readable explanations with document citations using retrieval-augmented generation and local LLM inference

**Components**:
- `src/agent/aml_agent.py`: AMLAgent class
  - Loads FAISS vector store with historical AML cases
  - Initializes Phi-3 Mini LLM (optional, with graceful degradation)
  - Executes RAG pipeline: retrieval → prompt → LLM inference
  - Returns structured explanations with document citations

- `src/agent/build_retriever.py`: FAISS retriever builder
  - Loads vector store from `data/vectorstore/faiss/`
  - Uses sentence-transformers (all-MiniLM-L6-v2) for embeddings
  - Returns top-K similar cases with metadata (doc_id)

- `src/agent/setup_rag.py`: FAISS index initialization script
  - Reads knowledge base from `data/knowledge_base/sample_aml_rules.jsonl`
  - Embeds documents and creates FAISS index
  - Idempotent: can be re-run to refresh index

**Data Flow: Feature Extraction → FAISS Retrieval → LangChain Orchestration → Phi-3 Inference**:
```
POST /explain { tx_features, tx_text }
    │
    ├─→ ExplanationService.explain_score()
    │       │
    │       ├─→ AMLAgent.run()
    │       │       │
    │       │       ├─→ ModelRunner.run() [LightGBM scoring]
    │       │       │       └─→ Returns: score, decision, top_features
    │       │       │
    │       │       ├─→ Retriever.invoke(tx_text) [FAISS]
    │       │       │       └─→ Returns: top-K docs with metadata.doc_id
    │       │       │
    │       │       ├─→ PromptTemplate.format() [LangChain]
    │       │       │       └─→ Injects: features, retrieved_docs, citations
    │       │       │
    │       │       └─→ LlamaCpp() [Phi-3 inference]
    │       │               └─→ Returns: natural language rationale
    │       │
    │       └─→ Return: { decision, rationale, cited_docs, llm_enabled }
    │
    └─→ Output: ExplainResponse { decision, rationale, cited_docs, ... }
```

**Graceful Degradation Strategy**:
| Scenario | Behavior | Output |
|----------|----------|--------|
| Phi-3 GGUF missing | Agent initializes with `llm=None` | Feature importance only, `llm_enabled: false` |
| FAISS index missing | Retriever returns empty list | No citations, feature importance only |
| Both missing | Full fallback to feature importance | Standard feature importance response |
| LLM_MODE=none | Agent skips LLM initialization | RAG retrieval without LLM reasoning |

**Knowledge Base**:
- Location: `data/knowledge_base/sample_aml_rules.jsonl`
- Document format: JSONL with `doc_id` (e.g., "rule_001") and `text` fields
- Current content: 4 AML typologies (structuring, smurfing, layering, integration)
- Extensible: Add more documented cases, re-run `setup_rag.py` to re-index

**Citation Format**:
- Retrieved documents cited as `[rule_XXX]` in LLM rationale
- Example: "Pattern suggests structuring [rule_001] with layering [rule_003]"
- Reviewers can look up cited documents in knowledge base for full details

**Key Principle**: Explainable AI with traceable citations for regulatory compliance

---

## Data Flow Examples

### Example 1: Score Transaction

```
Input: POST /score { tx_features }
    │
    ├─→ Pydantic: Validate ScoreRequest
    │
    ├─→ Service: ScoringService.score_and_create_case()
    │       ├─→ ModelRunner.validate_features() [FeatureContract]
    │       ├─→ ModelRunner.run() [LightGBM scoring]
    │       ├─→ Storage: create_case() [insert into DB]
    │       └─→ MetricsCollector: record_scoring_event() [latency tracking]
    │
    ├─→ Pydantic: Validate ScoreResponse
    │
    └─→ Output: { case_id, score, decision, model_version, ... }
```

### Example 2: Review Case

```
Input: POST /review { case_id, decision, reviewer_id, note }
    │
    ├─→ Pydantic: Validate ReviewRequest
    │
    ├─→ Service: ReviewService.submit_review()
    │       ├─→ Storage: update_case_status()
    │       ├─→ Storage: record_review() [create review record]
    │       └─→ Storage: log_audit_event() [immutable event]
    │
    ├─→ Pydantic: Validate ReviewResponse
    │
    └─→ Output: { review_id, status_transition, timestamp, ... }
```

### Example 3: Get Metrics

```
Input: GET /metrics
    │
    ├─→ Service: MetricsService.get_metrics()
    │       ├─→ MetricsCollector: get_current_metrics() [latency, throughput]
    │       ├─→ FeatureDriftDetector: compute_ks_statistic() [recent vs training]
    │       ├─→ ModelPerformanceCalculator: calculate_from_reviews() [recent reviews]
    │       ├─→ AlertingPolicy: evaluate_alerts() [check thresholds]
    │       └─→ Storage: get_case_count_by_status() [queue stats]
    │
    ├─→ Pydantic: Validate MonitoringStatusResponse
    │
    └─→ Output: { metrics, drift, performance, alerts, ... }
```

---

## Design Patterns

### 1. Contract-First Design
- All external data validated against Pydantic schemas
- Feature contract enforces ML schema consistency
- Versioning enables safe model/schema evolution

### 2. Service Layer Abstraction
- Business logic decoupled from HTTP framework
- Services are stateless, testable, reusable
- Lazy imports prevent circular dependencies

### 3. Immutable Audit Trail
- Append-only event log (no updates to audit_events)
- Status transitions tracked via separate review_history table
- Full compliance audit trail for regulatory review

### 4. Versioning for Governance
- Model version tracked per case
- Feature contract version tracked per case
- Threshold version tracked per case
- Schema hash enables reproducibility verification

### 5. Thread-Safe Storage
- Thread-local database connections
- SQLite with journal mode for concurrent access
- Safe for multi-threaded FastAPI workers

### 6. Lazy Dependency Injection
- Services import dependencies on-demand, not at module load
- Prevents circular import issues during app initialization
- Enables testing with mock dependencies

---

## Deployment Architecture

### Docker Container Structure
```
┌──────────────────────────────────────┐
│   Docker Image (Python 3.11)         │
├──────────────────────────────────────┤
│ /app                                 │
│  ├─ src/                             │
│  ├─ models/                          │
│  ├─ configs/                         │
│  ├─ requirements.txt                 │
│  └─ src/api/main.py                  │
├──────────────────────────────────────┤
│ Exposed Port: 8000                   │
│ Health Check: GET /health            │
│ Working Directory: /app              │
└──────────────────────────────────────┘
```

### Volume Mounts (Production)
- `/app/data/`: SQLite database (persistent)
- `/app/configs/`: Configuration files (read-only)
- `/app/models/`: LightGBM model, feature contract, artifact metadata

---

## Scalability & Performance

### Design for Scale

1. **Stateless Services**: Horizontal scaling via multiple FastAPI workers
2. **SQLite with WAL**: Write-Ahead Logging for concurrent access
3. **Bounded Buffers**: MetricsCollector limits memory (max 1000 events)
4. **Efficient Queries**: Indexed by case_id, status, created_at
5. **Feature Contract Caching**: Loaded once at startup, reused

### Performance Characteristics

- **Scoring Latency**: ~100-200ms (feature validation + model inference)
- **Review Processing**: ~50ms (database update)
- **Metrics Computation**: ~500ms (percentile calculations, drift analysis)
- **Database Throughput**: 1000+ transactions/sec with proper indexing

### Bottleneck Mitigation

- **Feature Validation**: Pydantic compiled for speed
- **Model Inference**: LightGBM is optimized C++ backend
- **Drift Computation**: Windowed analysis (last 24h data)
- **Database**: SQLite suitable for <10K transactions/day; migrate to PostgreSQL for 100K+/day

---

## Security & Compliance

### Data Protection
- **Audit Trail**: Immutable event log for regulatory compliance
- **Versioning**: All decisions traceable to model/schema versions
- **Lineage**: Artifact metadata enables reproducibility
- **Access Control**: Ready for API authentication layer (beyond scope)

### Input Validation
- **Pydantic Strict Mode**: Reject invalid types, extra fields
- **Feature Contract**: Prevents data drift attacks
- **Type Coercion**: Controlled conversions prevent injection

### Output Integrity
- **Deterministic Hashing**: Schema hash enables verification
- **Versioned Models**: No silent model changes
- **Comprehensive Logging**: All decisions recorded with actor

---

## Testing Strategy

### Test Coverage (87+ tests)
1. **Unit Tests**: Storage, services, schemas, monitoring (67 tests)
2. **Integration Tests**: End-to-end workflows (20+ tests)
3. **Contract Tests**: Feature validation, schema enforcement (25 tests)
4. **Monitoring Tests**: Drift, performance, alerts (20 tests)

### Test Categories
- Storage layer: CRUD, audit trail, status transitions
- API contracts: Request/response validation
- Feature contracts: Type coercion, null handling, versioning
- Monitoring: Drift detection, performance calculation, alerting
- Integration: Score → Review → Audit → Metrics workflows

---

## Future Enhancements

1. **PostgreSQL Migration**: For high-throughput deployments (>100K tx/day)
2. **Feature Store Integration**: MLflow/Feast for feature management
3. **A/B Testing Framework**: Shadow model evaluation
4. **Explainability**: SHAP values for decision explanation
5. **API Authentication**: OAuth2 / API keys
6. **Rate Limiting**: Per-endpoint quotas
7. **Advanced Alerting**: Slack/PagerDuty integration
8. **Dashboard**: Real-time metrics visualization (Grafana)

---

## References

- **FastAPI**: https://fastapi.tiangolo.com/
- **Pydantic**: https://docs.pydantic.dev/
- **LightGBM**: https://lightgbm.readthedocs.io/
- **SQLite**: https://www.sqlite.org/
- **Pytest**: https://pytest.org/
