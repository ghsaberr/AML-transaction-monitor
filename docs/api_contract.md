# API Contract Reference

## Overview

This document defines the formal HTTP API contracts for the AML Transaction Monitoring System. All requests must conform to the specified schemas. Responses are guaranteed to match their schemas.

**Base URL**: `http://localhost:8000` (development)

**Content-Type**: `application/json`

**Authentication**: Currently unauthenticated (add OAuth2 in production)

---

## Error Handling

### Error Response Schema

```json
{
  "detail": "Human-readable error message",
  "status_code": 400,
  "error_type": "VALIDATION_ERROR" | "NOT_FOUND" | "INVALID_STATUS" | "INTERNAL_ERROR"
}
```

### Common Status Codes

| Code | Meaning | Example |
|------|---------|---------|
| 200 | Success | All GET requests, successful POST |
| 400 | Bad Request | Invalid feature types, missing fields |
| 404 | Not Found | Case ID does not exist |
| 422 | Validation Error | Pydantic schema violation |
| 500 | Internal Error | Database connection failure |

---

## 1. Health Check

### GET /health

**Purpose**: Verify system health, model availability, database connectivity

**Request**: No body required

**Response**:
```json
{
  "status": "healthy" | "degraded",
  "version": "1.0.0",
  "database_healthy": true | false,
  "model_loaded": true | false,
  "timestamp": "2026-04-27T15:30:45Z"
}
```

**Example**:
```bash
curl http://localhost:8000/health
```

**Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "database_healthy": true,
  "model_loaded": true,
  "timestamp": "2026-04-27T15:30:45Z"
}
```

---

## 2. Score Transaction

### POST /score

**Purpose**: Score a transaction for AML risk and create a review case

**Request Body**:
```json
{
  "tx_features": {
    "transaction_amount": 5000.0,
    "transaction_count_24h": 15,
    "unique_destinations_24h": 3,
    "avg_transaction_amount_7d": 2500.0,
    "days_since_account_creation": 365,
    "is_flagged_as_high_risk": false,
    "account_age_category": 2,
    "transaction_velocity": 0.1,
    "geographic_risk_score": 0.3,
    "transaction_type_risk": 0.2
  }
}
```

**Request Schema Validation**:
- `tx_features`: Required, dict of feature values
- Features must include all 35 schema fields (see [Feature Contract](#feature-contract-specification))
- Numeric types automatically coerced (e.g., "5.7" → 5 for int features)

**Response**:
```json
{
  "case_id": "case_550e8400e29b",
  "request_id": "req_abc123",
  "score": 0.25,
  "decision": "PASS" | "ALERT",
  "review_flag": false,
  "threshold_used": 0.5,
  "model_version": "1.0.0",
  "threshold_version": "1.0.0",
  "feature_contract_version": "1.0.0",
  "schema_hash": "20401fb2ad1d0f0f",
  "timestamp": "2026-04-27T15:30:45Z"
}
```

**Response Schema Details**:
- `case_id`: Unique transaction case identifier (UUID)
- `score`: Risk score ∈ [0, 1] where 1 = highest risk
- `decision`: "ALERT" if score ≥ threshold, else "PASS"
- `model_version`: Model version used (e.g., "1.0.0")
- `threshold_version`: Threshold version used (e.g., "1.0.0")
- `feature_contract_version`: Feature schema version (e.g., "1.0.0")
- `schema_hash`: SHA256 hash of feature contract for verification

**Example**:
```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "tx_features": {
      "transaction_amount": 5000.0,
      "transaction_count_24h": 15,
      "unique_destinations_24h": 3,
      "avg_transaction_amount_7d": 2500.0,
      "days_since_account_creation": 365,
      "is_flagged_as_high_risk": false,
      "account_age_category": 2,
      "transaction_velocity": 0.1,
      "geographic_risk_score": 0.3,
      "transaction_type_risk": 0.2
    }
  }'
```

**Error Cases**:
```json
{
  "detail": "tx_features: field required"
}
```

---

## 3. Submit Review

### POST /review

**Purpose**: Submit a human review decision for a scored case

**Request Body**:
```json
{
  "case_id": "case_550e8400e29b",
  "reviewer_id": "reviewer_001",
  "decision": "APPROVED" | "REJECTED" | "ESCALATED",
  "note": "Optional detailed comment about the decision"
}
```

**Request Schema Validation**:
- `case_id`: Required, must exist in database
- `reviewer_id`: Required, string identifier for reviewer
- `decision`: Required, one of APPROVED / REJECTED / ESCALATED
- `note`: Optional, string (max 500 chars)

**Response**:
```json
{
  "review_id": "review_abc123",
  "case_id": "case_550e8400e29b",
  "reviewer_id": "reviewer_001",
  "decision": "APPROVED",
  "previous_status": "QUEUED",
  "new_status": "APPROVED",
  "note": "Transaction verified as legitimate",
  "timestamp": "2026-04-27T15:30:45Z"
}
```

**Response Schema Details**:
- `review_id`: Unique review identifier (UUID)
- `previous_status`: Case status before review
- `new_status`: Case status after review (matches decision)
- Status mapping: APPROVED → APPROVED, REJECTED → REJECTED, ESCALATED → ESCALATED

**State Machine**:
```
QUEUED → APPROVED | REJECTED | ESCALATED
         ↓         ↓          ↓
       APPROVED  REJECTED  ESCALATED
       (final)    (final)    (final)
```

**Example**:
```bash
curl -X POST http://localhost:8000/review \
  -H "Content-Type: application/json" \
  -d '{
    "case_id": "case_550e8400e29b",
    "reviewer_id": "reviewer_001",
    "decision": "APPROVED",
    "note": "Verified legitimate transaction"
  }'
```

**Error Cases**:
```json
{
  "detail": "Case not found: case_invalid"
}
```

---

## 4. Get Audit Trail

### GET /audit/{case_id}

**Purpose**: Retrieve complete audit trail (immutable event log) for a case

**Path Parameters**:
- `case_id`: Case identifier

**Query Parameters**: None

**Response**:
```json
{
  "case_id": "case_550e8400e29b",
  "events": [
    {
      "event_id": "evt_001",
      "event_type": "CASE_CREATED",
      "actor": "system",
      "details": {
        "score": 0.25,
        "decision": "PASS"
      },
      "timestamp": "2026-04-27T15:30:00Z"
    },
    {
      "event_id": "evt_002",
      "event_type": "REVIEW_SUBMITTED",
      "actor": "reviewer_001",
      "details": {
        "decision": "APPROVED",
        "note": "Verified"
      },
      "timestamp": "2026-04-27T15:30:30Z"
    }
  ]
}
```

**Response Schema Details**:
- `events`: Chronologically ordered array of immutable events
- `event_type`: One of CASE_CREATED, REVIEW_SUBMITTED, AUDIT_LOG_QUERIED
- `actor`: System or human user identifier
- `details`: Event-specific data (varies by event_type)
- **Immutability Guarantee**: Events are append-only, never modified or deleted

**Event Types**:

| Event Type | Actor | Details | When |
|------------|-------|---------|------|
| CASE_CREATED | system | {score, decision, model_version} | Upon POST /score |
| REVIEW_SUBMITTED | reviewer_id | {decision, note, previous_status} | Upon POST /review |
| STATUS_UPDATED | system | {old_status, new_status} | Auto on review |

**Example**:
```bash
curl http://localhost:8000/audit/case_550e8400e29b
```

**Error Cases**:
```json
{
  "detail": "Case not found: case_invalid"
}
```

---

## 5. Get Cases by Status

### GET /cases

**Purpose**: List cases filtered by status with pagination

**Query Parameters**:
```
status: "QUEUED" | "APPROVED" | "REJECTED" | "ESCALATED" (default: QUEUED)
limit: integer, max 100 (default: 20)
offset: integer (default: 0)
```

**Response**:
```json
{
  "cases": [
    {
      "case_id": "case_550e8400e29b",
      "score": 0.25,
      "decision": "PASS",
      "status": "QUEUED",
      "created_at": "2026-04-27T15:30:00Z",
      "model_version": "1.0.0",
      "threshold_version": "1.0.0"
    }
  ],
  "total": 150,
  "limit": 20,
  "offset": 0
}
```

**Example**:
```bash
curl "http://localhost:8000/cases?status=QUEUED&limit=20&offset=0"
```

---

## 6. Get Case Details

### GET /cases/{case_id}

**Purpose**: Retrieve full case details including metadata and recent audit events

**Path Parameters**:
- `case_id`: Case identifier

**Response**:
```json
{
  "case_id": "case_550e8400e29b",
  "request_id": "req_abc123",
  "score": 0.25,
  "decision": "PASS",
  "status": "QUEUED",
  "model_version": "1.0.0",
  "threshold_version": "1.0.0",
  "feature_contract_version": "1.0.0",
  "created_at": "2026-04-27T15:30:00Z",
  "updated_at": "2026-04-27T15:30:30Z",
  "recent_events": [
    {
      "event_type": "CASE_CREATED",
      "actor": "system",
      "timestamp": "2026-04-27T15:30:00Z"
    }
  ]
}
```

**Example**:
```bash
curl http://localhost:8000/cases/case_550e8400e29b
```

---

## 7. Get Review Statistics

### GET /review-stats

**Purpose**: Get queue statistics and review metrics

**Response**:
```json
{
  "total_cases": 1500,
  "queued": 150,
  "approved": 800,
  "rejected": 400,
  "escalated": 150,
  "approval_rate": 0.533,
  "average_time_to_review_hours": 2.5
}
```

**Example**:
```bash
curl http://localhost:8000/review-stats
```

---

## 8. Get Metrics & Monitoring

### GET /metrics

**Purpose**: Retrieve real-time system metrics, drift analysis, performance, and alerts

**Response**:
```json
{
  "timestamp": "2026-04-27T15:30:45Z",
  "metrics": {
    "request_count": 15230,
    "latency_avg": 145.5,
    "latency_p95": 280.0,
    "latency_p99": 450.0,
    "score_mean": 0.35,
    "score_median": 0.28,
    "score_std": 0.22,
    "score_min": 0.01,
    "score_max": 0.99,
    "case_counts": {
      "total": 15230,
      "queued": 450,
      "approved": 12000,
      "rejected": 2500,
      "escalated": 280
    },
    "review_rate": 0.97
  },
  "drift_analysis": {
    "features_analyzed": 35,
    "features_alert": 2,
    "alert": true,
    "details": {
      "transaction_amount": {
        "ks_statistic": 0.12,
        "p_value": 0.001,
        "training_mean": 2500.0,
        "current_mean": 3200.0,
        "std_devs_apart": 2.5
      }
    }
  },
  "performance": {
    "precision": 0.92,
    "recall": 0.88,
    "accuracy": 0.90,
    "f1": 0.90,
    "auc": 0.96,
    "tp": 800,
    "fp": 70,
    "tn": 4000,
    "fn": 130
  },
  "alerts": [
    {
      "alert_type": "DRIFT",
      "triggered": true,
      "reason": "Feature 'transaction_amount' KS p-value 0.001 < threshold 0.05",
      "threshold": 0.05,
      "current_value": 0.001
    },
    {
      "alert_type": "LATENCY",
      "triggered": false,
      "reason": "P99 latency 450ms < threshold 1000ms",
      "threshold": 1000.0,
      "current_value": 450.0
    }
  ]
}
```

**Metrics Details**:
- **Latency**: p95, p99 percentiles in milliseconds
- **Drift**: KS statistic per feature, alert if p-value < 0.05
- **Performance**: Precision, recall, F1 from recent reviews
- **Alerts**: List of triggered alerts with reasons

**Example**:
```bash
curl http://localhost:8000/metrics
```

---

## 9. Get Explain Score

### POST /explain

**Purpose**: Get explanation for a scored case (model feature importance or agent analysis)

**Request Body**:
```json
{
  "case_id": "case_550e8400e29b",
  "tx_features": {
    "transaction_amount": 5000.0,
    ...
  }
}
```

**Response**:
```json
{
  "case_id": "case_550e8400e29b",
  "score": 0.25,
  "explanation": "Low-risk transaction based on feature analysis",
  "feature_importance": {
    "transaction_velocity": 0.25,
    "geographic_risk_score": 0.20,
    "transaction_count_24h": 0.15,
    ...
  },
  "top_risk_factors": [
    {
      "feature": "transaction_velocity",
      "value": 0.1,
      "importance": 0.25
    }
  ]
}
```

**Example**:
```bash
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{
    "case_id": "case_550e8400e29b",
    "tx_features": {...}
  }'
```

---

## Feature Contract Specification

### 35-Feature Schema

All features required in `/score` requests must conform to this schema:

| # | Feature Name | Type | Position | Null Behavior | Description |
|----|--------------|------|----------|---------------|-------------|
| 1 | transaction_amount | FLOAT | 0 | DISALLOW | Transaction amount in USD |
| 2 | transaction_count_24h | INT | 1 | DEFAULT:0 | Number of transactions in 24h |
| 3 | unique_destinations_24h | INT | 2 | DEFAULT:0 | Unique destination accounts in 24h |
| 4 | avg_transaction_amount_7d | FLOAT | 3 | DEFAULT:0.0 | Average transaction amount in 7d |
| 5 | days_since_account_creation | INT | 4 | DEFAULT:0 | Days since account opened |
| 6 | is_flagged_as_high_risk | BOOL | 5 | DEFAULT:false | High-risk flag |
| 7 | account_age_category | INT | 6 | DEFAULT:0 | Age category (0-3) |
| 8 | transaction_velocity | FLOAT | 7 | DEFAULT:0.0 | Transaction rate (per hour) |
| 9 | geographic_risk_score | FLOAT | 8 | DEFAULT:0.0 | Geographic risk [0,1] |
| 10 | transaction_type_risk | FLOAT | 9 | DEFAULT:0.0 | Transaction type risk [0,1] |

*(See `models/lgbm_final/feature_contract.json` for complete 35-feature specification)*

**Type Coercion Rules**:
- `"5.7"` → `5` for INT fields (truncated)
- `"123"` → `123.0` for FLOAT fields
- `"true"` → `true` for BOOL fields
- `null` or missing → handled per Null Behavior policy

**Null Behavior Policies**:
- `DISALLOW`: Reject transaction with error
- `ZERO`: Fill with 0 (or 0.0 / false)
- `DEFAULT`: Fill with specified default value
- `MEAN`: Fill with training set mean

---

## API Versioning

Current version: **1.0.0**

Breaking changes will increment major version (e.g., 2.0.0).

Add version header (optional):
```bash
curl -H "X-API-Version: 1.0.0" http://localhost:8000/health
```

---

## Rate Limiting

Currently unlimited. Production deployments should implement:
- 1000 requests/min per API key
- 100 requests/min for unauthenticated clients

---

## Authentication (Future)

Currently unauthenticated. Production deployments will require:

```bash
curl -H "Authorization: Bearer <token>" http://localhost:8000/score
```

---

## Backward Compatibility

All endpoints guarantee backward compatibility within major version:
- Response fields will not be removed
- New optional fields may be added
- Existing fields will not change type
- Feature contract versioning enables model migration

---
