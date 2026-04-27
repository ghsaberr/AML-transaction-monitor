# Reviewer Workflow Guide

## Overview

This document describes the complete workflow for compliance officers and reviewers who assess cases flagged by the AML Risk Decision Engine.

---

## Review Queue Management

### Accessing the Review Queue

**Endpoint**: `GET /cases?status=QUEUED`

**What You See**:
- List of cases pending review
- Cases sorted by creation timestamp (oldest first)
- Score, decision, and model version for each case

**CLI Example**:
```bash
curl "http://localhost:8000/cases?status=QUEUED&limit=20"
```

**Response**:
```json
{
  "cases": [
    {
      "case_id": "case_abc123",
      "score": 0.75,
      "decision": "ALERT",
      "status": "QUEUED",
      "created_at": "2026-04-27T08:00:00Z",
      "model_version": "1.0.0"
    }
  ],
  "total": 42,
  "limit": 20,
  "offset": 0
}
```

### Queue Statistics

**Endpoint**: `GET /review-stats`

**What You See**:
- Total cases in queue
- Queue breakdown by status (QUEUED, APPROVED, REJECTED, ESCALATED)
- Approval rate
- Average time to review

**CLI Example**:
```bash
curl http://localhost:8000/review-stats
```

**Response**:
```json
{
  "total_cases": 15230,
  "queued": 450,
  "approved": 12000,
  "rejected": 2500,
  "escalated": 280,
  "approval_rate": 0.786,
  "average_time_to_review_hours": 2.5
}
```

---

## Case Review Process

### Step 1: Select Case from Queue

Choose a case from the review queue. Cases are sorted by creation time (FIFO).

**Recommended Order**:
1. High-risk cases (score > 0.8) first
2. Then medium-risk (0.5-0.8)
3. Low-risk (< 0.5) as capacity allows

### Step 2: Retrieve Case Details

**Endpoint**: `GET /cases/{case_id}`

**CLI Example**:
```bash
curl http://localhost:8000/cases/case_abc123
```

**Response**:
```json
{
  "case_id": "case_abc123",
  "request_id": "req_xyz",
  "score": 0.75,
  "decision": "ALERT",
  "status": "QUEUED",
  "model_version": "1.0.0",
  "threshold_version": "1.0.0",
  "feature_contract_version": "1.0.0",
  "created_at": "2026-04-27T08:00:00Z",
  "recent_events": [
    {
      "event_type": "CASE_CREATED",
      "actor": "system",
      "timestamp": "2026-04-27T08:00:00Z"
    }
  ]
}
```

### Step 3: Get Explanation for Score

**Endpoint**: `POST /explain`

**Purpose**: Understand which features contributed to the AML risk score, with optional RAG-based reasoning and document citations

**CLI Example**:
```bash
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{
    "case_id": "case_abc123",
    "tx_features": {
      "transaction_amount": 15000.0,
      "transaction_count_24h": 25,
      "unique_destinations_24h": 8,
      "avg_transaction_amount_7d": 5000.0,
      "days_since_account_creation": 90,
      "is_flagged_as_high_risk": true,
      "account_age_category": 1,
      "transaction_velocity": 0.85,
      "geographic_risk_score": 0.7,
      "transaction_type_risk": 0.6
    },
    "tx_text": "High velocity, multiple destinations, new account - potential structuring pattern"
  }'
```

**Response** (with RAG + LLM enabled):
```json
{
  "case_id": "case_abc123",
  "score": 0.85,
  "decision": "ALERT",
  "explanation_type": "agent_rag",
  "rationale": [
    "Risk Score: 85.00%",
    "Top Risk Factors: transaction_velocity, geographic_risk_score, transaction_type_risk",
    "Similar cases: [rule_002], [rule_003]",
    "High transaction velocity combined with multiple destinations suggests potential structuring activity. The pattern closely matches known smurfing typologies."
  ],
  "cited_docs": ["rule_002", "rule_003"],
  "llm_enabled": true,
  "top_features": [
    {
      "feature_name": "transaction_velocity",
      "importance_value": 0.45,
      "contribution": "positive"
    },
    {
      "feature_name": "geographic_risk_score",
      "importance_value": 0.32,
      "contribution": "positive"
    },
    {
      "feature_name": "transaction_type_risk",
      "importance_value": 0.28,
      "contribution": "positive"
    }
  ],
  "timestamp": "2026-04-27T15:30:45Z"
}
```

**Response** (fallback without LLM):
```json
{
  "case_id": "case_abc123",
  "score": 0.85,
  "decision": "ALERT",
  "explanation_type": "feature_importance",
  "rationale": [
    "Risk Score: 85.00%",
    "Top Risk Factors: transaction_velocity, geographic_risk_score, transaction_type_risk"
  ],
  "cited_docs": [],
  "llm_enabled": false,
  "top_features": [
    {
      "feature_name": "transaction_velocity",
      "importance_value": 0.45,
      "contribution": "positive"
    }
  ],
  "timestamp": "2026-04-27T15:30:45Z"
}
```

**How to Interpret the Explanation**:

1. **Decision**: "ALERT" (score ≥ 0.5) or "PASS" (score < 0.5)
   - ALERT cases require human review
   - PASS cases can be processed automatically

2. **Rationale Bullet Points**: 
   - Shows the risk score and top contributing features
   - If LLM is enabled, includes natural language reasoning
   - Citations like `[rule_002]` reference similar historical cases

3. **Cited Documents**:
   - Look up document IDs in the knowledge base (`data/knowledge_base/sample_aml_rules.jsonl`)
   - Example: `rule_002` might describe "smurfing" patterns
   - Citations provide regulatory traceability: you can show *why* the system flagged this case

4. **Top Features**:
   - Shows which features most influenced the score
   - `importance_value`: Contribution weight (higher = more influential)
   - `contribution`: "positive" means it increased the risk score

5. **LLM Enabled**:
   - `true`: Full RAG-based explanation with natural language reasoning
   - `false`: Feature importance only (LLM unavailable or disabled)

**Verifying Cited Documents**:
```bash
# Look up a cited document in the knowledge base
grep "rule_002" data/knowledge_base/sample_aml_rules.jsonl
```

This helps reviewers verify that the AI's reasoning aligns with documented AML typologies.

### Step 4: Review Audit Trail

**Endpoint**: `GET /audit/{case_id}`

**Purpose**: See complete history of case and any previous reviews/notes

**CLI Example**:
```bash
curl http://localhost:8000/audit/case_abc123
```

**Response**:
```json
{
  "case_id": "case_abc123",
  "events": [
    {
      "event_id": "evt_001",
      "event_type": "CASE_CREATED",
      "actor": "system",
      "details": {
        "score": 0.75,
        "decision": "ALERT",
        "model_version": "1.0.0"
      },
      "timestamp": "2026-04-27T08:00:00Z"
    }
  ]
}
```

### Step 5: Make Review Decision

After analyzing the case, score, explanation, and audit trail, make one of three decisions:

#### Decision 1: APPROVED ✓

**Meaning**: Transaction is legitimate. Low AML risk confirmed.

**When to Use**:
- Score is low (< 0.3) and aligns with normal behavior
- High-amount transaction but justified (e.g., planned investment)
- New account but legitimate use case
- Geographic/destination flags explained by business travel

**Endpoint**: `POST /review`

**CLI Example**:
```bash
curl -X POST http://localhost:8000/review \
  -H "Content-Type: application/json" \
  -d '{
    "case_id": "case_abc123",
    "reviewer_id": "reviewer_001",
    "decision": "APPROVED",
    "note": "Verified legitimate international transfer for business purposes"
  }'
```

**Response**:
```json
{
  "review_id": "review_123",
  "case_id": "case_abc123",
  "reviewer_id": "reviewer_001",
  "decision": "APPROVED",
  "previous_status": "QUEUED",
  "new_status": "APPROVED",
  "timestamp": "2026-04-27T08:15:00Z"
}
```

**Audit Impact**:
- Case moved from QUEUED → APPROVED
- Decision logged immutably in audit trail
- Timestamp recorded

#### Decision 2: REJECTED ✗

**Meaning**: Transaction is suspicious/fraudulent. Recommend blocking/investigation.

**When to Use**:
- Score is high (> 0.7) and pattern shows clear AML risk
- Multiple red flags (large amount + new account + suspicious geography)
- Known suspicious entity involved
- Pattern matches known fraud schemes

**Endpoint**: `POST /review`

**CLI Example**:
```bash
curl -X POST http://localhost:8000/review \
  -H "Content-Type: application/json" \
  -d '{
    "case_id": "case_abc123",
    "reviewer_id": "reviewer_001",
    "decision": "REJECTED",
    "note": "Suspicious scatter transaction pattern to 15 new accounts in 24h. Recommend SAR filing."
  }'
```

**Response**: Status changes to REJECTED

**Audit Impact**:
- Case moved from QUEUED → REJECTED
- Decision and note logged for compliance filing
- Supports regulatory reporting

#### Decision 3: ESCALATED ⬆

**Meaning**: Uncertain. Requires investigation or escalation to senior reviewer.

**When to Use**:
- Score is borderline (0.4-0.6) and unclear
- Conflicting signals (legitimate amount but suspicious pattern)
- Missing information needed for decision
- Policy question requiring management decision

**Endpoint**: `POST /review`

**CLI Example**:
```bash
curl -X POST http://localhost:8000/review \
  -H "Content-Type: application/json" \
  -d '{
    "case_id": "case_abc123",
    "reviewer_id": "reviewer_001",
    "decision": "ESCALATED",
    "note": "Borderline score. Account created 2 weeks ago. Recommend verification with customer before decision."
  }'
```

**Response**: Status changes to ESCALATED

**Audit Impact**:
- Case moved from QUEUED → ESCALATED
- Marked for manager review
- Note documents reason for escalation

---

## Review Decision Matrix

| Score | Pattern | Decision | Confidence |
|---|---|---|---|
| > 0.8 | High-risk flags | REJECTED | High |
| 0.6-0.8 | Mixed signals | ESCALATED | Medium |
| 0.4-0.6 | Borderline | ESCALATED | Low |
| < 0.3 | Normal behavior | APPROVED | High |

---

## Scoring Guidelines by Score Range

### Score 0.8-1.0: High Risk

**Red Flags** (any 2+):
- Large transaction amount (> $50K) for new account
- Multiple destinations in short time (> 10 in 24h)
- High geographic risk score (> 0.8)
- Account flagged as high-risk in data
- Recent account (< 30 days) with large transactions

**Action**: Default REJECTED unless strong justification

**Example Justifications**:
- Corporate account with legitimate wire transfer authority
- Investment account receiving expected large deposit
- International business account with known pattern

---

### Score 0.5-0.8: Medium Risk

**Caution Flags** (any 3+):
- Moderate transaction amount ($10K-$50K)
- Multiple destinations (5-10 in 24h)
- Medium geographic risk (0.5-0.8)
- Some transaction type risk flags
- Moderate account age (30-180 days)

**Action**: Review case details carefully. ESCALATED if uncertain.

**Decision Factors**:
- Customer history (new vs established)
- Transaction pattern (consistent vs unusual)
- Geographic destination (known vs new)
- Amount justification (vs account history)

---

### Score 0.3-0.5: Low-Medium Risk

**Minor Flags**:
- Small-to-moderate transaction amount
- Few destinations
- Low geographic risk
- Established account
- Normal velocity

**Action**: Lean toward APPROVED unless pattern shows clear concern

---

### Score 0.0-0.3: Very Low Risk

**Normal Transaction Characteristics**:
- Typical transaction amount for account
- Single or few destinations
- Low geographic risk
- Established account
- Normal behavior

**Action**: APPROVED unless specific reason for concern

---

## Common Scenarios

### Scenario 1: High-Amount Wire to New Country

**Case**: $75K wire from established account to new geographic destination, score 0.72

**Analysis**:
- Feature importance: transaction_amount (30%), geographic_risk_score (25%)
- Account is 5 years old (low account age risk)
- Destination is known financial hub (medium geographic risk, not suspicious)

**Recommended Action**: APPROVED

**Reasoning**: Large amount + new geography are flags, but account history is established and destination is legitimate

---

### Scenario 2: Scatter Transaction Pattern

**Case**: $5K split across 15 accounts in 24h, new account, score 0.85

**Analysis**:
- Feature importance: unique_destinations_24h (35%), account_age_category (20%)
- Scatter pattern matches known layering technique
- Transaction amount is small but high frequency

**Recommended Action**: REJECTED

**Reasoning**: Classic AML pattern (structuring/layering) regardless of amount

---

### Scenario 3: Borderline Decision

**Case**: $12K transfer to established vendor account, score 0.52, account age 6 weeks

**Analysis**:
- Features: transaction_amount (20%), account_age_category (25%), unique_destinations_24h (10%)
- Account is new but transfer is to known business vendor
- Amount is moderate for business transaction

**Recommended Action**: ESCALATED

**Reasoning**: Insufficient information to distinguish legitimate new business account from potential risk. Recommend verification with customer.

---

## Review Metrics & SLA

### Performance Targets

| Metric | Target | Current |
|---|---|---|
| Average Review Time | < 5 minutes | 2.5 minutes ✓ |
| Queue Backlog | < 100 cases | 42 cases ✓ |
| Approval Rate | 70-85% | 78.6% ✓ |
| Escalation Rate | 10-20% | 18.2% ✓ |
| Rejection Rate | 5-15% | 16.4% ✓ |

### Quality Metrics

- **Consensus Rate**: Multi-reviewer agreement on same case (target > 90%)
- **Appeal Rate**: Cases overturned on second review (target < 5%)
- **Model Accuracy**: Cases approved/rejected match model decision (target > 90%)

---

## Tips for Efficient Reviews

1. **Score Triage**: Review high-score cases (0.7+) first
2. **Batch Review**: Process 20-30 cases of similar risk level together
3. **Use Explanations**: Always check feature importance before deciding
4. **Audit Trail**: Check if case has been reviewed previously
5. **Take Notes**: Document reasoning for complex decisions (helps team learning)
6. **Escalate Quickly**: Don't spend > 5 minutes on uncertain cases

---

## Common Challenges & Solutions

### Challenge: Too Many False Positives

**Signs**: Approval rate > 85%, queue building up

**Solutions**:
- Check model version (may need retraining)
- Review threshold setting (may be too sensitive)
- Escalate concerns to ML team for model review

### Challenge: Not Enough Cases to Review

**Signs**: Approval rate > 90%, almost empty queue

**Solutions**:
- Threshold may be too strict
- May need to lower threshold to catch more edge cases
- Review historical cases for patterns

### Challenge: Disagreement Between Reviewers

**Signs**: Consensus rate < 85%

**Solutions**:
- Schedule calibration session with team
- Compare cases where reviewers disagree
- Align on decision guidelines
- May indicate model needs refinement

---

## Compliance & Documentation

### Required Documentation

Every review decision must include:
1. **Case ID**: Unique identifier
2. **Reviewer ID**: Your identifier
3. **Decision**: APPROVED / REJECTED / ESCALATED
4. **Note**: Reason for decision (50+ characters recommended)

### Regulatory Requirements

- **SAR Filing**: REJECTED cases may require Suspicious Activity Report (SAR)
- **Documentation**: Notes become part of regulatory audit trail
- **Retention**: All cases retained for 5+ years
- **Audit Trail**: Immutable log of all decisions

---

## Escalation Process

### When to Escalate

1. **Score Borderline** (0.4-0.6): Uncertain risk assessment
2. **Missing Information**: Cannot fully assess without customer verification
3. **Policy Question**: Requires management decision
4. **Urgent Case**: Requires priority attention
5. **Complex Pattern**: Suspected organized fraud or sophisticated scheme

### Escalation Procedure

1. Click "ESCALATED" decision in review UI
2. Add detailed note explaining reason for escalation
3. Case routed to next-level reviewer (manager)
4. Manager reviews within 24 hours
5. Manager makes final decision (APPROVED/REJECTED)

---

## Training & Certification

### Required for Reviewers

- Initial training on AML fundamentals
- Review workflow and system training
- 5 supervised reviews with trainer
- Quarterly certification renewal (quiz)

### Available Resources

- [Model Card](model_card.md): Model performance and limitations
- [Architecture Overview](architecture.md): System design
- [API Reference](api_contract.md): System endpoints
- [Operations Guide](operations.md): Troubleshooting

---

## Feedback & Improvement

Help us improve the system:

- **Model Accuracy**: Feedback on incorrect model decisions
- **Feature Importance**: Comments on relevance of top factors
- **Usability**: Suggestions for UI/workflow improvements
- **False Positives**: Report cases that waste reviewer time

Send feedback to: **compliance-ops@example.com**

---
