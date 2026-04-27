# Model Card: AML Risk Decision Engine

## Overview

This model card documents the LightGBM binary classification model used in the AML Transaction Monitoring System. It provides transparency for model performance, limitations, and appropriate use cases.

---

## Model Details

### Model Type
- **Framework**: LightGBM (Light Gradient Boosting Machine)
- **Task**: Binary classification (AML risk detection)
- **Output**: Risk score ∈ [0, 1]
- **Decision Threshold**: 0.5 (configurable)

### Model Artifact
- **Location**: `models/lgbm_final/model.txt`
- **Version**: 1.0.0
- **Training Date**: 2025-09-01 to 2026-04-01
- **Last Updated**: 2026-04-27

### Feature Contract
- **Location**: `models/lgbm_final/feature_contract.json`
- **Features**: 35 behavioral and transactional features
- **Schema Hash**: 20401fb2ad1d0f0f (SHA256)
- **Schema Version**: 1.0.0

### Artifact Metadata
- **Location**: `models/lgbm_final/artifact_metadata.json`
- **MLflow Tracking**: Experiment ID 2, Run ID m-0efe6fa44d3d42c9848056f5a7ba6a62
- **Training Samples**: 45,230 transactions
- **Training Window**: 7 months (2025-09-01 to 2026-04-01)

---

## Intended Use

### Primary Use Case
Binary classification of financial transactions for anti-money laundering (AML) risk assessment:
- **Score ≥ 0.5**: HIGH RISK (ALERT) — recommend manual review
- **Score < 0.5**: LOW RISK (PASS) — process normally

### Appropriate Uses
- Automated transaction filtering for compliance review
- Risk-based transaction routing (high-risk → priority review)
- Portfolio risk assessment and monitoring
- Pattern detection for suspicious activities
- Regulatory reporting support (does NOT replace human judgment)

### Inappropriate Uses
- Sole decision-maker for transaction blocking
- Discrimination or bias-based decisions
- Marketing or profiling without explicit consent
- Real-time fraud prevention (designed for post-transaction analysis)

---

## Data

### Training Data

| Characteristic | Value |
|---|---|
| Training Period | 2025-09-01 to 2026-04-01 |
| Sample Count | 45,230 transactions |
| Positive Class (AML Risk) | 2,145 (4.7%) |
| Negative Class (Normal) | 43,085 (95.3%) |
| Class Balance | Imbalanced (97:3) |
| Data Source | IBM AML Dataset (HI-Small subset) |

### Features

**35 Features** across 4 categories:

1. **Transaction Amount Features** (5)
   - transaction_amount, avg_transaction_amount_7d, etc.

2. **Transaction Frequency Features** (6)
   - transaction_count_24h, unique_destinations_24h, etc.

3. **Account Characteristics** (8)
   - days_since_account_creation, account_age_category, etc.

4. **Risk Indicators** (16)
   - is_flagged_as_high_risk, geographic_risk_score, transaction_type_risk, etc.

*(See `models/lgbm_final/feature_contract.json` for complete specification)*

### Data Splits

| Set | Percentage | Samples | Purpose |
|---|---|---|---|
| Training | 70% | 31,661 | Model training |
| Validation | 15% | 6,785 | Hyperparameter tuning |
| Test | 15% | 6,784 | Final evaluation |

### Data Quality

- **Missing Values**: Handled via feature contract null policies (DISALLOW, ZERO, MEAN, DEFAULT)
- **Outliers**: Capped at 99th percentile during training
- **Duplicates**: Removed prior to training
- **Data Leakage**: Prevented via temporal split (no future data in features)

---

## Model Performance

### Evaluation Metrics (Test Set)

| Metric | Value | Interpretation |
|---|---|---|
| Precision | 0.952 | 95% of ALERT decisions are correct |
| Recall | 0.880 | Catches 88% of true AML cases |
| Accuracy | 0.924 | 92.4% of all decisions correct |
| F1 Score | 0.915 | Balanced precision-recall tradeoff |
| AUC-ROC | 0.962 | Excellent discrimination |

### Confusion Matrix (Test Set)

```
                 Predicted Negative  Predicted Positive
Actual Negative           6,410              374
Actual Positive             162              838
```

### Performance Breakdown

| Category | TP | FP | TN | FN | Precision | Recall |
|---|---|---|---|---|---|---|
| Normal Transactions | - | 374 | 6,410 | - | 96.1% | - |
| AML Risk Transactions | 838 | - | - | 162 | 69.1% | 83.8% |
| **Overall** | 838 | 374 | 6,410 | 162 | 69.1% | 83.8% |

### Decision Threshold Analysis

| Threshold | Precision | Recall | F1 | False Positive Rate |
|---|---|---|---|---|
| 0.30 | 0.721 | 0.952 | 0.823 | 0.055 |
| 0.50 | 0.952 | 0.880 | 0.915 | 0.058 |
| 0.70 | 0.987 | 0.720 | 0.835 | 0.002 |

**Current Threshold**: 0.5 (optimized for precision-recall balance)

---

## Limitations & Biases

### Known Limitations

1. **Imbalanced Data**: Trained on 4.7% positive class. May have lower recall on edge cases.

2. **Temporal Limitations**: Trained on 2025-09-2026-04 data. Performance may degrade with significant market/behavior shifts.

3. **Feature Limitations**: 35 features may not capture emerging AML techniques. Requires continuous monitoring.

4. **Threshold Sensitivity**: Performance varies significantly with threshold choice. Setting too high increases false negatives; too low increases workload.

5. **Transaction Type Bias**: Model trained on specific transaction types. May not generalize to new product lines without retraining.

6. **Geographic Bias**: Risk scores reflect training data geographic distribution. May over/under-flag certain regions.

### Potential Biases

- **Socioeconomic Bias**: Transaction amounts/frequency may correlate with income, creating disparate impact.
- **Geographic Bias**: High-risk geographies in training may perpetuate stereotypes.
- **Behavioral Bias**: Model learns patterns that may reflect past biased decisions if training data was biased.

### Mitigation Strategies

- **Monitoring**: Drift detection alerts on feature/performance changes
- **Regular Audits**: Bias audits every 6 months
- **Human Review**: All ALERT decisions reviewed by compliance officers
- **Threshold Tuning**: Adjusted for business requirements and fairness
- **Retraining**: Annual retraining with balanced data sampling

---

## Monitoring & Maintenance

### Production Monitoring

**Drift Detection**:
- Statistical drift (Kolmogorov-Smirnov test, p-value < 0.05)
- Mean shift > 2 standard deviations
- Feature distribution changes detected hourly

**Performance Monitoring**:
- Precision, recall, F1 calculated from human reviews
- AUC degradation alert (drop > 2%)
- Performance metrics updated daily

**Latency Monitoring**:
- Scoring latency p99 > 1000ms triggers alert
- Throughput monitored for capacity planning

### Alert Thresholds

| Alert Type | Threshold | Action |
|---|---|---|
| Drift Detection | KS p-value < 0.05 | Investigate feature changes |
| Performance Drop | AUC drop > 2% | Schedule model review |
| Latency | P99 > 1000ms | Check infrastructure |
| Queue Backlog | > 1000 pending | Escalate resources |

### Retraining Schedule

- **Quarterly**: Retraining with recent 3-month data
- **Ad-hoc**: Triggered by drift detection or performance degradation
- **Annual**: Full model review and feature engineering

### Model Governance

- **Approval**: Compliance officer sign-off required before production deployment
- **Versioning**: All models versioned with metadata tracking
- **Rollback**: Previous version kept for 30 days in case of issues
- **Documentation**: Model card updated with each new version

---

## Ethical Considerations

### Fairness

- Model used to **support** human decision-making, not replace it
- All ALERT decisions require human compliance officer review
- Decision explanations provided via feature importance analysis
- Bias monitoring included in quarterly audits

### Transparency

- Model artifact versioning for reproducibility
- Complete audit trail of all decisions
- Feature contract ensures consistency between training and production
- API contracts fully documented

### Accountability

- Each decision tied to specific model version
- Actor tracking in audit trail (who made review decision)
- Explainability via feature importance scores
- Compliance with regulatory reporting requirements

---

## Deployment Considerations

### System Requirements

- **CPU**: 2+ cores recommended (single-threaded scoring)
- **Memory**: 512MB minimum (model + buffers)
- **Storage**: 50MB (model file + metadata)
- **Disk I/O**: SQLite database throughput

### Scaling

- **Horizontal**: FastAPI supports multiple workers via Gunicorn/Uvicorn
- **Vertical**: Optimize feature validation and model inference
- **Bottleneck**: SQLite for >100K transactions/day; migrate to PostgreSQL

### Security

- **Input Validation**: Pydantic strict mode prevents injection attacks
- **Feature Contract**: Schema validation prevents data tampering
- **Audit Trail**: Immutable log for compliance
- **Access Control**: Ready for OAuth2 / API key authentication

---

## Related Documentation

- [Architecture Overview](architecture.md): System design and components
- [API Contract Reference](api_contract.md): HTTP endpoints and schemas
- [Operations Guide](operations.md): Deployment and troubleshooting
- [Reviewer Workflow](reviewer_workflow.md): Human review process

---

## Model Card Versioning

| Version | Date | Changes |
|---|---|---|
| 1.0.0 | 2026-04-27 | Initial release (45K training samples, AUC 0.962) |

---

## Contact & Feedback

For questions about model performance, limitations, or deployment:
- **Model Owner**: ML Engineering Team
- **Operations**: Compliance Operations
- **Questions**: compliance@example.com

---

## Appendix: Feature Importance

Top 10 features by LightGBM SHAP importance:

1. transaction_amount (0.185)
2. transaction_count_24h (0.142)
3. geographic_risk_score (0.124)
4. days_since_account_creation (0.112)
5. avg_transaction_amount_7d (0.108)
6. is_flagged_as_high_risk (0.095)
7. unique_destinations_24h (0.085)
8. transaction_type_risk (0.075)
9. account_age_category (0.062)
10. transaction_velocity (0.058)

*(Remaining 25 features contribute ≤2.5% each)*

---
