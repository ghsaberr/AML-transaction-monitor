# Explainable, Agent-Assisted AML Transaction Monitoring  
**Cloud-Native End-to-End Machine Learning System**

## Overview
This project implements an **end-to-end Anti-Money Laundering (AML) transaction monitoring system** with a strong focus on:

- Temporal-safe ML modeling (no data leakage)
- Graph-aware feature engineering
- Explainability via an optional agent layer
- Cloud-native deployment on AWS
- Production-ready API, containerization

The system is designed as a **reference architecture** for modern AML pipelines in banks and fintech environments.

---

## Key Features
- **Hybrid ML approach**
  - Tabular transaction features
  - Graph-based network features
  - Node2Vec embeddings
- **Explainable AI**
  - Optional agent-based explanation layer
  - Retrieval-augmented reasoning over AML rules
- **Production-ready**
  - FastAPI inference service
  - Dockerized deployment
  - AWS Lambda (container image) + API Gateway
- **Operational flexibility**
  - Explainability layer can be enabled/disabled via environment variables
  - Lightweight default deployment for low latency

---

## Dataset
- **IBM AML Dataset – HI-Small**
- Temporal split to prevent leakage:
  - **Training:** Days 1–9
  - **Testing:** Day 10
- Post-Day-10 transactions are handled carefully as they contain only laundering patterns.

---

## Feature Engineering
### Tabular Features
- Transaction amount statistics
- Time-based aggregates (1h, 24h, 7d)
- Behavioral deltas
- Account activity summaries

### Graph Features
- In-degree, out-degree, total degree
- PageRank
- Ego-network statistics
- **Node2Vec embeddings** for structural context

All features are generated reproducibly and stored as Parquet files.

---

## Modeling
- **Model:** LightGBM
- **Objective:** Binary classification (suspicious vs. normal)
- **Threshold optimization:**  
  Business-aligned trade-off between:
  - Recall (catching suspicious transactions)
  - Alert volume (operational cost)

Artifacts stored:
- Trained model
- Feature list
- Decision threshold

---

## Agent & Explainability Layer
An optional **agent-based explanation layer** enriches model predictions by:

- Combining numerical features with textual signals
- Retrieving AML rules from a knowledge base
- Producing human-readable explanations

To ensure production stability:
- The agent layer is **disabled by default**
- Enabled only when explicitly required

---

## Containerization
- Docker image includes:
  - Trained ML model
  - Feature configuration
  - Vector store (FAISS)
- Uses **AWS Lambda Web Adapter** to run FastAPI inside Lambda without code changes

---

## Cloud Architecture (AWS)
- **Amazon ECR** – container image registry  
- **AWS Lambda (image-based)** – inference runtime  
- **API Gateway** – public HTTP interface  
- **IAM (least privilege)** – secure execution role  
- **CloudWatch Logs** – monitoring and debugging  

Environment-driven configuration allows multiple deployment modes (with or without explainability).

---

## Testing
- Local testing with Docker
- Cloud testing via `curl`
- Realistic transaction payloads used for validation
- Health and inference endpoints verified end-to-end

---

## Monitoring & Observability
This deployment currently uses **AWS Lambda’s built-in CloudWatch logging** for observability.

While no explicit monitoring or alerting rules are configured yet, this stage validates:
- Correct model loading
- Stable API behavior under cold and warm starts
- End-to-end inference execution in a cloud environment

Monitoring and alerting are intentionally left as a **future operational concern**, as the primary goal of this project phase is to demonstrate:
- ML system design
- Cloud-native deployment
- Explainability-aware architecture

---

## Design Philosophy
- **Reproducibility first**
- **No temporal leakage**
- **Explainability as a controlled feature, not a liability**
- **Cloud-native, not cloud-dependent**

---

## Author
This project was built as a **portfolio-grade, production-oriented AML system**