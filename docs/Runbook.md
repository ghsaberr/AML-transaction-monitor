# RUNBOOK — AML Transaction Monitoring API

This document describes how to run, validate, and troubleshoot the AML Scoring API
in local Docker and AWS Lambda deployments.

The service exposes three endpoints:
- GET /health
- POST /score
- POST /explain (may be disabled via feature flag)

---

## How to Run (Local Docker)

Build and start the service:

docker build -t aml-api:local .
docker run -p 8080:8080 aml-api:local

### Health check:

curl http://127.0.0.1:8080/health

### Score request:

curl -X POST http://127.0.0.1:8080/score \
  -H "Content-Type: application/json" \
  --data-binary "@req_score.json"

### Explain request:

curl -X POST http://127.0.0.1:8080/explain \
  -H "Content-Type: application/json" \
  --data-binary "@req_explain.json"

---

## How to Run (AWS Lambda + API Gateway)

### Health check:

curl https://<api-id>.execute-api.<region>.amazonaws.com/health

### Score request:

curl -X POST https://<api-id>.execute-api.<region>.amazonaws.com/score \
  -H "Content-Type: application/json" \
  --data-binary "@req_score.json"

---

## Logs & Monitoring

Logs are written automatically to CloudWatch Logs:

- Log group: /aws/lambda/<function-name>

To tail logs:

aws logs tail "/aws/lambda/<function-name>" --region <region> --since 10m

Default CloudWatch metrics are available for Lambda and API Gateway
(invocations, errors, latency).

