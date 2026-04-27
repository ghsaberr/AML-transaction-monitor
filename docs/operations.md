# Operations Guide: Deployment, Monitoring & Troubleshooting

## Overview

This guide provides operations teams with instructions for deploying, monitoring, and maintaining the AML Risk Decision Engine in production.

---

## System Requirements

### Development Environment
- **OS**: Windows, macOS, Linux
- **Python**: 3.11+
- **Memory**: 4GB minimum, 8GB recommended
- **Storage**: 50GB (models + data)
- **Network**: Internet access for dependency installation

### Production Environment
- **OS**: Linux (Ubuntu 20.04+ or compatible)
- **CPU**: 2+ cores (1 core minimum, performance degrades)
- **Memory**: 2GB minimum, 4GB recommended
- **Storage**: SSD 100GB (database growth + model versions)
- **Network**: Internal network (no internet required after bootstrap)

### Docker Deployment
- **Image**: Python 3.11 slim + dependencies
- **Port**: 8000 (FastAPI)
- **Health Check**: GET /health (10-second interval)

---

## Deployment

### Option 1: Local Development

#### 1. Clone Repository
```bash
git clone <repo-url>
cd AML-Transaction-Monitoring3
```

#### 2. Set Up Python Environment
```bash
# Using Python venv
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1

# Or using uv (recommended package manager for this project)
uv venv .venv
source .venv/bin/activate
```

#### 3. Install Dependencies
```bash
# Using uv (project standard)
uv pip install -r requirements.txt

# Or traditional pip
pip install -r requirements.txt
```

#### 4. Initialize Database
```bash
python -c "from src.storage.db import init_db; init_db()"
```

#### 5. Start API Server
```bash
# Using uv
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Or traditional
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

**Verify**: http://localhost:8000/health should return status "healthy"

---

### Option 2: Docker Deployment (Production)

#### 1. Build Docker Image
```bash
docker build -t aml-engine:1.0.0 .
```

**Dockerfile**:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy code and models
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Initialize database
RUN python -c "from src.storage.db import init_db; init_db()"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 2. Run Container
```bash
docker run -d \
  --name aml-engine \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/configs:/app/configs:ro \
  aml-engine:1.0.0
```

**Volume Mounts**:
- `/app/data`: Persistent SQLite database
- `/app/configs`: Read-only configuration files
- `/app/models`: Model artifacts (read-only)

#### 3. Verify Container
```bash
docker logs aml-engine
docker ps | grep aml-engine
curl http://localhost:8000/health
```

---

### Option 3: Cloud Deployment (AWS/GCP/Azure)

#### AWS ECS Deployment

**1. Create ECR Repository**:
```bash
aws ecr create-repository --repository-name aml-engine
aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
```

**2. Push Image**:
```bash
docker tag aml-engine:1.0.0 <account>.dkr.ecr.<region>.amazonaws.com/aml-engine:1.0.0
docker push <account>.dkr.ecr.<region>.amazonaws.com/aml-engine:1.0.0
```

**3. Create ECS Task Definition**:
```json
{
  "family": "aml-engine",
  "networkMode": "awsvpc",
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "aml-engine",
      "image": "<account>.dkr.ecr.<region>.amazonaws.com/aml-engine:1.0.0",
      "portMappings": [{"containerPort": 8000, "protocol": "tcp"}],
      "environment": [{"name": "ENV", "value": "production"}],
      "mountPoints": [
        {"sourceVolume": "data", "containerPath": "/app/data"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/aml-engine",
          "awslogs-region": "<region>",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ],
  "volumes": [{"name": "data", "efsVolumeConfiguration": {"filesystemId": "<efs-id>"}}]
}
```

**4. Create ECS Service**:
```bash
aws ecs create-service \
  --cluster production \
  --service-name aml-engine \
  --task-definition aml-engine:1 \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[<subnet-id>],securityGroups=[<sg-id>]}"
```

---

## Monitoring

### Health Check Endpoint

**Endpoint**: `GET /health`

**What to Monitor**:
- Response time < 100ms
- Status = "healthy" (not "degraded")
- Database connectivity = true
- Model loaded = true

**Example**:
```bash
curl -i http://localhost:8000/health
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

### Metrics Endpoint

**Endpoint**: `GET /metrics`

**Key Metrics to Monitor**:
- **Latency**: p99 < 1000ms (alert if > 1000ms)
- **Request Count**: Trending upward = healthy
- **Drift Alert**: Alert if triggered (inspect features)
- **Performance**: AUC >= 0.95, precision >= 0.90
- **Queue Backlog**: Queued < 1000 (alert if > 1000)

**Example**:
```bash
curl http://localhost:8000/metrics | jq '.metrics.latency_p99'
```

### Logging

**Log Location**: `data/aml-engine.log` (configurable)

**Log Levels**:
- `DEBUG`: Detailed tracing (development only)
- `INFO`: General events (deployments, restarts)
- `WARNING`: Potential issues (drift detected, degradation)
- `ERROR`: Failures (database error, model load failed)
- `CRITICAL`: System failure (complete outage)

**Sample Logs**:
```
2026-04-27T15:30:00 INFO Starting AML Engine v1.0.0
2026-04-27T15:30:05 INFO Database initialized: cases=15230, audit_events=42500
2026-04-27T15:30:10 INFO Model loaded: lgbm_final v1.0.0, AUC=0.962
2026-04-27T15:31:00 WARNING Drift detected on feature 'transaction_amount' (KS p-value=0.001)
2026-04-27T15:32:00 ERROR Database connection timeout (retrying...)
```

### Monitoring Stack Recommendations

**Option 1: Prometheus + Grafana** (Recommended)
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'aml-engine'
    static_configs:
      - targets: ['localhost:8000']
```

**Option 2: CloudWatch** (AWS)
```bash
aws cloudwatch put-metric-data \
  --metric-name AMLEngineLatencyP99 \
  --value 450.0 \
  --unit Milliseconds \
  --namespace AML-Engine
```

**Option 3: Datadog / New Relic** (Commercial)
- Pre-built integrations for FastAPI
- Dashboard templates for ML systems

---

## Performance Tuning

### Bottleneck Identification

#### 1. Check Scoring Latency
```bash
# Query metrics endpoint
curl http://localhost:8000/metrics | jq '.metrics.latency_p99'

# If p99 > 1000ms, investigate:
```

**Causes & Fixes**:
- **Feature Validation Slow**: Simplify feature contract validation
- **Model Inference Slow**: Check CPU usage (consider GPU)
- **Database Slow**: Check SQLite journal mode, add indexes

#### 2. Check Database Performance
```bash
# Get database info
sqlite3 data/aml_engine.db "SELECT COUNT(*) FROM cases;"
sqlite3 data/aml_engine.db ".indices"

# Optimize indexes if needed
sqlite3 data/aml_engine.db "CREATE INDEX idx_case_status ON cases(status);"
```

#### 3. Check Memory Usage
```bash
# On Linux
ps aux | grep uvicorn
free -h

# If memory > 2GB, restart application
docker restart aml-engine
```

### Optimization Strategies

| Bottleneck | Symptom | Fix |
|---|---|---|
| Scoring Latency | p99 > 1000ms | Add CPU cores, batch process |
| Database Queries | Query time > 100ms | Add indexes, migrate to PostgreSQL |
| Memory Growth | RAM > 2GB | Restart container weekly |
| Queue Backlog | > 1000 pending cases | Scale horizontally (add workers) |

---

## Scaling

### Horizontal Scaling (Multiple Workers)

**Using Gunicorn** (production-grade):
```bash
gunicorn src.api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --access-logfile - \
  --error-logfile -
```

**Using Docker Compose** (multiple containers):
```yaml
version: '3.8'
services:
  load-balancer:
    image: nginx:latest
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - api-1
      - api-2
      - api-3

  api-1:
    build: .
    environment:
      - WORKER_ID=1
    volumes:
      - ./data:/app/data

  api-2:
    build: .
    environment:
      - WORKER_ID=2
    volumes:
      - ./data:/app/data

  api-3:
    build: .
    environment:
      - WORKER_ID=3
    volumes:
      - ./data:/app/data
```

### Vertical Scaling (Larger Instance)

- Increase CPU: 1 core → 2-4 cores (diminishing returns above 4)
- Increase RAM: 2GB → 4GB (model + buffer)
- Use SSD: Much faster database access

### Database Scaling

**SQLite** (current, suitable for < 100K tx/day):
- Single-file database
- Automatic backups
- No additional infrastructure

**PostgreSQL** (recommended for > 100K tx/day):
```sql
-- Migration steps
-- 1. Create PostgreSQL instance
-- 2. Export SQLite data
-- 3. Import into PostgreSQL
-- 4. Update connection string in config
-- 5. Run integration tests

CREATE TABLE cases (
  case_id TEXT PRIMARY KEY,
  request_id TEXT,
  score REAL,
  status TEXT,
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_case_status ON cases(status);
CREATE INDEX idx_case_created ON cases(created_at);
```

---

## Troubleshooting

### Issue 1: Health Check Returns "degraded"

**Symptom**: `GET /health` returns status = "degraded"

**Diagnosis**:
```bash
curl http://localhost:8000/health | jq '.database_healthy, .model_loaded'
```

**Fixes**:

**If database_healthy = false**:
```bash
# 1. Check database file exists
ls -la data/aml_engine.db

# 2. Check file permissions
chmod 644 data/aml_engine.db

# 3. Reinitialize database
python -c "from src.storage.db import init_db; init_db()"

# 4. Verify
curl http://localhost:8000/health
```

**If model_loaded = false**:
```bash
# 1. Check model files exist
ls -la models/lgbm_final/model.txt
ls -la models/lgbm_final/feature_contract.json

# 2. Check file permissions
chmod 644 models/lgbm_final/*

# 3. Check Python can load model
python -c "from src.agent.model_runner import ModelRunner; mr = ModelRunner('models/lgbm_final'); print(mr.get_metadata())"

# 4. Restart application
docker restart aml-engine
```

### Issue 2: Scoring Latency Spike

**Symptom**: `GET /metrics` shows latency_p99 > 2000ms (sudden increase)

**Diagnosis**:
```bash
# 1. Check CPU usage
top -p $(pgrep -f uvicorn)

# 2. Check memory usage
ps aux | grep uvicorn | awk '{print $6}'

# 3. Check database queries
sqlite3 data/aml_engine.db "SELECT COUNT(*) FROM cases; SELECT COUNT(*) FROM audit_events;"
```

**Fixes**:

**If CPU > 80%**:
```bash
# 1. Increase workers (horizontal scaling)
# 2. Or increase CPU allocation (vertical scaling)
# 3. Or check for unusual transaction volume spike
```

**If Memory > 2GB**:
```bash
# 1. Restart application (clears in-memory buffers)
docker restart aml-engine

# 2. Check MetricsCollector buffer size
# Default: 1000 events, set to 500 if memory constrained
```

**If Database Slow**:
```bash
# 1. Add indexes to frequently queried columns
sqlite3 data/aml_engine.db "CREATE INDEX idx_case_status ON cases(status);"
sqlite3 data/aml_engine.db "CREATE INDEX idx_audit_case_id ON audit_events(case_id);"

# 2. Vacuum database to optimize
sqlite3 data/aml_engine.db "VACUUM;"

# 3. Check for transaction lock conflicts
sqlite3 data/aml_engine.db ".timeout 5000" "SELECT COUNT(*) FROM cases WHERE status='QUEUED';"
```

### Issue 3: Drift Alert Triggered

**Symptom**: `GET /metrics` shows `drift_analysis.alert = true`

**Diagnosis**:
```bash
# 1. Inspect which features are drifting
curl http://localhost:8000/metrics | jq '.drift_analysis.details | keys'

# 2. Check recent transaction values
curl http://localhost:8000/metrics | jq '.drift_analysis.details.transaction_amount'
```

**Fixes**:

**If Legitimate Business Change**:
```bash
# 1. Investigate root cause (new product, seasonal change, etc.)
# 2. If expected, recalibrate drift thresholds in AlertingPolicy
# 3. Plan model retraining

# Example: Increase KS test p-value threshold from 0.05 to 0.10
# Edit: src/monitoring/metrics.py, line ~60
# OLD: if p_value < 0.05:
# NEW: if p_value < 0.10:
```

**If Data Quality Issue**:
```bash
# 1. Check for missing/null values
# 2. Validate feature contract enforcement
# 3. Review recent scoring requests

# Example: Too many nulls in 'transaction_amount'
# Fix: Check feature contract default value, verify input validation
```

**If Model Degradation**:
```bash
# 1. Check model performance on recent reviews
curl http://localhost:8000/metrics | jq '.performance'

# 2. If AUC dropped > 2%, trigger retraining
# 3. Shadow test new model before deployment

# Retraining command (manual):
python src/modeling/train_eval_lgbm_final.py --data-path data/processed/ --output-dir models/lgbm_final_v2/
```

### Issue 4: Database Lock Timeout

**Symptom**: Error in logs: "database is locked"

**Diagnosis**:
```bash
# 1. Check concurrent connections
lsof | grep aml_engine.db

# 2. Check active transactions
sqlite3 data/aml_engine.db "SELECT * FROM sqlite_master WHERE type='table';"
```

**Fixes**:
```bash
# 1. Enable WAL (Write-Ahead Logging) mode
sqlite3 data/aml_engine.db "PRAGMA journal_mode=WAL;"

# 2. Increase timeout in code
# Edit: src/storage/db.py, line ~30
# Change: timeout = 10.0  (seconds)

# 3. Restart application to apply changes
docker restart aml-engine
```

### Issue 5: Out of Memory

**Symptom**: Application crashes with "MemoryError" or OOM killer

**Fixes**:
```bash
# 1. Reduce MetricsCollector buffer size
# Edit: src/monitoring/metrics.py, line ~100
# Change: self.max_buffer_size = 500  (from 1000)

# 2. Increase container memory allocation
docker run -d -m 4g aml-engine:1.0.0  # 4GB limit

# 3. Implement periodic restarts
# Add to crontab (Linux):
0 2 * * * docker restart aml-engine  # Restart at 2 AM daily
```

---

## Backup & Recovery

### Database Backup

**Automated Daily Backup**:
```bash
#!/bin/bash
# backup.sh
DATE=$(date +%Y%m%d_%H%M%S)
sqlite3 data/aml_engine.db ".backup data/backups/aml_engine_${DATE}.db"
# Retain last 30 days only
find data/backups -name "aml_engine_*.db" -mtime +30 -delete
```

**Cron Job** (run at 1 AM daily):
```bash
0 1 * * * /path/to/backup.sh
```

### Disaster Recovery

**If database corrupted**:
```bash
# 1. Restore from latest backup
cp data/backups/aml_engine_20260427_010000.db data/aml_engine.db

# 2. Verify integrity
sqlite3 data/aml_engine.db "PRAGMA integrity_check;"

# 3. Restart application
docker restart aml-engine
```

**If complete system failure**:
```bash
# 1. Provision new server (identical specs)
# 2. Install Docker and pull image
docker pull <registry>/aml-engine:1.0.0

# 3. Restore database backup
cp backups/aml_engine.db /mnt/data/

# 4. Start container
docker run -d -v /mnt/data:/app/data aml-engine:1.0.0

# 5. Verify health
curl http://localhost:8000/health
```

---

## Model Retraining

### When to Retrain

- **Scheduled**: Quarterly (every 3 months)
- **Triggered**: AUC drops > 2%, drift alert ≥ 5 features
- **Urgent**: False negative rate > 2%

### Retraining Process

```bash
# 1. Prepare training data (last 3 months of reviews)
python scripts/prepare_training_data.py \
  --lookback-days 90 \
  --output-dir data/processed/

# 2. Train new model
python src/modeling/train_eval_lgbm_final.py \
  --data-path data/processed/ \
  --output-dir models/lgbm_final_v2/ \
  --hyperparams src/modeling/lgbm_params.json

# 3. Evaluate new model
python -c "
from src.modeling.train_eval_lgbm_final import evaluate_model
metrics = evaluate_model('models/lgbm_final_v2/model.txt', 'data/processed/test.csv')
print(metrics)
"

# 4. Compare with current model
# If AUC improvement > 1%, or recall improvement > 3%, proceed

# 5. Shadow test (optional but recommended)
# Deploy new model alongside current in shadow mode (logging only, not used for decisions)
# Run for 1 week, compare decisions

# 6. Deploy new model
cp -r models/lgbm_final models/lgbm_final_backup_v1.0.0
cp -r models/lgbm_final_v2/* models/lgbm_final/

# 7. Restart application
docker restart aml-engine

# 8. Verify new model loaded
curl http://localhost:8000/health | jq '.version'
```

---

## Maintenance Schedule

### Daily
- Monitor health check (automated)
- Review alerts from drift detection
- Check queue backlog

### Weekly
- Review model performance metrics
- Check database size and optimize if > 1GB
- Verify backup completion

### Monthly
- Audit review decisions for calibration
- Check for unusual patterns (false positives/negatives)
- Plan next retraining cycle

### Quarterly
- Retrain model with new data
- Bias audit on model decisions
- Update thresholds based on business requirements

### Annually
- Full system audit
- Security review
- Capacity planning for next year

---

## Support Contacts

- **Technical Issues**: engineering-oncall@example.com
- **Operational Questions**: ops-team@example.com
- **Model Performance**: ml-engineering@example.com
- **Compliance Questions**: compliance-ops@example.com

---
