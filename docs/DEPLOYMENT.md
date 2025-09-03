# Deployment Guide

This document provides comprehensive instructions for deploying the RedBus Demand Forecasting system.

## Table of Contents

- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Monitoring and Logging](#monitoring-and-logging)

## Local Development

### Prerequisites
- Python 3.8+
- pip
- Virtual environment (recommended)

### Setup
```bash
# Clone repository
git clone https://github.com/yourusername/redbus-demand-forecasting.git
cd redbus-demand-forecasting

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run application
python run_streamlit.py
```

## Docker Deployment

### Single Container
```bash
# Build image
docker build -t redbus-forecasting .

# Run container
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/config:/app/config \
  redbus-forecasting
```

### Docker Compose
```bash
# Development environment
docker-compose up --build

# Production environment
docker-compose -f docker-compose.prod.yml up -d
```

## Cloud Deployment

### AWS ECS

1. **Create ECR Repository**
```bash
aws ecr create-repository --repository-name redbus-forecasting
```

2. **Build and Push Image**
```bash
# Get login token
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-west-2.amazonaws.com

# Build and tag image
docker build -t redbus-forecasting .
docker tag redbus-forecasting:latest <account-id>.dkr.ecr.us-west-2.amazonaws.com/redbus-forecasting:latest

# Push image
docker push <account-id>.dkr.ecr.us-west-2.amazonaws.com/redbus-forecasting:latest
```

3. **Create ECS Service**
```json
{
  "family": "redbus-forecasting-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::<account-id>:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "redbus-forecasting",
      "image": "<account-id>.dkr.ecr.us-west-2.amazonaws.com/redbus-forecasting:latest",
      "portMappings": [
        {
          "containerPort": 8501,
          "protocol": "tcp"
        }
      ],
      "essential": true,
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/redbus-forecasting",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### Google Cloud Run

```bash
# Build and deploy
gcloud run deploy redbus-forecasting \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --port 8501
```

### Azure Container Instances

```bash
# Create resource group
az group create --name redbus-rg --location eastus

# Deploy container
az container create \
  --resource-group redbus-rg \
  --name redbus-forecasting \
  --image yourdockerhub/redbus-forecasting:latest \
  --dns-name-label redbus-forecasting \
  --ports 8501 \
  --cpu 2 \
  --memory 4
```

## Kubernetes Deployment

### Manifests

**Deployment**
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redbus-forecasting
  labels:
    app: redbus-forecasting
spec:
  replicas: 3
  selector:
    matchLabels:
      app: redbus-forecasting
  template:
    metadata:
      labels:
        app: redbus-forecasting
    spec:
      containers:
      - name: redbus-forecasting
        image: yourdockerhub/redbus-forecasting:latest
        ports:
        - containerPort: 8501
        env:
        - name: PYTHONPATH
          value: "/app/src"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1"
        livenessProbe:
          httpGet:
            path: /_stcore/health
            port: 8501
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /_stcore/health
            port: 8501
          initialDelaySeconds: 5
          periodSeconds: 5
```

**Service**
```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: redbus-forecasting-service
spec:
  selector:
    app: redbus-forecasting
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8501
  type: LoadBalancer
```

**Ingress**
```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: redbus-forecasting-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - redbus.yourdomain.com
    secretName: redbus-tls
  rules:
  - host: redbus.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: redbus-forecasting-service
            port:
              number: 80
```

### Deploy to Kubernetes

```bash
# Apply manifests
kubectl apply -f k8s/

# Check deployment
kubectl get pods -l app=redbus-forecasting
kubectl get svc redbus-forecasting-service

# Scale deployment
kubectl scale deployment redbus-forecasting --replicas=5
```

## Helm Chart

```yaml
# helm/redbus-forecasting/values.yaml
replicaCount: 3

image:
  repository: yourdockerhub/redbus-forecasting
  tag: latest
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 80

ingress:
  enabled: true
  annotations:
    kubernetes.io/ingress.class: nginx
  hosts:
    - host: redbus.yourdomain.com
      paths:
        - path: /
          pathType: Prefix

resources:
  limits:
    cpu: 1000m
    memory: 2Gi
  requests:
    cpu: 500m
    memory: 1Gi

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 80
```

## Monitoring and Logging

### Prometheus Monitoring

```yaml
# monitoring/servicemonitor.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: redbus-forecasting
spec:
  selector:
    matchLabels:
      app: redbus-forecasting
  endpoints:
  - port: http
    interval: 30s
    path: /metrics
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "RedBus Forecasting Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph", 
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))"
          }
        ]
      }
    ]
  }
}
```

### Logging with ELK Stack

```yaml
# logging/filebeat.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: filebeat-config
data:
  filebeat.yml: |
    filebeat.inputs:
    - type: container
      paths:
        - /var/log/containers/*redbus-forecasting*.log
      processors:
      - add_kubernetes_metadata:
          host: ${NODE_NAME}
          matchers:
          - logs_path:
              logs_path: "/var/log/containers/"

    output.elasticsearch:
      hosts: ["elasticsearch:9200"]

    setup.kibana:
      host: "kibana:5601"
```

## Environment Variables

### Production Configuration

```bash
# Required environment variables
export PYTHONPATH=/app/src
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_HEADLESS=true

# Optional configuration
export MODEL_PATH=/app/data/models
export LOG_LEVEL=INFO
export CACHE_TTL=3600
```

## Health Checks

The application provides health check endpoints:

- `/_stcore/health` - Streamlit health check
- `/health` - Custom application health check (if implemented)

## Troubleshooting

### Common Issues

1. **Memory Issues**
   - Increase container memory limits
   - Optimize model loading and caching

2. **Slow Startup**
   - Use init containers for model preloading
   - Implement lazy loading patterns

3. **Connection Issues**
   - Check firewall rules
   - Verify service discovery configuration

### Debug Commands

```bash
# Check container logs
docker logs redbus-forecasting

# Connect to running container
docker exec -it redbus-forecasting /bin/bash

# Check Kubernetes pod logs
kubectl logs -f deployment/redbus-forecasting

# Port forwarding for debugging
kubectl port-forward svc/redbus-forecasting-service 8501:80
```

## Security Considerations

1. **Image Security**
   - Use minimal base images
   - Scan for vulnerabilities
   - Keep dependencies updated

2. **Runtime Security**
   - Run as non-root user
   - Use read-only file systems
   - Implement resource limits

3. **Network Security**
   - Use TLS/SSL certificates
   - Implement network policies
   - Restrict ingress/egress

## Performance Optimization

1. **Caching**
   - Implement model caching
   - Use Redis for session storage
   - Cache feature computations

2. **Scaling**
   - Horizontal pod autoscaling
   - Load balancing strategies
   - Database connection pooling

3. **Monitoring**
   - Set up alerts for high latency
   - Monitor resource utilization
   - Track prediction accuracy
