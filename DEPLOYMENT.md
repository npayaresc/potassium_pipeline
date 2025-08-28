# Cloud-Agnostic Deployment Guide

This guide provides instructions for deploying the Magnesium Pipeline to various cloud platforms while maintaining cloud agnosticism.

## Overview

The Magnesium Pipeline is designed to be cloud-agnostic, using containerization and standardized APIs to ensure easy migration between cloud providers. The deployment strategy focuses on:

- **Docker containerization** for consistent environments
- **RESTful API** for universal access
- **Environment-based configuration** for different deployment targets
- **GPU support** across cloud platforms

## Quick Start

### Local Development
```bash
# Build and run with Docker Compose
docker-compose up --build

# Or run in development mode
docker-compose --profile dev up
```

### Production Deployment
```bash
# Build production image
docker build --target production -t magnesium-pipeline:latest .

# Run with GPU support
docker run --gpus all -p 8000:8000 magnesium-pipeline:latest
```

## Deployment Options

### 1. Google Cloud Platform (Recommended)

#### Option A: Cloud Run (Serverless)
```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/magnesium-pipeline

# Deploy to Cloud Run with GPU (when available)
gcloud run deploy magnesium-pipeline \
  --image gcr.io/PROJECT_ID/magnesium-pipeline \
  --platform managed \
  --region us-central1 \
  --memory 8Gi \
  --cpu 4 \
  --timeout 3600 \
  --allow-unauthenticated
```

#### Option B: GKE (Kubernetes)
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: magnesium-pipeline
spec:
  replicas: 1
  selector:
    matchLabels:
      app: magnesium-pipeline
  template:
    metadata:
      labels:
        app: magnesium-pipeline
    spec:
      containers:
      - name: magnesium-pipeline
        image: gcr.io/PROJECT_ID/magnesium-pipeline:latest
        ports:
        - containerPort: 8000
        env:
        - name: STORAGE_TYPE
          value: "gcs"
        - name: STORAGE_BUCKET
          value: "your-bucket-name"
        resources:
          limits:
            nvidia.com/gpu: 1
          requests:
            memory: "8Gi"
            cpu: "4"
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: magnesium-data-pvc
      nodeSelector:
        accelerator: nvidia-tesla-t4
```

#### Option C: Vertex AI Custom Training
```python
# vertex_ai_deploy.py
from google.cloud import aiplatform

aiplatform.init(project="PROJECT_ID", location="us-central1")

job = aiplatform.CustomContainerTrainingJob(
    display_name="magnesium-pipeline-training",
    container_uri="gcr.io/PROJECT_ID/magnesium-pipeline:latest",
    command=["python", "main.py", "autogluon", "--gpu"],
    machine_type="n1-standard-4",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
)

job.run()
```

### 2. Amazon Web Services (AWS)

#### Option A: ECS Fargate
```bash
# Push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ACCOUNT.dkr.ecr.us-east-1.amazonaws.com
docker tag magnesium-pipeline:latest ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/magnesium-pipeline:latest
docker push ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/magnesium-pipeline:latest

# Create ECS service (use AWS Console or CLI)
```

#### Option B: EKS (Kubernetes)
```yaml
# Similar to GKE deployment, adjust image registry
apiVersion: apps/v1
kind: Deployment
metadata:
  name: magnesium-pipeline
spec:
  template:
    spec:
      containers:
      - name: magnesium-pipeline
        image: ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/magnesium-pipeline:latest
        env:
        - name: STORAGE_TYPE
          value: "s3"
        - name: STORAGE_BUCKET
          value: "your-s3-bucket"
```

#### Option C: SageMaker
```python
# sagemaker_deploy.py
import boto3
from sagemaker.pytorch import PyTorchModel

model = PyTorchModel(
    image_uri="ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/magnesium-pipeline:latest",
    role="SageMakerExecutionRole",
    entry_point="api_server.py"
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.g4dn.xlarge"
)
```

### 3. Microsoft Azure

#### Option A: Container Instances
```bash
# Create container instance
az container create \
  --resource-group magnesium-rg \
  --name magnesium-pipeline \
  --image magnesium-pipeline:latest \
  --cpu 4 \
  --memory 8 \
  --gpu-count 1 \
  --gpu-sku V100 \
  --ports 8000 \
  --environment-variables STORAGE_TYPE=azure
```

#### Option B: AKS (Kubernetes)
```yaml
# Similar deployment to GKE/EKS
apiVersion: apps/v1
kind: Deployment
metadata:
  name: magnesium-pipeline
spec:
  template:
    spec:
      containers:
      - name: magnesium-pipeline
        image: your-registry.azurecr.io/magnesium-pipeline:latest
        env:
        - name: STORAGE_TYPE
          value: "azure"
```

### 4. Any Kubernetes Cluster

#### Helm Chart Deployment
```yaml
# helm/values.yaml
image:
  repository: magnesium-pipeline
  tag: latest
  pullPolicy: IfNotPresent

resources:
  limits:
    nvidia.com/gpu: 1
    memory: 8Gi
    cpu: 4

storage:
  type: local
  size: 100Gi

service:
  type: LoadBalancer
  port: 8000

ingress:
  enabled: true
  hosts:
    - magnesium-api.yourdomain.com
```

```bash
helm install magnesium-pipeline ./helm
```

## Configuration Management

### Environment Variables
```bash
# API Configuration
export API_HOST="0.0.0.0"
export API_PORT="8000"
export API_WORKERS="4"

# Storage Configuration
export STORAGE_TYPE="gcs"  # or s3, azure, local
export STORAGE_BUCKET="your-bucket"
export DATA_PATH="/app/data"

# Compute Configuration
export GPU_ENABLED="true"
export CPU_CORES="4"

# Security
export API_KEY_REQUIRED="true"
export API_KEY="your-secure-api-key"
```

### Cloud-Specific Configuration Files

#### GCP Configuration
```yaml
# config/gcp.yml
storage:
  type: gcs
  bucket_name: magnesium-pipeline-data
  credentials_path: /var/secrets/google/key.json

compute:
  gpu_enabled: true
  gpu_memory_fraction: 0.8

cloud_providers:
  gcp:
    project_id: your-project-id
    region: us-central1
```

#### AWS Configuration
```yaml
# config/aws.yml
storage:
  type: s3
  bucket_name: magnesium-pipeline-data
  
compute:
  gpu_enabled: true

cloud_providers:
  aws:
    region: us-east-1
    instance_type: ml.g4dn.xlarge
```

## API Usage

### Training a Model
```bash
# Start training job
curl -X POST "https://your-api-endpoint/train" \
  -H "Content-Type: application/json" \
  -d '{
    "pipeline_type": "autogluon",
    "use_gpu": true
  }'

# Check training status
curl "https://your-api-endpoint/train/train_20240721_123456"
```

### Making Predictions
```bash
# Single prediction
curl -X POST "https://your-api-endpoint/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample.csv.txt" \
  -F "model_path=/app/models/autogluon/model_123"

# Batch prediction
curl -X POST "https://your-api-endpoint/predict/batch" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@sample1.csv.txt" \
  -F "files=@sample2.csv.txt" \
  -F "model_path=/app/models/best_model.pkl"
```

## Monitoring and Logging

### Health Checks
```bash
# API health check
curl "https://your-api-endpoint/health"
```

### Metrics Collection
The API exposes metrics at `/metrics` (Prometheus format) when monitoring is enabled.

### Cloud Logging
- **GCP**: Automatically uses Cloud Logging
- **AWS**: Integrates with CloudWatch
- **Azure**: Uses Azure Monitor

## Security Best Practices

### 1. Network Security
- Use private subnets for compute resources
- Implement API gateways with authentication
- Enable HTTPS/TLS encryption

### 2. Container Security
```dockerfile
# Use non-root user
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# Scan for vulnerabilities
# docker scan magnesium-pipeline:latest
```

### 3. Secrets Management
- Use cloud-native secret management (GCP Secret Manager, AWS Secrets Manager, Azure Key Vault)
- Never hardcode secrets in containers
- Use service accounts with minimal permissions

### 4. Access Control
```yaml
# Kubernetes RBAC example
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: magnesium-pipeline-role
rules:
- apiGroups: [""]
  resources: ["pods", "configmaps"]
  verbs: ["get", "list", "create"]
```

## Scaling Strategies

### Horizontal Scaling
```yaml
# Kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: magnesium-pipeline-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: magnesium-pipeline
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### GPU Scaling
- Use GPU node pools with auto-scaling
- Implement GPU sharing for inference workloads
- Consider spot/preemptible instances for training

## Cost Optimization

### 1. Resource Management
- Use appropriate instance sizes
- Implement auto-scaling based on demand
- Use spot/preemptible instances for non-critical workloads

### 2. Storage Optimization
- Use lifecycle policies for data retention
- Compress models and datasets
- Implement intelligent caching strategies

### 3. Multi-Cloud Cost Comparison
- Use terraform for infrastructure as code
- Implement cost monitoring dashboards
- Regular cost optimization reviews

## Migration Between Cloud Providers

### 1. Data Migration
```bash
# GCP to AWS example
gsutil -m cp -r gs://source-bucket s3://destination-bucket

# AWS to Azure example
aws s3 sync s3://source-bucket azure-blob-destination
```

### 2. Configuration Updates
- Update storage configuration
- Modify container registry references
- Update DNS and load balancer configurations

### 3. Testing and Validation
- Run integration tests on new environment
- Performance benchmarking
- Validate GPU acceleration is working

## Troubleshooting

### Common Issues

#### GPU Not Detected
```bash
# Check GPU availability in container
docker run --gpus all magnesium-pipeline:latest python check_gpu_support.py
```

#### Storage Access Issues
```bash
# Check storage permissions
kubectl logs magnesium-pipeline-pod

# Test storage connectivity
curl -X POST "https://api-endpoint/health"
```

#### Memory Issues
```bash
# Check resource usage
kubectl top pods
docker stats
```

### Debug Mode
```bash
# Run in debug mode
docker run -e LOG_LEVEL=DEBUG magnesium-pipeline:latest
```

## Support

For deployment issues:
1. Check the logs: `kubectl logs <pod-name>` or `docker logs <container-id>`
2. Verify GPU access: Run `check_gpu_support.py`
3. Test API health: `curl /health` endpoint
4. Review configuration: Check environment variables and config files

## Next Steps

1. **Set up CI/CD pipeline** for automated deployment
2. **Implement monitoring and alerting**
3. **Set up backup and disaster recovery**
4. **Performance optimization and benchmarking**
5. **Security auditing and compliance**