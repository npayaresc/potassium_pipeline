# GCP Cloud Experiment Plan for Magnesium Pipeline (Linux)
## Target Metrics: R² ≥ 0.8, MAPE < 10%, MAE < 0.04

## Setup Instructions

### Initial Setup (One-time)
```bash
# Make script executable
chmod +x deploy/gcp_deploy.sh

# Set your GCP project
export PROJECT_ID="mapana-ai-models"
export REGION="us-central1"
export BUCKET_NAME="${PROJECT_ID}-magnesium-data"

# Setup GCP project and APIs
./deploy/gcp_deploy.sh setup

# Build container image
./deploy/gcp_deploy.sh build

# Upload training data to GCS
./deploy/gcp_deploy.sh upload-data
```

## Phase 1: Baseline & Feature Engineering (Days 1-3)

### 1.1 Feature Strategy Experiments

```bash
# Experiment 1A: Mg_only features
export TRAINING_MODE="train"
export STRATEGY="Mg_only"
export USE_GPU="true"
export MODELS="xgboost,lightgbm,catboost,random_forest,extratrees"
./deploy/gcp_deploy.sh vertex-ai

# Experiment 1B: Simple_only features
export STRATEGY="simple_only"
./deploy/gcp_deploy.sh vertex-ai

# Experiment 1C: Full_context features
export STRATEGY="full_context"
./deploy/gcp_deploy.sh vertex-ai
```

### 1.2 Raw Spectral Data Mode

```bash
# Experiment 2A: Raw spectral with XGBoost
export USE_RAW_SPECTRAL="true"
export STRATEGY="full_context"
export MODELS="xgboost"
./deploy/gcp_deploy.sh vertex-ai

# Experiment 2B: Raw spectral with Neural Network
export MODELS="neural_network"
./deploy/gcp_deploy.sh vertex-ai
```

## Phase 2: Hyperparameter Optimization (Days 3-6)

### 2.1 XGBoost Optimization

```bash
# Experiment 3A: XGBoost with full_context (500 trials)
export TRAINING_MODE="optimize-xgboost"
export STRATEGY="full_context"
export TRIALS="500"
export USE_GPU="true"
export MACHINE_TYPE="n1-highmem-8"
export ACCELERATOR_TYPE="NVIDIA_TESLA_T4"
export ACCELERATOR_COUNT="1"
./deploy/gcp_deploy.sh vertex-ai

# Experiment 3B: XGBoost with simple_only
export STRATEGY="simple_only"
export TRIALS="400"
./deploy/gcp_deploy.sh vertex-ai

# Experiment 3C: XGBoost with Mg_only
export STRATEGY="Mg_only"
export TRIALS="300"
./deploy/gcp_deploy.sh vertex-ai
```

### 2.2 Multi-Model Optimization

```bash
# Experiment 4A: LightGBM & CatBoost optimization
export TRAINING_MODE="optimize-models"
export MODELS="lightgbm,catboost"
export STRATEGY="full_context"
export TRIALS="400"
export USE_GPU="true"
./deploy/gcp_deploy.sh vertex-ai

# Experiment 4B: Tree-based models optimization
export MODELS="random_forest,extratrees"
export STRATEGY="simple_only"
export TRIALS="300"
export USE_GPU="false"
export MACHINE_TYPE="n1-standard-8"
./deploy/gcp_deploy.sh vertex-ai

# Experiment 4C: Neural Network optimization
export MODELS="neural_network,neural_network_light"
export STRATEGY="full_context"
export TRIALS="200"
export USE_GPU="true"
export MACHINE_TYPE="n1-highmem-4"
export ACCELERATOR_TYPE="NVIDIA_TESLA_T4"
./deploy/gcp_deploy.sh vertex-ai
```

### 2.3 Hyperparameter Tuning with Different Objectives

```bash
# Experiment 5A: Tune with distribution-based objective
export TRAINING_MODE="tune"
export STRATEGY="full_context"
export USE_GPU="true"
export TRIALS="400"
export TIMEOUT="14400"
./deploy/gcp_deploy.sh vertex-ai

# Experiment 5B: Tune with MAPE-focused objective
# (Requires config file modification - create custom config)
export TRAINING_MODE="tune"
export STRATEGY="simple_only"
export TRIALS="300"
./deploy/gcp_deploy.sh vertex-ai
```

## Phase 3: AutoGluon Ensemble Learning (Days 6-10)

### 3.1 Standard AutoGluon Training

```bash
# Experiment 6A: AutoGluon with best_quality preset
export TRAINING_MODE="autogluon"
export USE_GPU="true"
export STRATEGY="full_context"
export MACHINE_TYPE="n1-highmem-8"
export ACCELERATOR_TYPE="NVIDIA_TESLA_T4"
export ACCELERATOR_COUNT="1"
export TIMEOUT="21600"
./deploy/gcp_deploy.sh vertex-ai

# Experiment 6B: AutoGluon with simple_only features
export STRATEGY="simple_only"
./deploy/gcp_deploy.sh vertex-ai

# Experiment 6C: AutoGluon with Mg_only features
export STRATEGY="Mg_only"
./deploy/gcp_deploy.sh vertex-ai
```

### 3.2 Extended AutoGluon Training

```bash
# Experiment 7A: Extended AutoGluon (6 hours)
export TRAINING_MODE="autogluon"
export USE_GPU="true"
export STRATEGY="full_context"
export MACHINE_TYPE="n1-highmem-16"
export ACCELERATOR_TYPE="NVIDIA_TESLA_V100"
export ACCELERATOR_COUNT="1"
export TIMEOUT="21600"
./deploy/gcp_deploy.sh vertex-ai

# Experiment 7B: AutoGluon with raw spectral data
export USE_RAW_SPECTRAL="true"
export STRATEGY="full_context"
./deploy/gcp_deploy.sh vertex-ai
```

## Phase 4: Advanced Combinations (Days 10-14)

### 4.1 Best Strategy Combinations

```bash
# Experiment 8A: Best feature strategy + XGBoost optimization
# (Use the best strategy from Phase 1)
export TRAINING_MODE="optimize-xgboost"
export STRATEGY="simple_only"
export TRIALS="800"
export USE_GPU="true"
export MACHINE_TYPE="n1-highmem-8"
./deploy/gcp_deploy.sh vertex-ai

# Experiment 8B: Best features + AutoGluon extended
export TRAINING_MODE="autogluon"
export STRATEGY="simple_only"
export USE_GPU="true"
export TIMEOUT="28800"
export MACHINE_TYPE="n1-highmem-16"
export ACCELERATOR_TYPE="NVIDIA_TESLA_V100"
./deploy/gcp_deploy.sh vertex-ai
```

### 4.2 Neural Network Deep Training

```bash
# Experiment 9A: Neural Network with extended training
export TRAINING_MODE="optimize-models"
export MODELS="neural_network"
export STRATEGY="full_context"
export TRIALS="500"
export USE_GPU="true"
export MACHINE_TYPE="n1-highmem-8"
export ACCELERATOR_TYPE="NVIDIA_TESLA_V100"
./deploy/gcp_deploy.sh vertex-ai

# Experiment 9B: Neural Network with raw spectral
export USE_RAW_SPECTRAL="true"
export TRIALS="400"
./deploy/gcp_deploy.sh vertex-ai
```

## Phase 5: Production Optimization (Days 14-16)

### 5.1 Final Optimization Runs

```bash
# Experiment 10A: Massive hyperparameter search
export TRAINING_MODE="tune"
export STRATEGY="full_context"
export TRIALS="1000"
export TIMEOUT="43200"
export USE_GPU="true"
export MACHINE_TYPE="n1-highmem-32"
export ACCELERATOR_TYPE="NVIDIA_TESLA_V100"
export ACCELERATOR_COUNT="2"
./deploy/gcp_deploy.sh vertex-ai

# Experiment 10B: Ultimate AutoGluon run (12 hours)
export TRAINING_MODE="autogluon"
export USE_GPU="true"
export TIMEOUT="43200"
export MACHINE_TYPE="n1-highmem-32"
export ACCELERATOR_TYPE="NVIDIA_TESLA_V100"
export ACCELERATOR_COUNT="2"
./deploy/gcp_deploy.sh vertex-ai
```

## Batch Execution Scripts

Create shell scripts for running experiments in batches:

### create_batch_scripts.sh
```bash
#!/bin/bash

# Create batch execution scripts
cat > batch_phase1.sh << 'EOF'
#!/bin/bash
# Phase 1: Feature Engineering Experiments

# Function to run experiment and log
run_experiment() {
    local exp_name=$1
    shift
    echo "Starting experiment: $exp_name"
    echo "Command: $@"
    $@ > logs/${exp_name}.log 2>&1 &
    echo "PID: $! - $exp_name"
}

mkdir -p logs

# Run all feature strategy experiments
export USE_GPU="true"
export TRAINING_MODE="train"
export MODELS="xgboost,lightgbm,catboost,random_forest,extratrees"

export STRATEGY="Mg_only"
run_experiment "exp1a_mg_only" ./deploy/gcp_deploy.sh vertex-ai

export STRATEGY="simple_only"
run_experiment "exp1b_simple_only" ./deploy/gcp_deploy.sh vertex-ai

export STRATEGY="full_context"
run_experiment "exp1c_full_context" ./deploy/gcp_deploy.sh vertex-ai

echo "All Phase 1 experiments started. Check logs/ directory for progress."
EOF

chmod +x batch_phase1.sh
```

### Parallel Execution (Advanced)
```bash
# Run multiple experiments in parallel using GNU parallel
cat > parallel_experiments.txt << EOF
train:Mg_only:xgboost,lightgbm
train:simple_only:xgboost,lightgbm
train:full_context:xgboost,lightgbm
optimize-xgboost:full_context:
optimize-xgboost:simple_only:
autogluon:full_context:
autogluon:simple_only:
EOF

# Execute in parallel (requires GNU parallel)
cat parallel_experiments.txt | parallel --colsep ':' \
    'export TRAINING_MODE={1}; export STRATEGY={2}; export MODELS={3}; export USE_GPU=true; ./deploy/gcp_deploy.sh vertex-ai'
```

## Monitoring & Results Collection

### Monitor Running Jobs
```bash
# List all running jobs
gcloud ai custom-jobs list --region=us-central1

# Stream logs from specific job
gcloud ai custom-jobs stream-logs [JOB_ID] --region=us-central1

# Check job status
gcloud ai custom-jobs describe [JOB_ID] --region=us-central1

# Watch job progress (updates every 30 seconds)
watch -n 30 'gcloud ai custom-jobs list --region=us-central1 --filter="state=JOB_STATE_RUNNING"'
```

### Create Monitoring Dashboard Script
```bash
cat > monitor_jobs.sh << 'EOF'
#!/bin/bash
# Monitor all running Vertex AI jobs

while true; do
    clear
    echo "=== Vertex AI Jobs Dashboard ==="
    echo "Time: $(date)"
    echo ""
    echo "Running Jobs:"
    gcloud ai custom-jobs list --region=us-central1 \
        --filter="state=JOB_STATE_RUNNING" \
        --format="table(displayName,state,createTime.date())"
    echo ""
    echo "Recently Completed:"
    gcloud ai custom-jobs list --region=us-central1 \
        --filter="state=JOB_STATE_SUCCEEDED" \
        --limit=5 \
        --format="table(displayName,state,endTime.date())"
    sleep 60
done
EOF
chmod +x monitor_jobs.sh
```

### Download Results from GCS
```bash
# List all results in bucket
gsutil ls -r gs://${PROJECT_ID}-magnesium-data/magnesium-pipeline/

# Download all reports
gsutil -m cp -r gs://${PROJECT_ID}-magnesium-data/magnesium-pipeline/reports/* ./cloud_results/

# Download specific model
gsutil cp gs://${PROJECT_ID}-magnesium-data/magnesium-pipeline/models/best_model.pkl ./

# Download AutoGluon models
gsutil -m cp -r gs://${PROJECT_ID}-magnesium-data/magnesium-pipeline/models/autogluon/* ./autogluon_models/

# Sync results continuously
gsutil -m rsync -r gs://${PROJECT_ID}-magnesium-data/magnesium-pipeline/reports/ ./cloud_results/
```

### Results Analysis Script
```bash
cat > analyze_results.sh << 'EOF'
#!/bin/bash
# Analyze experiment results

RESULTS_DIR="./cloud_results"
OUTPUT_FILE="experiment_summary.csv"

echo "Experiment,Strategy,R2,MAPE,MAE,Training_Time" > $OUTPUT_FILE

for report in $RESULTS_DIR/training_summary_*.csv; do
    if [ -f "$report" ]; then
        # Extract metrics from report (adjust based on actual format)
        exp_name=$(basename $report .csv)
        # Parse CSV and extract best model metrics
        tail -n +2 $report | head -1 | awk -F',' \
            '{print "'$exp_name'," $2 "," $3 "," $4 "," $5 "," $6}' >> $OUTPUT_FILE
    fi
done

echo "Summary saved to $OUTPUT_FILE"
# Display top performers
echo "Top 5 experiments by R2:"
sort -t',' -k3 -rn $OUTPUT_FILE | head -6
EOF
chmod +x analyze_results.sh
```

## Quick Start Commands

For immediate high-impact experiments:

```bash
# 1. Best AutoGluon configuration
export TRAINING_MODE="autogluon"
export USE_GPU="true"
export STRATEGY="simple_only"
export MACHINE_TYPE="n1-highmem-8"
export ACCELERATOR_TYPE="NVIDIA_TESLA_T4"
./deploy/gcp_deploy.sh vertex-ai

# 2. Optimized XGBoost
export TRAINING_MODE="optimize-xgboost"
export STRATEGY="full_context"
export TRIALS="500"
export USE_GPU="true"
./deploy/gcp_deploy.sh vertex-ai

# 3. Multi-model optimization
export TRAINING_MODE="optimize-models"
export MODELS="xgboost,lightgbm,catboost"
export STRATEGY="simple_only"
export TRIALS="400"
export USE_GPU="true"
./deploy/gcp_deploy.sh vertex-ai
```

## Environment Management

### Save experiment configurations
```bash
# Save current environment for reproducibility
cat > experiments/exp_config_$(date +%Y%m%d_%H%M%S).env << EOF
PROJECT_ID=$PROJECT_ID
REGION=$REGION
TRAINING_MODE=$TRAINING_MODE
USE_GPU=$USE_GPU
STRATEGY=$STRATEGY
MODELS=$MODELS
TRIALS=$TRIALS
MACHINE_TYPE=$MACHINE_TYPE
ACCELERATOR_TYPE=$ACCELERATOR_TYPE
EOF

# Load saved configuration
source experiments/exp_config_20250131_120000.env
```

## Cost Optimization Tips

1. **Use Preemptible VMs** for non-critical experiments
   ```bash
   export USE_PREEMPTIBLE="true"
   ```

2. **Start with smaller trials** (100-200) to identify promising approaches
3. **Use n1-standard machines** for CPU-only experiments
4. **Use T4 GPUs** for most experiments (V100 only for final runs)
5. **Set appropriate timeouts** to avoid runaway jobs

## Machine Type Recommendations

| Experiment Type | Machine Type | GPU | Use Case |
|----------------|--------------|-----|----------|
| CPU-only | n1-standard-4 | None | Tree-based models |
| Small GPU | n1-standard-4 | 1x T4 | Quick tests |
| Standard GPU | n1-highmem-8 | 1x T4 | Most experiments |
| Large GPU | n1-highmem-16 | 1x V100 | Extended training |
| Production | n1-highmem-32 | 2x V100 | Final runs |

## Expected Timeline

- **Days 1-3**: Feature engineering (15 experiments)
- **Days 3-6**: Hyperparameter optimization (12 experiments)
- **Days 6-10**: AutoGluon training (8 experiments)
- **Days 10-14**: Advanced combinations (10 experiments)
- **Days 14-16**: Production optimization (5 experiments)

**Total**: ~50 cloud experiments over 16 days

## Troubleshooting

### Common Issues and Solutions

1. **Job fails immediately**
   ```bash
   # Check if image exists
   ./deploy/gcp_deploy.sh build
   # Check if data uploaded
   ./deploy/gcp_deploy.sh check-data
   ```

2. **Out of memory errors**
   ```bash
   export MACHINE_TYPE="n1-highmem-16"
   ```

3. **GPU not available**
   ```bash
   # Check quota
   gcloud compute project-info describe --project=$PROJECT_ID
   # Try different region
   export REGION="us-east1"
   ```

4. **Timeout errors**
   ```bash
   export TIMEOUT="43200"  # 12 hours
   ```

5. **Can't find results**
   ```bash
   # Check GCS bucket
   gsutil ls gs://${PROJECT_ID}-magnesium-data/
   # Check job logs
   ./deploy/gcp_deploy.sh upload-logs
   ```

## Next Steps After Experiments

1. **Analysis**: Run `./analyze_results.sh` to compare all results
2. **Model Selection**: Choose top 3-5 models based on metrics
3. **Ensemble Creation**: Combine best models for improved performance
4. **Validation**: Test on held-out validation data
5. **Deployment**: Deploy best model as inference service
   ```bash
   ./deploy/gcp_deploy.sh inference
   ```

## Success Metrics Tracking

Create a tracking spreadsheet or use this command to generate CSV:
```bash
# Generate metrics report
echo "JobID,Experiment,Strategy,Models,R2,MAPE,MAE,Time,Cost" > metrics_tracking.csv
# Append results after each experiment
```