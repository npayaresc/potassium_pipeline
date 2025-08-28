#!/bin/bash
# Monitor AutoGluon training progress

echo "=== AutoGluon Training Monitor ==="
echo "Started at: $(date)"
echo

# Check process
echo "Process Status:"
if ps aux | grep -E "main.py.*autogluon" | grep -v grep > /dev/null; then
    echo "✓ AutoGluon process is running"
    ps aux | grep -E "main.py.*autogluon" | grep -v grep | awk '{print "  PID:", $2, "CPU:", $3"%", "MEM:", $4"%"}'
else
    echo "✗ AutoGluon process not found"
fi
echo

# Check log progress
echo "Latest Log Entries:"
tail -10 logs/pipeline.log | grep -E "(AutoGluon|Training|Fitting|Model)" | tail -5
echo

# Check model directory
MODEL_DIR="models/autogluon/simple_only_20250722_214846"
echo "Model Directory Status:"
if [ -d "$MODEL_DIR" ]; then
    echo "✓ Model directory exists: $MODEL_DIR"
    echo "  Contents: $(ls -1 $MODEL_DIR 2>/dev/null | wc -l) items"
    if [ -d "$MODEL_DIR/models" ]; then
        echo "  Models: $(ls -1 $MODEL_DIR/models 2>/dev/null | wc -l) trained"
    fi
else
    echo "✗ Model directory not yet created"
fi
echo

# Check GPU usage if available
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | awk -F', ' '{print "  GPU Util:", $1"%, Memory:", $2"/"$3" MB"}'
fi