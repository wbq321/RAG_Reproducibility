#!/bin/bash

# Cluster Cross-Model Analysis Runner
# This script runs the cross-model retrieval analysis on the cluster with local models

# Cluster-specific paths
PROJECT_ROOT="/scratch/user/u.bw269205/rag_reproducibility/RAG_Reproducibility"
MODEL_BASE_PATH="/scratch/user/u.bw269205/shared_models"

# Change to project directory
cd "$PROJECT_ROOT"

# Set offline mode environment variables
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Set CUDA workspace for deterministic operations
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Verify model paths exist
echo "Verifying model paths..."
if [ ! -d "$MODEL_BASE_PATH/bge_model" ]; then
    echo "ERROR: BGE model not found at $MODEL_BASE_PATH/bge_model"
    exit 1
fi

if [ ! -d "$MODEL_BASE_PATH/intfloat_e5-base-v2" ]; then
    echo "ERROR: E5 model not found at $MODEL_BASE_PATH/intfloat_e5-base-v2"
    exit 1
fi

if [ ! -d "$MODEL_BASE_PATH/Qwen_Qwen3-Embedding-0.6B" ]; then
    echo "ERROR: Qwen model not found at $MODEL_BASE_PATH/Qwen_Qwen3-Embedding-0.6B"
    exit 1
fi

echo "All model paths verified successfully."

# Create data directory if it doesn't exist
mkdir -p data
mkdir -p results

# Run the cross-model analysis with local model paths
echo "Starting cross-model retrieval analysis..."
python cross_model_retrieval_analysis.py \
    --models bge e5 qw \
    --top-k 50 \
    --max-docs 5000 \
    --max-queries 100 \
    --data-dir data \
    --output-dir results \
    --model-base-path "$MODEL_BASE_PATH"

if [ $? -eq 0 ]; then
    echo "Cross-model analysis completed successfully!"
    echo "Results saved in: $PROJECT_ROOT/results"
else
    echo "Cross-model analysis failed!"
    exit 1
fi
