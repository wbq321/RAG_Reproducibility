# Cluster Cross-Model Analysis Setup

This document explains how to run the cross-model retrieval analysis on your cluster with local models.

## Environment Setup

Your cluster configuration:
- Project root: `/scratch/user/u.bw269205/rag_reproducibility/RAG_Reproducibility`
- Model base path: `/scratch/user/u.bw269205/shared_models`

### Model Paths
- BGE: `/scratch/user/u.bw269205/shared_models/bge_model`
- E5: `/scratch/user/u.bw269205/shared_models/intfloat_e5-base-v2`
- Qwen: `/scratch/user/u.bw269205/shared_models/Qwen_Qwen3-Embedding-0.6B`

## Files Modified/Created

### 1. Updated `cross_model_retrieval_analysis.py`
- Added offline mode environment variables
- Added model path mapping functionality
- Updated to use local model paths instead of downloading from HuggingFace

### 2. Created `cluster_cross_model_analysis.py`
- Cluster-specific wrapper script
- Automatically configures paths for your environment
- Includes model path verification

### 3. Created `test_cluster_models.py`
- Test script to verify models work correctly
- Tests both SentenceTransformers and EmbeddingConfig

### 4. Created `run_cluster_cross_model_analysis.sh`
- Shell script for easy execution
- Sets up environment variables
- Verifies model paths before running

### 5. Fixed `dataset_loader.py`
- Fixed JSON serialization issue with numpy types
- Applied `_convert_numpy_types` to statistics

## Usage Options

### Option 1: Direct Python Script (Recommended)
```bash
cd /scratch/user/u.bw269205/rag_reproducibility/RAG_Reproducibility

# Set offline mode
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Run analysis
python cross_model_retrieval_analysis.py \
    --models bge e5 qw \
    --top-k 50 \
    --max-docs 5000 \
    --max-queries 100 \
    --model-base-path /scratch/user/u.bw269205/shared_models
```

### Option 2: Using Cluster Wrapper
```bash
cd /scratch/user/u.bw269205/rag_reproducibility/RAG_Reproducibility
python cluster_cross_model_analysis.py --max-docs 5000 --max-queries 100
```

### Option 3: Using Shell Script
```bash
cd /scratch/user/u.bw269205/rag_reproducibility/RAG_Reproducibility
chmod +x run_cluster_cross_model_analysis.sh
./run_cluster_cross_model_analysis.sh
```

## Testing Models First

Before running the full analysis, test that your models work:

```bash
cd /scratch/user/u.bw269205/rag_reproducibility/RAG_Reproducibility

# Test model loading
python test_cluster_models.py --test-all

# Or just verify paths
python cluster_cross_model_analysis.py --verify-only
```

## Expected Output

The analysis will:
1. Load your MSMARCO dataset (or create synthetic data if not available)
2. Generate embeddings using all three models with your local paths
3. Build FAISS indices for retrieval
4. Perform top-50 retrieval for correlation analysis
5. Calculate four ranking correlation metrics:
   - Kendall's Tau
   - Rank-Biased Overlap (RBO)
   - Overlap Coefficient
   - Overlap Origin Analysis
6. Generate publication-ready visualizations with 30pt fonts
7. Save results in `results/` directory

## Troubleshooting

### If you get "Model not found" errors:
1. Verify your model paths exist:
   ```bash
   ls -la /scratch/user/u.bw269205/shared_models/
   ```

2. Check model directory contents:
   ```bash
   ls -la /scratch/user/u.bw269205/shared_models/bge_model/
   ls -la /scratch/user/u.bw269205/shared_models/intfloat_e5-base-v2/
   ls -la /scratch/user/u.bw269205/shared_models/Qwen_Qwen3-Embedding-0.6B/
   ```

### If you get import errors:
Make sure you have the required packages installed:
```bash
pip install sentence-transformers faiss-cpu numpy pandas matplotlib seaborn scipy torch
```

### If you get CUDA errors:
The script automatically detects CUDA availability. If you want to force CPU:
```bash
export CUDA_VISIBLE_DEVICES=""
python cross_model_retrieval_analysis.py --model-base-path /scratch/user/u.bw269205/shared_models
```

## Configuration Options

All scripts accept these parameters:
- `--models`: Which models to test (default: bge e5 qw)
- `--top-k`: Number of documents to retrieve (default: 50)
- `--max-docs`: Maximum documents to load (default: 5000)
- `--max-queries`: Maximum queries to generate (default: 100)
- `--model-base-path`: Base path for local models

## Next Steps

After the analysis completes:
1. Check the `results/` directory for output files
2. Review the correlation analysis plots
3. Examine the ranking consistency metrics
4. The results will show how consistently different models rank the same documents

The analysis focuses on cross-model ranking correlation rather than precision impact, giving you insights into embedding model consistency for retrieval tasks.
