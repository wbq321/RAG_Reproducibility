# Integrated RAG Reproducibility Analysis Report

This report analyzes reproducibility from embedding generation through retrieval results.

## Executive Summary

**Most stable embedding configuration**: fp32_True (L2 distance: 0.00e+00)

## Embedding Stability Analysis

| Configuration | L2 Distance | Cosine Similarity | Max Abs Diff | Exact Match Rate |
|---------------|-------------|-------------------|--------------|------------------|
| fp32_True | 0.00e+00 | 1.000000 | 0.00e+00 | 1.000 |
| fp32_False | 0.00e+00 | 1.000000 | 0.00e+00 | 1.000 |
| fp16_True | 0.00e+00 | 1.000000 | 0.00e+00 | 1.000 |
| fp16_False | 0.00e+00 | 1.000000 | 0.00e+00 | 1.000 |
| tf32_True | 0.00e+00 | 1.000000 | 0.00e+00 | 1.000 |
| tf32_False | 0.00e+00 | 1.000000 | 0.00e+00 | 1.000 |
| bf16_True | 0.00e+00 | 1.000000 | 0.00e+00 | 1.000 |
| bf16_False | 0.00e+00 | 1.000000 | 0.00e+00 | 1.000 |

## Retrieval Reproducibility Analysis

| Configuration | Exact Match | Jaccard | Kendall Tau |
|---------------|-------------|---------|-------------|
| fp32_True | 1.000 | 1.000 | 1.000 |
| fp32_False | 1.000 | 1.000 | 1.000 |
| fp16_True | 1.000 | 1.000 | 1.000 |
| fp16_False | 1.000 | 1.000 | 1.000 |
| tf32_True | 1.000 | 1.000 | 1.000 |
| tf32_False | 1.000 | 1.000 | 1.000 |
| bf16_True | 1.000 | 1.000 | 1.000 |
| bf16_False | 1.000 | 1.000 | 1.000 |

## Cross-Configuration Analysis

### Precision Effects

**TF32**:
- Embedding L2 distance: 0.00e+00
- Retrieval Jaccard: 1.000

**FP16**:
- Embedding L2 distance: 0.00e+00
- Retrieval Jaccard: 1.000

**FP32**:
- Embedding L2 distance: 0.00e+00
- Retrieval Jaccard: 1.000

**BF16**:
- Embedding L2 distance: 0.00e+00
- Retrieval Jaccard: 1.000

### Deterministic vs Non-Deterministic

**Deterministic**:
- Embedding L2 distance: 0.00e+00
- Retrieval Jaccard: 1.000

**Non Deterministic**:
- Embedding L2 distance: 0.00e+00
- Retrieval Jaccard: 1.000

## Recommendations

Based on the integrated analysis:

1. **For maximum stability**: Use FP32 precision with deterministic mode
2. **For performance**: FP16 may be acceptable if stability requirements are moderate
3. **For production**: Monitor embedding drift alongside retrieval metrics
4. **For reproducibility**: Enable deterministic mode at both embedding and retrieval levels
