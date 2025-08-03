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
