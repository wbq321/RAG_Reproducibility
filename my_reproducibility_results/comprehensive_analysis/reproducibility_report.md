# RAG Reproducibility Analysis Report

Generated: 2025-08-03 03:32:12

## Executive Summary

This report analyzes the reproducibility of RAG retrieval systems across various factors.

## Key Findings

- **Most reproducible**: Flat_L2 (Jaccard: 1.000)
- **Least reproducible**: None (Jaccard: 1.000)

## Detailed Results

### Index Type Analysis

| Index Type | Exact Match | Jaccard | Kendall Tau | Latency (ms) |
|------------|-------------|---------|-------------|-------------|
| Flat_L2 | 1.000 | 1.000 | 1.000 | 0.32 |
| Flat_IP | 1.000 | 1.000 | 1.000 | 0.31 |
| IVF_small | 1.000 | 1.000 | 1.000 | 0.05 |
| IVF_large | 1.000 | 1.000 | 1.000 | 0.08 |
| HNSW_fast | 1.000 | 1.000 | 1.000 | 0.06 |
| HNSW_accurate | 1.000 | 1.000 | 1.000 | 0.18 |
| LSH | 1.000 | 1.000 | 1.000 | 0.84 |

### GPU Non-determinism Factors

#### atomicAdd_order

- **deterministic_False**: Jaccard=1.000
- **deterministic_True**: Jaccard=1.000

#### parallel_reduction

- **batch_1**: Jaccard=1.000
- **batch_32**: Jaccard=1.000
- **batch_128**: Jaccard=1.000
- **batch_512**: Jaccard=1.000

#### multi_gpu_sync


#### mixed_precision


#### tensor_core_usage

- **dim_384**: Jaccard=0.007
- **dim_512**: Jaccard=0.007
- **dim_768**: Jaccard=0.002

## Recommendations

Based on the analysis, we recommend:

1. **For maximum reproducibility**: Use Flat index with deterministic mode enabled
2. **For production systems**: Consider HNSW with higher M and ef_search values
3. **For distributed systems**: Use hash-based sharding for consistent results
4. **GPU considerations**: Enable deterministic CUDA operations when reproducibility is critical


Full data saved to: my_reproducibility_results/comprehensive_analysis/full_analysis_data.json
