# Cross-Model Retrieval Ranking Correlation Analysis

This script analyzes the ranking consistency across different embedding models (BGE, E5, QW) using FAISS retrieval as a downstream task. It focuses on comparing how different models rank the same documents for the same queries, rather than comparing precision impact.

## Features

- **FAISS Flat Index Retrieval**: Uses exact search for consistent comparison across models
- **MSMARCO Dataset Integration**: Leverages existing MSMARCO infrastructure for realistic evaluation
- **Top-50 Retrieval Analysis**: Focuses on top-50 retrieved documents as specified
- **Fixed Precision (FP32)**: Uses consistent FP32 precision to isolate model differences
- **Four Ranking Correlation Metrics**:
  1. **Kendall's Tau**: Measures rank correlation for common items
  2. **Rank-Biased Overlap (RBO)**: Weighted overlap focusing on top ranks
  3. **Overlap Coefficient**: Proportion of common items in top-K
  4. **Overlap Origin Analysis**: Where overlapping items come from (top-10, top-20, top-50)

## Usage

### Quick Start

```bash
python run_cross_model_analysis.py
```

### Advanced Usage

```bash
python cross_model_retrieval_analysis.py \\
    --models bge e5 qw \\
    --top-k 50 \\
    --max-docs 5000 \\
    --max-queries 100 \\
    --data-dir data \\
    --output-dir results
```

### Command Line Options

- `--models`: Models to test (default: bge e5 qw)
- `--top-k`: Number of top documents to retrieve (default: 50)
- `--max-docs`: Maximum documents to load (default: 5000)
- `--max-queries`: Maximum queries to generate (default: 100)
- `--data-dir`: Directory containing MSMARCO data (default: data)
- `--output-dir`: Output directory for results (default: results)

## Requirements

### Python Dependencies

```bash
pip install faiss-cpu scipy numpy pandas matplotlib seaborn
```

### Data Requirements

Place MSMARCO dataset in one of these locations:
- `data/ms_marco_passages.csv`
- `data/msmarco.csv`
- `data/passages.csv`

Download from: https://microsoft.github.io/msmarco/

## Output

The analysis generates:

### Visualizations
- `cross_model_ranking_correlations.png`: Publication-ready plots with 30pt fonts
- `correlation_matrix_heatmaps.png`: Correlation matrices as heatmaps

### Data Files
- `ranking_correlations.json`: Detailed correlation results
- `analysis_summary.json`: Summary statistics and configuration

### Log File
- `cross_model_retrieval_analysis.log`: Detailed execution log

## Analysis Overview

The script performs the following steps:

1. **Load MSMARCO Data**: Uses existing dataset loader infrastructure
2. **Generate Embeddings**: Creates FP32 embeddings for all models
3. **Build FAISS Indices**: Creates Flat indices for exact retrieval
4. **Perform Retrieval**: Retrieves top-50 documents for each query/model
5. **Calculate Correlations**: Computes four correlation metrics between model pairs
6. **Create Visualizations**: Generates publication-ready plots with 30pt fonts
7. **Save Results**: Exports detailed results and summary statistics

## Key Insights

The analysis focuses on:

- **Model Consistency**: How consistently different models rank the same documents
- **Top-K Overlap**: Which documents appear in top rankings across models
- **Ranking Stability**: Statistical significance of ranking differences
- **Precision Impact Isolation**: Uses fixed precision to focus on model differences

## Configuration

The analysis is designed to:
- Use fixed FP32 precision for all models
- Focus on top-50 retrieval as specified
- Test cross-model ranking correlation (not precision impact)
- Generate publication-ready visualizations with 30pt fonts
- Leverage existing MSMARCO infrastructure

## Integration

This script integrates with the existing RAG reproducibility framework:
- Uses `EmbeddingReproducibilityTester` for embedding generation
- Leverages `dataset_loader` for MSMARCO data handling
- Follows the established project structure and conventions
