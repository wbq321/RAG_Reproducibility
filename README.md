# RAG Reproducibility Testing Framework

A comprehensive framework for testing and analyzing the reproducibility of Retrieval-Augmented Generation (RAG) systems across different configurations and distributed computing environments.

## 🏗️ Project Structure

```
rag_reproducibility/
├── README.md                   # This file
├── setup-cluster-env.sh        # Environment setup script
│
├── src/                        # Core framework source code
│   ├── distributed-rag-cluster-test.py    # Distributed testing script
│   ├── rag_reproducibility_framework.py   # Main framework
│   └── RAG_reproducibility_test.py        # Test implementations
│
├── scripts/                    # Utility and analysis scripts
│   ├── generate-cluster-report.py         # Report generation
│   └── optimized_small_test.py           # Optimized testing script
│
├── slurm/                      # SLURM job scripts
│   ├── cluster-distributed-test.sh       # Main distributed test
│   ├── debug-quick-test.sh               # Debug test
│   ├── fixed_quick_test.sh               # Fixed configuration test
│   └── quick_test_slurm.sh               # Quick validation test
│
├── config/                     # Configuration files
│   └── cluster-config.json               # Main cluster configuration
│
├── results/                    # Test results and outputs
├── docs/                       # Documentation
├── examples/                   # Example usage and demos
├── notebooks/                  # Jupyter notebooks for analysis
└── tests/                      # Unit tests
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Set up conda environment with all dependencies
./setup-cluster-env.sh

# Activate the environment
conda activate rag_env
```

### 2. Run Tests
```bash
# Quick test (2 nodes, 30 minutes)
sbatch slurm/quick_test_slurm.sh

# Full distributed test
sbatch slurm/cluster-distributed-test.sh
```

### 3. Generate Reports
```bash
# Analyze results
python scripts/generate-cluster-report.py results/test_output_dir/
```

## 📋 Components

### Core Framework (`src/`)
- **`rag_reproducibility_framework.py`**: Main testing framework with FAISS integration
- **`distributed-rag-cluster-test.py`**: Distributed testing for HPC clusters
- **`RAG_reproducibility_test.py`**: Test implementations and metrics

### Scripts (`scripts/`)
- **`generate-cluster-report.py`**: Comprehensive analysis and visualization
- **`optimized_small_test.py`**: Fast testing for validation

### SLURM Jobs (`slurm/`)
- **`cluster-distributed-test.sh`**: Production distributed testing
- **`debug-quick-test.sh`**: Debugging and validation
- **`fixed_quick_test.sh`**: Fixed configuration testing
- **`quick_test_slurm.sh`**: Quick validation tests

## 🔧 Configuration

Edit `config/cluster-config.json` to adjust:
- FAISS index parameters
- GPU/CUDA settings
- Distributed computing options
- Test parameters

## 📊 Features

- **Multiple FAISS Index Types**: Flat, IVF, HNSW, LSH
- **Reproducibility Metrics**: Exact match, Jaccard similarity, rank correlation
- **Distributed Testing**: MPI-based multi-node execution
- **GPU Support**: CUDA-optimized vector operations
- **Comprehensive Reporting**: Detailed analysis with visualizations

## 📈 Metrics Analyzed

- **Exact Match Rate**: Identical results across runs
- **Jaccard Similarity**: Document overlap consistency
- **Kendall Tau**: Rank correlation analysis
- **Latency Stability**: Performance consistency
- **Score Stability**: Numerical precision analysis
- **Distributed Consistency**: Multi-node result agreement

## 🛠️ Dependencies

- Python 3.9+
- PyTorch with CUDA support
- FAISS (GPU-enabled)
- sentence-transformers
- mpi4py
- NumPy, SciPy, Pandas
- Matplotlib, Seaborn, Plotly

## 📝 Usage Examples

See the `examples/` directory for detailed usage examples and the `notebooks/` directory for interactive analysis examples.

## 📄 License

[Add your license information here]

## 🤝 Contributing

[Add contribution guidelines here]
