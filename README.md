# RAG Reproducibility Testing Framework

A comprehensive framework for testing and analyzing reproducibility in Retrieval-Augmented Generation (RAG) systems, with integrated embedding uncertainty analysis and FAISS-based retrieval testing.

## 🎯 Overview

This framework provides **end-to-end reproducibility testing** for RAG systems, from embedding generation through final retrieval results. It combines advanced embedding uncertainty analysis with comprehensive FAISS index testing to identify and quantify sources of non-determinism.

## 🔬 Sources of Uncertainty in RAG Systems

### 1. Embedding Uncertainty (**NEW - Integrated from embedding_uncertainty/**)
- **Different embedding models**: Model architecture and training differences
- **Floating point precision**: FP16 vs FP32 vs BF16 vs TF32 computational variations
- **Hardware variations**: Different GPU architectures and drivers
- **Deterministic vs non-deterministic execution**: CUDA operation ordering effects
- **Model quantization**: Precision reduction impacts on embedding stability

### 2. Retrieval Uncertainty
- **Index uncertainty**: Different index types and parameters
- **Retrieval algorithm uncertainty**: KNN implementation variations
- **FAISS reproducibility**: CPU vs GPU versions, parallel execution effects
- **Hardware-specific optimizations**: CUDA operations and memory management

### 3. Distributed System Uncertainty
- **Sharding strategies**: Document distribution methods
- **Node synchronization**: Multi-node consistency challenges
- **Network effects**: Communication latency and ordering
- **Load balancing**: Dynamic resource allocation impacts

## 🚀 Quick Start

### Prerequisites
```bash
# Install core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install faiss-gpu  # or faiss-cpu for CPU-only
pip install sentence-transformers
pip install numpy pandas matplotlib seaborn scipy scikit-learn
pip install pyyaml tqdm

# Optional: For distributed testing
pip install mpi4py
```

### Run Quick Tests
```bash
# Quick diagnostic check
python quick_start.py --diagnostic

# Quick embedding reproducibility test
python quick_start.py --test embedding

# Quick retrieval reproducibility test
python quick_start.py --test retrieval

# Quick integrated test (embedding + retrieval)
python quick_start.py --test integrated

# All quick tests
python quick_start.py --test all
```

### Run Comprehensive Analysis
```bash
# Full test suite (may take 30-60 minutes)
python run_comprehensive_tests.py

# Quick mode (reduced datasets)
python run_comprehensive_tests.py --quick

# Skip specific test categories
python run_comprehensive_tests.py --skip-gpu --skip-distributed

# Custom output directory
python run_comprehensive_tests.py --output-dir my_results
```

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
