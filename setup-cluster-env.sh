#!/bin/bash
# setup_cluster_env.sh - Setup script for RAG reproducibility testing on HPC cluster

echo "Setting up RAG reproducibility testing environment..."

# Load conda module (adjust module name based on your cluster)
module load conda
# Alternative: module load anaconda3 or module load miniconda3

# Create conda environment for RAG reproducibility testing
conda create -n rag_env python=3.9 -y
conda activate rag_env

# Install packages using conda first (for better dependency resolution)
conda install -c conda-forge numpy scipy pandas matplotlib seaborn -y
conda install -c conda-forge scikit-learn psutil -y
conda install -c plotly plotly -y

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install FAISS with GPU support
conda install -c conda-forge faiss-gpu -y

# Install remaining packages with pip (if not available in conda)
pip install sentence-transformers
pip install mpi4py

# Create environment file for reproducibility
conda env export > environment.yml
conda list --export > requirements.txt

echo "Setup complete!"
echo ""
echo "Conda environment 'rag_env' created with all dependencies."
echo ""
echo "Environment files created:"
echo "  - environment.yml (for recreating environment)"
echo "  - requirements.txt (package list)"
echo ""
echo "To activate the environment:"
echo "conda activate rag_env"