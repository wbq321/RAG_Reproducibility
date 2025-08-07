#!/usr/bin/env python3
"""
Run reproducibility test for BGE model
"""

from run_embedding_reproducibility import run_embedding_reproducibility

if __name__ == "__main__":
    # BGE model test
    run_embedding_reproducibility(
        model_name="bge",
        model_path="/scratch/user/u.bw269205/shared_models/bge_model"
    )
