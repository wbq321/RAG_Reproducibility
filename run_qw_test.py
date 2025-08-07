#!/usr/bin/env python3
"""
Run reproducibility test for QW model
"""

from run_embedding_reproducibility import run_embedding_reproducibility

if __name__ == "__main__":
    # QW model test - UPDATE THE PATH TO YOUR QW MODEL
    run_embedding_reproducibility(
        model_name="qw", 
        model_path="/path/to/your/qw_model"  # UPDATE THIS PATH
    )
