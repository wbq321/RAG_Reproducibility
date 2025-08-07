#!/usr/bin/env python3
"""
Run reproducibility test for E5 model
"""

from run_embedding_reproducibility import run_embedding_reproducibility

if __name__ == "__main__":
    # E5 model test - UPDATE THE PATH TO YOUR E5 MODEL
    run_embedding_reproducibility(
        model_name="e5",
        model_path="/path/to/your/e5_model"  # UPDATE THIS PATH
    )
