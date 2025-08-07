#!/usr/bin/env python3
"""
Cluster Configuration for Cross-Model Retrieval Analysis
Sets up the correct paths for the cluster environment without internet access
"""

import os
import sys
from pathlib import Path

# Cluster-specific configuration
CLUSTER_MODEL_BASE_PATH = "/scratch/user/u.bw269205/shared_models"
CLUSTER_PROJECT_ROOT = "/scratch/user/u.bw269205/rag_reproducibility/RAG_Reproducibility"

# Model path mapping for cluster
CLUSTER_MODEL_PATHS = {
    "bge": f"{CLUSTER_MODEL_BASE_PATH}/bge_model",
    "e5": f"{CLUSTER_MODEL_BASE_PATH}/intfloat_e5-base-v2", 
    "qw": f"{CLUSTER_MODEL_BASE_PATH}/Qwen_Qwen3-Embedding-0.6B"
}

def verify_model_paths():
    """Verify that all model paths exist on the cluster"""
    print("Verifying model paths on cluster...")
    all_exist = True
    
    for model_name, model_path in CLUSTER_MODEL_PATHS.items():
        if os.path.exists(model_path):
            print(f"✓ {model_name}: {model_path}")
        else:
            print(f"✗ {model_name}: {model_path} (NOT FOUND)")
            all_exist = False
    
    return all_exist

def run_cross_model_analysis(max_docs=5000, max_queries=100):
    """Run the cross-model analysis with cluster-specific paths"""
    
    # Verify paths first
    if not verify_model_paths():
        print("ERROR: Some model paths are missing. Please check your model installation.")
        return False
    
    # Change to project directory
    os.chdir(CLUSTER_PROJECT_ROOT)
    
    # Set environment for offline mode
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    
    # Import and run analysis
    try:
        sys.path.append(str(Path(CLUSTER_PROJECT_ROOT)))
        
        from cross_model_retrieval_analysis import CrossModelRetrievalAnalyzer
        
        # Initialize analyzer with cluster paths
        analyzer = CrossModelRetrievalAnalyzer(
            models_to_test=["bge", "e5", "qw"],
            top_k=50,
            data_dir="data",
            output_dir="results",
            model_base_path=CLUSTER_MODEL_BASE_PATH
        )
        
        # Run complete analysis
        print(f"Starting cross-model analysis with {max_docs} docs and {max_queries} queries...")
        analyzer.run_complete_analysis(
            max_documents=max_docs,
            max_queries=max_queries
        )
        
        print("Cross-model analysis completed successfully!")
        return True
        
    except Exception as e:
        print(f"ERROR: Cross-model analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run cross-model analysis on cluster")
    parser.add_argument("--max-docs", type=int, default=5000,
                       help="Maximum documents to process (default: 5000)")
    parser.add_argument("--max-queries", type=int, default=100, 
                       help="Maximum queries to generate (default: 100)")
    parser.add_argument("--verify-only", action="store_true",
                       help="Only verify model paths, don't run analysis")
    
    args = parser.parse_args()
    
    if args.verify_only:
        verify_model_paths()
    else:
        success = run_cross_model_analysis(args.max_docs, args.max_queries)
        sys.exit(0 if success else 1)
