#!/usr/bin/env python3
"""
Simple example of using the RAG reproducibility framework
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from rag_reproducibility_framework import (
    ExperimentConfig,
    FaissRetrieval,
    ReproducibilityMetrics
)

def main():
    """Run a simple reproducibility test"""
    
    # Create test documents
    documents = [
        {"id": f"doc_{i}", "text": f"Document {i} about topic {i % 5}"}
        for i in range(100)
    ]
    
    # Create test queries
    queries = [f"Find documents about topic {i}" for i in range(10)]
    
    # Test configuration
    config = ExperimentConfig(
        index_type="Flat",
        deterministic_mode=True,
        seed=42
    )
    
    print("Running reproducibility test...")
    
    # Run multiple times to test reproducibility
    runs = []
    for run_idx in range(3):
        print(f"Run {run_idx + 1}/3...")
        
        retrieval = FaissRetrieval(config)
        retrieval.index_documents(documents)
        results = retrieval.search(queries)
        runs.append(results)
        retrieval.reset()
    
    # Calculate metrics
    metrics = ReproducibilityMetrics.calculate_all_metrics(runs)
    
    # Print results
    print("\n" + "="*50)
    print("REPRODUCIBILITY RESULTS")
    print("="*50)
    print(f"Exact match rate: {metrics['exact_match']['exact_match_rate']:.3f}")
    print(f"Mean Jaccard similarity: {metrics['overlap']['mean_jaccard']:.3f}")
    print(f"Mean Kendall tau: {metrics['rank_correlation']['mean_kendall_tau']:.3f}")
    print(f"Mean latency: {metrics['latency']['mean_latency_ms']:.2f} ms")
    print("="*50)

if __name__ == "__main__":
    main()
