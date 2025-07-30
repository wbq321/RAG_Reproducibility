#!/usr/bin/env python
"""
optimized_small_test.py - Optimized version for faster testing
Reduces data size and number of runs for quick validation
"""

import os
import sys
import time
import json
import numpy as np
from datetime import datetime
from pathlib import Path
import logging

# MPI imports
from mpi4py import MPI

# Add the framework to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_reproducibility_framework import (
    ExperimentConfig,
    FaissRetrieval,
    ReproducibilityMetrics,
    RetrievalResult
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [Rank %(rank)d] - %(message)s',
    datefmt='%H:%M:%S'
)

class LoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return msg, {**kwargs, 'rank': self.extra['rank']}

def get_logger(rank):
    return LoggerAdapter(logging.getLogger(__name__), {'rank': rank})


class OptimizedDistributedTest:
    """Optimized test class for faster execution"""
    
    def __init__(self, output_dir):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = get_logger(self.rank)
        
        # Log start
        self.logger.info(f"Initialized on {self.size} ranks")
    
    def create_test_data(self, num_docs=1000, num_queries=10):
        """Create smaller test dataset"""
        if self.rank == 0:
            self.logger.info(f"Creating {num_docs} docs and {num_queries} queries")
            
            # Simple documents
            documents = []
            for i in range(num_docs):
                documents.append({
                    "id": f"doc_{i}",
                    "text": f"Document {i} about topic {i % 10}"
                })
            
            # Simple queries
            queries = [f"Query about topic {i % 10}" for i in range(num_queries)]
            
            return documents, queries
        return None, None
    
    def run_quick_test(self):
        """Run a quick reproducibility test"""
        
        # Create small dataset
        docs, queries = self.create_test_data(num_docs=1000, num_queries=10)
        
        # Broadcast to all ranks
        docs = self.comm.bcast(docs, root=0)
        queries = self.comm.bcast(queries, root=0)
        
        # Configuration for quick test
        config = ExperimentConfig(
            index_type="Flat",  # Fastest index type
            use_gpu=True,
            batch_size=100,
            deterministic_mode=True,
            seed=42
        )
        
        # Run 3 times for reproducibility check
        all_results = []
        
        for run_idx in range(3):
            if self.rank == 0:
                self.logger.info(f"Run {run_idx + 1}/3")
            
            # Time the operations
            start_time = MPI.Wtime()
            
            # Local indexing (each rank indexes subset)
            local_docs = self._get_local_subset(docs)
            
            retrieval = FaissRetrieval(config)
            retrieval.index_documents(local_docs)
            
            # Search
            local_results = retrieval.search(queries)
            
            # Gather results
            all_local_results = self.comm.gather(local_results, root=0)
            
            if self.rank == 0:
                # Aggregate results
                aggregated = self._aggregate_results(all_local_results, len(queries))
                all_results.append(aggregated)
                
                elapsed = MPI.Wtime() - start_time
                self.logger.info(f"Run {run_idx + 1} completed in {elapsed:.2f}s")
            
            # Clean up
            retrieval.reset()
            self.comm.Barrier()
        
        # Calculate metrics on rank 0
        if self.rank == 0:
            self._calculate_and_save_metrics(all_results)
    
    def _get_local_subset(self, docs):
        """Get subset of documents for this rank"""
        n_docs = len(docs)
        docs_per_rank = n_docs // self.size
        start_idx = self.rank * docs_per_rank
        end_idx = start_idx + docs_per_rank if self.rank < self.size - 1 else n_docs
        return docs[start_idx:end_idx]
    
    def _aggregate_results(self, all_results, n_queries):
        """Simple aggregation of results"""
        aggregated = []
        
        for q_idx in range(n_queries):
            # Collect results for this query from all ranks
            combined_docs = []
            combined_scores = []
            
            for rank_results in all_results:
                if rank_results and q_idx < len(rank_results):
                    result = rank_results[q_idx]
                    combined_docs.extend(result.doc_ids)
                    combined_scores.extend(result.scores)
            
            # Sort and take top-k
            if combined_docs:
                sorted_pairs = sorted(zip(combined_scores, combined_docs))[:10]
                top_scores, top_docs = zip(*sorted_pairs)
                
                aggregated.append(RetrievalResult(
                    query_id=f"q_{q_idx}",
                    doc_ids=list(top_docs),
                    scores=list(top_scores),
                    latency_ms=0,
                    metadata={}
                ))
        
        return aggregated
    
    def _calculate_and_save_metrics(self, runs):
        """Calculate and save reproducibility metrics"""
        
        metrics = {
            "exact_match": ReproducibilityMetrics.exact_match_rate(runs),
            "overlap": ReproducibilityMetrics.top_k_overlap(runs),
            "rank_correlation": ReproducibilityMetrics.rank_correlation(runs)
        }
        
        # Print summary
        self.logger.info("\n" + "="*50)
        self.logger.info("REPRODUCIBILITY METRICS")
        self.logger.info("="*50)
        self.logger.info(f"Exact Match Rate: {metrics['exact_match']['exact_match_rate']:.3f}")
        self.logger.info(f"Jaccard Similarity: {metrics['overlap']['mean_jaccard']:.3f}")
        self.logger.info(f"Kendall Tau: {metrics['rank_correlation']['mean_kendall_tau']:.3f}")
        
        # Save results
        results = {
            "timestamp": datetime.now().isoformat(),
            "num_ranks": self.size,
            "metrics": metrics
        }
        
        with open(self.output_dir / "quick_test_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"\nResults saved to: {self.output_dir}")


def main():
    """Main entry point"""
    
    # Simple argument handling
    output_dir = sys.argv[1] if len(sys.argv) > 1 else f"quick_test_{int(time.time())}"
    
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print(f"\nStarting optimized test with {comm.Get_size()} processes")
        print(f"Output directory: {output_dir}\n")
    
    # Run test
    tester = OptimizedDistributedTest(output_dir)
    tester.run_quick_test()
    
    # Finalize
    if rank == 0:
        print("\nTest completed successfully!")
    
    MPI.Finalize()


if __name__ == "__main__":
    main()
