"""
distributed_rag_test.py - Main distributed test script for HPC clusters
Designed for SLURM-based supercomputing environments
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
import socket
import psutil

# MPI imports
from mpi4py import MPI

# Import the main framework components
from rag_reproducibility_framework import (
    ExperimentConfig,
    DistributedFaissRetrieval,
    ReproducibilityMetrics,
    RetrievalResult
)

# Configure MPI logging
def setup_mpi_logging(rank, output_dir):
    """Setup logging for MPI processes"""
    log_file = f"{output_dir}/rank_{rank}.log"
    logging.basicConfig(
        level=logging.INFO,
        format=f'[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


class ClusterEnvironmentInfo:
    """Collect and log cluster environment information"""
    
    def __init__(self, comm):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        
    def collect_node_info(self):
        """Collect information about the current node"""
        info = {
            "rank": self.rank,
            "hostname": socket.gethostname(),
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "gpu_info": self._get_gpu_info(),
            "slurm_info": self._get_slurm_info()
        }
        return info
    
    def _get_gpu_info(self):
        """Get GPU information using nvidia-smi"""
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                gpus = []
                for line in result.stdout.strip().split('\n'):
                    name, memory = line.split(', ')
                    gpus.append({"name": name, "memory": memory})
                return gpus
            return []
        except:
            return []
    
    def _get_slurm_info(self):
        """Get SLURM environment variables"""
        slurm_vars = {}
        for key, value in os.environ.items():
            if key.startswith('SLURM_'):
                slurm_vars[key] = value
        return slurm_vars
    
    def gather_cluster_info(self):
        """Gather information from all nodes"""
        local_info = self.collect_node_info()
        all_info = self.comm.gather(local_info, root=0)
        return all_info if self.rank == 0 else None


class DistributedRAGTester:
    """Main tester class for distributed RAG experiments on clusters"""
    
    def __init__(self, config_path, output_dir, comm):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = setup_mpi_logging(self.rank, output_dir)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config_dict = json.load(f)
        
        # Log environment info
        self.env_info = ClusterEnvironmentInfo(comm)
        if self.rank == 0:
            cluster_info = self.env_info.gather_cluster_info()
            with open(self.output_dir / "cluster_info.json", 'w') as f:
                json.dump(cluster_info, f, indent=2)
            self.logger.info(f"Cluster configuration saved. Total nodes: {len(cluster_info)}")
    
    def create_test_data(self, num_docs, num_queries):
        """Create test documents and queries"""
        # Use same seed on all ranks for consistency
        np.random.seed(42)
        
        # Only rank 0 creates the full dataset
        if self.rank == 0:
            self.logger.info(f"Creating {num_docs} documents and {num_queries} queries...")
            
            documents = []
            for i in range(num_docs):
                # Create more realistic documents
                topic = i % 1000
                subtopic = i % 100
                documents.append({
                    "id": f"doc_{i}",
                    "text": f"Document {i} discusses topic {topic} with focus on subtopic {subtopic}. "
                           f"This content includes keywords: {' '.join([f'keyword_{j}' for j in range(5)])}. "
                           f"Additional context about subject matter {i % 500}.",
                    "metadata": {
                        "category": f"cat_{topic % 50}",
                        "timestamp": i,
                        "node_assignment": i % self.size
                    }
                })
            
            queries = []
            for i in range(num_queries):
                topic = np.random.randint(0, 1000)
                queries.append(f"Find documents about topic {topic} with relevant keywords")
            
            return documents, queries
        else:
            return None, None
    
    def run_scaling_experiment(self, documents, queries, num_runs=5):
        """Test scalability with different numbers of nodes"""
        results = {}
        
        # Test with different subsets of nodes
        node_counts = [1, 2, 4, 8, 16, 32]
        
        for n_nodes in node_counts:
            if n_nodes > self.size:
                continue
            
            self.logger.info(f"Testing with {n_nodes} nodes...")
            
            # Create a sub-communicator with n_nodes processes
            color = 0 if self.rank < n_nodes else MPI.UNDEFINED
            sub_comm = self.comm.Split(color, self.rank)
            
            if sub_comm != MPI.COMM_NULL:
                # Run experiment on subset of nodes
                config = ExperimentConfig(
                    **self.config_dict,
                    distributed=True,
                    num_gpus=2 * n_nodes  # 2 GPUs per node
                )
                
                runs_results = []
                timings = []
                
                for run_idx in range(num_runs):
                    if self.rank == 0:
                        self.logger.info(f"  Run {run_idx + 1}/{num_runs}")
                    
                    # Create distributed retrieval system
                    dist_retrieval = DistributedFaissRetrieval(config)
                    dist_retrieval.comm = sub_comm  # Use sub-communicator
                    
                    # Time indexing
                    start_time = MPI.Wtime()
                    dist_retrieval.index_documents(documents)
                    index_time = MPI.Wtime() - start_time
                    
                    # Time search
                    start_time = MPI.Wtime()
                    search_results = dist_retrieval.search(queries)
                    search_time = MPI.Wtime() - start_time
                    
                    if self.rank == 0:
                        runs_results.append(search_results)
                        timings.append({
                            "index_time": index_time,
                            "search_time": search_time,
                            "total_time": index_time + search_time
                        })
                    
                    # Clean up
                    dist_retrieval.reset()
                    sub_comm.Barrier()
                
                # Calculate metrics on rank 0
                if self.rank == 0:
                    metrics = ReproducibilityMetrics.calculate_all_metrics(runs_results)
                    
                    results[f"nodes_{n_nodes}"] = {
                        "num_nodes": n_nodes,
                        "metrics": metrics,
                        "timings": timings,
                        "avg_index_time": np.mean([t["index_time"] for t in timings]),
                        "avg_search_time": np.mean([t["search_time"] for t in timings]),
                        "throughput_qps": len(queries) / np.mean([t["search_time"] for t in timings])
                    }
                
                sub_comm.Free()
            
            # Synchronize all processes
            self.comm.Barrier()
        
        return results if self.rank == 0 else None
    
    def run_shard_strategy_experiment(self, documents, queries, num_runs=5):
        """Test different sharding strategies"""
        results = {}
        
        shard_methods = ["hash", "range", "random"]
        
        for method in shard_methods:
            self.logger.info(f"Testing sharding method: {method}")
            
            config = ExperimentConfig(
                **self.config_dict,
                distributed=True,
                shard_method=method
            )
            
            runs_results = []
            
            for run_idx in range(num_runs):
                if self.rank == 0:
                    self.logger.info(f"  Run {run_idx + 1}/{num_runs}")
                
                dist_retrieval = DistributedFaissRetrieval(config)
                dist_retrieval.index_documents(documents)
                search_results = dist_retrieval.search(queries)
                
                if self.rank == 0:
                    runs_results.append(search_results)
                
                dist_retrieval.reset()
                self.comm.Barrier()
            
            if self.rank == 0:
                metrics = ReproducibilityMetrics.calculate_all_metrics(runs_results)
                results[method] = {
                    "metrics": metrics,
                    "config": config.to_dict()
                }
        
        return results if self.rank == 0 else None
    
    def run_fault_tolerance_experiment(self, documents, queries):
        """Test behavior under node failures"""
        if self.rank == 0:
            self.logger.info("Running fault tolerance experiments...")
        
        # This is a simulation - in real scenario, you'd actually kill processes
        results = {}
        
        # Simulate different failure scenarios
        failure_scenarios = [
            {"name": "no_failure", "failed_ranks": []},
            {"name": "single_node", "failed_ranks": [self.size - 1]},
            {"name": "multiple_nodes", "failed_ranks": [self.size - 1, self.size - 2]}
        ]
        
        for scenario in failure_scenarios:
            if self.rank == 0:
                self.logger.info(f"Testing scenario: {scenario['name']}")
            
            # Skip if this rank should "fail"
            if self.rank in scenario["failed_ranks"]:
                continue
            
            # Run normal test
            config = ExperimentConfig(
                **self.config_dict,
                distributed=True
            )
            
            try:
                dist_retrieval = DistributedFaissRetrieval(config)
                dist_retrieval.index_documents(documents)
                search_results = dist_retrieval.search(queries)
                
                if self.rank == 0:
                    results[scenario["name"]] = {
                        "success": True,
                        "num_results": len(search_results),
                        "failed_nodes": scenario["failed_ranks"]
                    }
            except Exception as e:
                if self.rank == 0:
                    results[scenario["name"]] = {
                        "success": False,
                        "error": str(e),
                        "failed_nodes": scenario["failed_ranks"]
                    }
            
            self.comm.Barrier()
        
        return results if self.rank == 0 else None
    
    def run_all_experiments(self, num_docs, num_queries, num_runs):
        """Run all distributed experiments"""
        
        # Create test data
        documents, queries = self.create_test_data(num_docs, num_queries)
        
        # Broadcast data to all ranks
        documents = self.comm.bcast(documents, root=0)
        queries = self.comm.bcast(queries, root=0)
        
        all_results = {}
        
        # 1. Scaling experiment
        if self.rank == 0:
            self.logger.info("\n" + "="*60)
            self.logger.info("EXPERIMENT 1: Scaling Analysis")
            self.logger.info("="*60)
        
        scaling_results = self.run_scaling_experiment(documents, queries, num_runs)
        if self.rank == 0:
            all_results["scaling"] = scaling_results
        
        # 2. Sharding strategy experiment
        if self.rank == 0:
            self.logger.info("\n" + "="*60)
            self.logger.info("EXPERIMENT 2: Sharding Strategies")
            self.logger.info("="*60)
        
        shard_results = self.run_shard_strategy_experiment(documents, queries, num_runs)
        if self.rank == 0:
            all_results["sharding"] = shard_results
        
        # 3. Fault tolerance experiment
        if self.rank == 0:
            self.logger.info("\n" + "="*60)
            self.logger.info("EXPERIMENT 3: Fault Tolerance")
            self.logger.info("="*60)
        
        fault_results = self.run_fault_tolerance_experiment(documents, queries)
        if self.rank == 0:
            all_results["fault_tolerance"] = fault_results
        
        # Save results
        if self.rank == 0:
            results_file = self.output_dir / "distributed_results.json"
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            
            self.logger.info(f"\nAll results saved to: {results_file}")
            
            # Generate summary
            self._generate_summary(all_results)
        
        return all_results
    
    def _generate_summary(self, results):
        """Generate a summary of the results"""
        summary_file = self.output_dir / "summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("DISTRIBUTED RAG REPRODUCIBILITY TEST SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            # Scaling results
            if "scaling" in results:
                f.write("1. SCALING ANALYSIS\n")
                f.write("-" * 40 + "\n")
                f.write(f"{'Nodes':<10} {'QPS':<15} {'Jaccard':<15} {'Exact Match':<15}\n")
                f.write("-" * 40 + "\n")
                
                for config, data in results["scaling"].items():
                    n_nodes = data["num_nodes"]
                    qps = data["throughput_qps"]
                    jaccard = data["metrics"]["overlap"]["mean_jaccard"]
                    exact = data["metrics"]["exact_match"]["exact_match_rate"]
                    
                    f.write(f"{n_nodes:<10} {qps:<15.2f} {jaccard:<15.3f} {exact:<15.3f}\n")
                
                f.write("\n")
            
            # Sharding results
            if "sharding" in results:
                f.write("2. SHARDING STRATEGIES\n")
                f.write("-" * 40 + "\n")
                f.write(f"{'Method':<15} {'Jaccard':<15} {'Kendall Tau':<15}\n")
                f.write("-" * 40 + "\n")
                
                for method, data in results["sharding"].items():
                    jaccard = data["metrics"]["overlap"]["mean_jaccard"]
                    kendall = data["metrics"]["rank_correlation"]["mean_kendall_tau"]
                    
                    f.write(f"{method:<15} {jaccard:<15.3f} {kendall:<15.3f}\n")
                
                f.write("\n")
            
            # Fault tolerance
            if "fault_tolerance" in results:
                f.write("3. FAULT TOLERANCE\n")
                f.write("-" * 40 + "\n")
                
                for scenario, data in results["fault_tolerance"].items():
                    f.write(f"Scenario: {scenario}\n")
                    f.write(f"  Success: {data['success']}\n")
                    f.write(f"  Failed nodes: {data['failed_nodes']}\n")
                    f.write("\n")
        
        self.logger.info(f"Summary saved to: {summary_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Distributed RAG Reproducibility Test")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num-docs", type=int, default=100000, help="Number of documents")
    parser.add_argument("--num-queries", type=int, default=1000, help="Number of queries")
    parser.add_argument("--num-runs", type=int, default=5, help="Number of runs per experiment")
    
    args = parser.parse_args()
    
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Create tester
    tester = DistributedRAGTester(args.config, args.output_dir, comm)
    
    # Run experiments
    results = tester.run_all_experiments(args.num_docs, args.num_queries, args.num_runs)
    
    # Finalize
    if rank == 0:
        print(f"\nAll experiments completed. Results saved to: {args.output_dir}")
    
    MPI.Finalize()


if __name__ == "__main__":
    main()
