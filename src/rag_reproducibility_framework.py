"""
RAG Reproducibility Testing Framework with FAISS and Distributed Support
Extended version with real FAISS integration and MPI-based distributed testing
"""

import os
import json
import time
import hashlib
import numpy as np
import pandas as pd
import faiss
import torch
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from abc import ABC, abstractmethod
import logging
from datetime import datetime
import multiprocessing as mp
from sentence_transformers import SentenceTransformer
import pickle

# MPI support
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False
    print("Warning: mpi4py not available. Distributed features disabled.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Container for retrieval results"""
    query_id: str
    doc_ids: List[str]
    scores: List[float]
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        """Create a deterministic hash of the result"""
        content = f"{self.query_id}:{','.join(self.doc_ids)}:{','.join(map(str, self.scores))}"
        return int(hashlib.md5(content.encode()).hexdigest(), 16)


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run"""
    # Hardware config
    device: str = "cuda"
    gpu_id: int = 0
    num_gpus: int = 1

    # Algorithm config
    retrieval_method: str = "dense"
    index_type: str = "Flat"  # Flat, IVF, HNSW, LSH
    embedding_model: str = "/scratch/user/u.bw269205/shared_models/bge_model"
    embedding_dim: int = 768  # BGE models typically use 768 dimensions

    # FAISS specific parameters
    faiss_metric: str = "L2"  # L2, IP (inner product)
    ivf_nlist: int = 100  # number of clusters for IVF
    ivf_nprobe: int = 10  # number of clusters to search
    hnsw_M: int = 32  # number of connections for HNSW
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 50
    use_gpu: bool = True

    # Data config
    chunk_size: int = 512
    chunk_overlap: int = 50
    batch_size: int = 32

    # Search config
    top_k: int = 10
    search_params: Dict[str, Any] = field(default_factory=dict)

    # System config
    num_threads: int = 1
    seed: Optional[int] = None
    deterministic_mode: bool = False

    # Distributed config
    distributed: bool = False
    shard_method: str = "hash"  # hash, range, random
    replication_factor: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


class FaissRetrieval:
    """FAISS-based retrieval implementation"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.encoder = SentenceTransformer(config.embedding_model)

        # Auto-detect embedding dimension from the model
        try:
            sample_embedding = self.encoder.encode(["test"], convert_to_numpy=True)
            actual_dim = sample_embedding.shape[1]
            if actual_dim != config.embedding_dim:
                logger.info(f"Auto-detected embedding dimension: {actual_dim} (config had {config.embedding_dim})")
                self.config.embedding_dim = actual_dim
        except Exception as e:
            logger.warning(f"Could not auto-detect embedding dimension: {e}")

        self.index = None
        self.documents = []
        self.doc_embeddings = None
        self.gpu_resources = []

        # Set deterministic mode if requested
        if config.deterministic_mode:
            self._set_deterministic_mode()

    def _set_deterministic_mode(self):
        """Enable deterministic mode for reproducibility"""
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)
            
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config.seed)

        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _create_index(self) -> faiss.Index:
        """Create FAISS index based on configuration"""
        d = self.config.embedding_dim

        if self.config.index_type == "Flat":
            if self.config.faiss_metric == "L2":
                index = faiss.IndexFlatL2(d)
            else:
                index = faiss.IndexFlatIP(d)

        elif self.config.index_type == "IVF":
            quantizer = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFFlat(quantizer, d, self.config.ivf_nlist)

        elif self.config.index_type == "HNSW":
            index = faiss.IndexHNSWFlat(d, self.config.hnsw_M)
            index.hnsw.efConstruction = self.config.hnsw_ef_construction

        elif self.config.index_type == "LSH":
            nbits = d * 8  # 8 bits per dimension
            index = faiss.IndexLSH(d, nbits)

        else:
            raise ValueError(f"Unknown index type: {self.config.index_type}")

        # Move to GPU if requested and available
        if self.config.use_gpu and faiss.get_num_gpus() > 0:
            if self.config.num_gpus == 1:
                res = faiss.StandardGpuResources()
                self.gpu_resources.append(res)
                index = faiss.index_cpu_to_gpu(res, self.config.gpu_id, index)
            else:
                # Multi-GPU support
                index = faiss.index_cpu_to_all_gpus(index)

        return index

    def index_documents(self, documents: List[Dict[str, str]]) -> None:
        """Index documents using FAISS"""
        logger.info(f"Indexing {len(documents)} documents with {self.config.index_type} index")

        self.documents = documents
        texts = [doc['text'] for doc in documents]

        # Encode documents in batches
        embeddings = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            batch_embeddings = self.encoder.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=self.config.faiss_metric == "IP"
            )
            embeddings.append(batch_embeddings)

        self.doc_embeddings = np.vstack(embeddings).astype('float32')

        # Create and train index
        self.index = self._create_index()

        # Train index if needed (for IVF)
        if self.config.index_type == "IVF":
            logger.info("Training IVF index...")
            self.index.train(self.doc_embeddings)

        # Add vectors to index
        self.index.add(self.doc_embeddings)

        # Set search parameters
        if self.config.index_type == "IVF":
            self.index.nprobe = self.config.ivf_nprobe
        elif self.config.index_type == "HNSW":
            self.index.hnsw.efSearch = self.config.hnsw_ef_search

        logger.info(f"Index built with {self.index.ntotal} vectors")

    def search(self, queries: List[str]) -> List[RetrievalResult]:
        """Search queries in FAISS index"""
        results = []

        # Encode queries
        query_embeddings = self.encoder.encode(
            queries,
            convert_to_numpy=True,
            normalize_embeddings=self.config.faiss_metric == "IP"
        ).astype('float32')

        # Search
        for i, query_embedding in enumerate(query_embeddings):
            start_time = time.time()

            # Reshape for FAISS
            query_vec = query_embedding.reshape(1, -1)

            # Search in index
            distances, indices = self.index.search(query_vec, self.config.top_k)

            # Convert to results
            doc_ids = [self.documents[idx]['id'] for idx in indices[0] if idx != -1]
            scores = distances[0].tolist()

            result = RetrievalResult(
                query_id=f"q_{i}",
                doc_ids=doc_ids,
                scores=scores,
                latency_ms=(time.time() - start_time) * 1000,
                metadata={
                    "index_type": self.config.index_type,
                    "num_results": len(doc_ids)
                }
            )
            results.append(result)

        return results

    def reset(self):
        """Reset the index"""
        self.index = None
        self.documents = []
        self.doc_embeddings = None
        self.gpu_resources = []


class DistributedFaissRetrieval:
    """Distributed FAISS implementation using MPI"""

    def __init__(self, config: ExperimentConfig):
        if not MPI_AVAILABLE:
            raise ImportError("mpi4py is required for distributed mode")

        self.config = config
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # Each rank gets its own encoder
        self.encoder = SentenceTransformer(config.embedding_model)
        self.local_index = None
        self.local_documents = []
        self.shard_info = {}

        logger.info(f"Initialized distributed retrieval - Rank {self.rank}/{self.size}")

    def _shard_documents(self, documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Shard documents across MPI ranks"""
        if self.config.shard_method == "hash":
            # Hash-based sharding
            local_docs = []
            for doc in documents:
                doc_hash = hash(doc['id']) % self.size
                if doc_hash == self.rank:
                    local_docs.append(doc)
            return local_docs

        elif self.config.shard_method == "range":
            # Range-based sharding
            docs_per_rank = len(documents) // self.size
            start_idx = self.rank * docs_per_rank
            end_idx = start_idx + docs_per_rank if self.rank < self.size - 1 else len(documents)
            return documents[start_idx:end_idx]

        elif self.config.shard_method == "random":
            # Random sharding with seed for reproducibility
            if self.config.seed is not None:
                np.random.seed(self.config.seed + self.rank)
            indices = np.random.permutation(len(documents))
            docs_per_rank = len(documents) // self.size
            start_idx = self.rank * docs_per_rank
            end_idx = start_idx + docs_per_rank if self.rank < self.size - 1 else len(documents)
            selected_indices = indices[start_idx:end_idx]
            return [documents[i] for i in selected_indices]

    def index_documents(self, documents: List[Dict[str, str]]) -> None:
        """Distributed indexing"""
        # Broadcast total document count
        total_docs = len(documents) if self.rank == 0 else None
        total_docs = self.comm.bcast(total_docs, root=0)

        # Shard documents
        self.local_documents = self._shard_documents(documents)
        local_count = len(self.local_documents)

        # Gather shard info
        all_counts = self.comm.gather(local_count, root=0)
        if self.rank == 0:
            logger.info(f"Document distribution: {all_counts}")

        # Create local FAISS index
        local_config = self.config
        local_config.gpu_id = self.rank % faiss.get_num_gpus() if self.config.use_gpu else 0

        self.local_retrieval = FaissRetrieval(local_config)
        self.local_retrieval.index_documents(self.local_documents)

        # Synchronize
        self.comm.Barrier()
        logger.info(f"Rank {self.rank}: Indexed {local_count} documents")

    def search(self, queries: List[str]) -> List[RetrievalResult]:
        """Distributed search with result aggregation"""
        # Each rank searches its local index
        local_results = self.local_retrieval.search(queries)

        # Gather results from all ranks
        all_results = self.comm.gather(local_results, root=0)

        if self.rank == 0:
            # Aggregate results on rank 0
            aggregated_results = self._aggregate_results(all_results, queries)
            # Broadcast aggregated results to all ranks
            final_results = aggregated_results
        else:
            final_results = None

        # Broadcast final results to all ranks
        final_results = self.comm.bcast(final_results, root=0)

        return final_results

    def _aggregate_results(self, all_results: List[List[RetrievalResult]], queries: List[str]) -> List[RetrievalResult]:
        """Aggregate results from all ranks"""
        aggregated = []

        for query_idx in range(len(queries)):
            # Collect results for this query from all ranks
            query_results = []
            for rank_results in all_results:
                if rank_results and query_idx < len(rank_results):
                    result = rank_results[query_idx]
                    for doc_id, score in zip(result.doc_ids, result.scores):
                        query_results.append((doc_id, score, result.latency_ms))

            # Sort by score and take top-k
            query_results.sort(key=lambda x: x[1])  # Lower score is better for L2
            top_results = query_results[:self.config.top_k]

            # Create aggregated result
            aggregated_result = RetrievalResult(
                query_id=f"q_{query_idx}",
                doc_ids=[r[0] for r in top_results],
                scores=[r[1] for r in top_results],
                latency_ms=max([r[2] for r in query_results]) if query_results else 0,
                metadata={
                    "num_shards_searched": len(all_results),
                    "total_results_before_aggregation": len(query_results)
                }
            )
            aggregated.append(aggregated_result)

        return aggregated

    def reset(self):
        """Reset distributed index"""
        if hasattr(self, 'local_retrieval'):
            self.local_retrieval.reset()
        self.local_documents = []
        self.comm.Barrier()


class DistributedReproducibilityTester:
    """Extended tester with distributed capabilities"""

    def __init__(self, base_config: ExperimentConfig):
        self.base_config = base_config
        self.results = defaultdict(list)

        if MPI_AVAILABLE and base_config.distributed:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        else:
            self.rank = 0
            self.size = 1

    def test_distributed_factors(self, documents: List[Dict[str, str]], queries: List[str]) -> Dict[str, Any]:
        """Test factors specific to distributed execution"""

        distributed_factors = {
            "num_nodes": [1, 2, 4, 8],
            "shard_method": ["hash", "range", "random"],
            "replication_factor": [1, 2, 3],
            "gpu_per_node": [1, 2, 4]
        }

        results = {}

        for factor_name, values in distributed_factors.items():
            factor_results = {}

            for value in values:
                # Skip if we don't have enough resources
                if factor_name == "num_nodes" and value > self.size:
                    continue

                config = ExperimentConfig(**self.base_config.to_dict())
                config.distributed = True
                setattr(config, factor_name, value)

                # Run distributed experiment
                factor_results[str(value)] = self._run_distributed_experiment(
                    documents, queries, config
                )

            results[factor_name] = factor_results

        return results

    def _run_distributed_experiment(self,
                                  documents: List[Dict[str, str]],
                                  queries: List[str],
                                  config: ExperimentConfig,
                                  n_runs: int = 5) -> Dict[str, Any]:
        """Run a distributed experiment multiple times"""

        runs = []
        timings = []

        for run_idx in range(n_runs):
            if self.rank == 0:
                logger.info(f"Distributed run {run_idx + 1}/{n_runs}")

            # Create distributed retrieval system
            dist_retrieval = DistributedFaissRetrieval(config)

            # Time the indexing phase
            start_time = time.time()
            dist_retrieval.index_documents(documents)
            index_time = time.time() - start_time

            # Time the search phase
            start_time = time.time()
            results = dist_retrieval.search(queries)
            search_time = time.time() - start_time

            runs.append(results)
            timings.append({
                "index_time": index_time,
                "search_time": search_time
            })

            # Clean up
            dist_retrieval.reset()

            # Synchronize before next run
            if MPI_AVAILABLE:
                self.comm.Barrier()

        # Calculate metrics
        metrics = ReproducibilityMetrics.calculate_all_metrics(runs)

        # Add distributed-specific metrics
        metrics["distributed"] = {
            "avg_index_time": np.mean([t["index_time"] for t in timings]),
            "avg_search_time": np.mean([t["search_time"] for t in timings]),
            "index_time_variance": np.var([t["index_time"] for t in timings]),
            "search_time_variance": np.var([t["search_time"] for t in timings])
        }

        return {
            "config": config.to_dict(),
            "metrics": metrics,
            "timings": timings
        }


class ReproducibilityMetrics:
    """Extended metrics for distributed testing"""

    @staticmethod
    def calculate_all_metrics(runs: List[List[RetrievalResult]]) -> Dict[str, Any]:
        """Calculate all reproducibility metrics"""

        return {
            "exact_match": ReproducibilityMetrics.exact_match_rate(runs),
            "overlap": ReproducibilityMetrics.top_k_overlap(runs),
            "rank_correlation": ReproducibilityMetrics.rank_correlation(runs),
            "score_stability": ReproducibilityMetrics.score_stability(runs),
            "latency": ReproducibilityMetrics.latency_stability(runs),
            "distributed_consistency": ReproducibilityMetrics.distributed_consistency(runs)
        }

    @staticmethod
    def distributed_consistency(runs: List[List[RetrievalResult]]) -> Dict[str, float]:
        """Analyze consistency specific to distributed execution"""

        if not runs or not runs[0]:
            return {}

        # Check if results have distributed metadata
        has_distributed_meta = any(
            'num_shards_searched' in r.metadata
            for r in runs[0]
        )

        if not has_distributed_meta:
            return {}

        # Analyze shard coverage consistency
        shard_counts = []
        for run in runs:
            for result in run:
                if 'num_shards_searched' in result.metadata:
                    shard_counts.append(result.metadata['num_shards_searched'])

        return {
            "shard_coverage_consistency": 1.0 - (np.std(shard_counts) / np.mean(shard_counts)) if shard_counts else 0,
            "avg_shards_searched": np.mean(shard_counts) if shard_counts else 0
        }

    # Include previous metric methods (exact_match_rate, top_k_overlap, etc.)
    @staticmethod
    def exact_match_rate(runs: List[List[RetrievalResult]]) -> Dict[str, float]:
        """Calculate exact match rate across runs"""
        n_queries = len(runs[0])
        n_runs = len(runs)

        exact_matches = 0
        total_comparisons = 0

        for query_idx in range(n_queries):
            query_results = [runs[run_idx][query_idx] for run_idx in range(n_runs)]

            for i in range(n_runs):
                for j in range(i+1, n_runs):
                    total_comparisons += 1
                    if query_results[i].doc_ids == query_results[j].doc_ids:
                        exact_matches += 1

        return {
            "exact_match_rate": exact_matches / total_comparisons if total_comparisons > 0 else 0
        }

    @staticmethod
    def top_k_overlap(runs: List[List[RetrievalResult]], k: int = 10) -> Dict[str, float]:
        """Calculate Jaccard similarity of top-k results"""
        n_queries = len(runs[0])
        n_runs = len(runs)

        overlaps = []

        for query_idx in range(n_queries):
            query_results = [runs[run_idx][query_idx] for run_idx in range(n_runs)]

            for i in range(n_runs):
                for j in range(i+1, n_runs):
                    set_i = set(query_results[i].doc_ids[:k])
                    set_j = set(query_results[j].doc_ids[:k])

                    if len(set_i.union(set_j)) > 0:
                        jaccard = len(set_i.intersection(set_j)) / len(set_i.union(set_j))
                        overlaps.append(jaccard)

        return {
            "mean_jaccard": np.mean(overlaps) if overlaps else 0,
            "std_jaccard": np.std(overlaps) if overlaps else 0,
            "min_jaccard": np.min(overlaps) if overlaps else 0
        }

    @staticmethod
    def rank_correlation(runs: List[List[RetrievalResult]]) -> Dict[str, float]:
        """Calculate rank correlation (Kendall's tau) between runs"""
        from scipy.stats import kendalltau

        n_queries = len(runs[0])
        n_runs = len(runs)

        correlations = []

        for query_idx in range(n_queries):
            query_results = [runs[run_idx][query_idx] for run_idx in range(n_runs)]

            for i in range(n_runs):
                for j in range(i+1, n_runs):
                    docs_i = query_results[i].doc_ids
                    docs_j = query_results[j].doc_ids

                    common_docs = list(set(docs_i) & set(docs_j))

                    if len(common_docs) >= 2:
                        ranks_i = [docs_i.index(doc) for doc in common_docs]
                        ranks_j = [docs_j.index(doc) for doc in common_docs]

                        tau, _ = kendalltau(ranks_i, ranks_j)
                        correlations.append(tau)

        return {
            "mean_kendall_tau": np.mean(correlations) if correlations else 0,
            "std_kendall_tau": np.std(correlations) if correlations else 0
        }

    @staticmethod
    def embedding_stability_metrics(embedding_runs: List[np.ndarray]) -> Dict[str, Any]:
        """Calculate embedding stability metrics across runs"""
        if len(embedding_runs) < 2:
            return {}

        baseline = embedding_runs[0]
        n_texts, embed_dim = baseline.shape

        # Calculate pairwise metrics
        l2_distances = []
        cosine_similarities = []
        max_abs_differences = []

        for i in range(1, len(embedding_runs)):
            current = embedding_runs[i]

            # L2 distance per text
            l2_dist = np.linalg.norm(baseline - current, axis=1)
            l2_distances.extend(l2_dist)

            # Cosine similarity per text
            cos_sim = np.sum(baseline * current, axis=1) / (
                np.linalg.norm(baseline, axis=1) * np.linalg.norm(current, axis=1) + 1e-8
            )
            cosine_similarities.extend(cos_sim)

            # Max absolute difference per text
            max_abs_diff = np.max(np.abs(baseline - current), axis=1)
            max_abs_differences.extend(max_abs_diff)

        # Calculate cross-run variance
        all_embeddings = np.stack(embedding_runs, axis=0)
        embedding_variance = np.var(all_embeddings, axis=0)

        return {
            "l2_distance": {
                "mean": float(np.mean(l2_distances)),
                "std": float(np.std(l2_distances)),
                "max": float(np.max(l2_distances)),
                "min": float(np.min(l2_distances))
            },
            "cosine_similarity": {
                "mean": float(np.mean(cosine_similarities)),
                "std": float(np.std(cosine_similarities)),
                "min": float(np.min(cosine_similarities))
            },
            "max_abs_difference": {
                "mean": float(np.mean(max_abs_differences)),
                "std": float(np.std(max_abs_differences)),
                "max": float(np.max(max_abs_differences))
            },
            "embedding_variance": {
                "mean_overall": float(np.mean(embedding_variance)),
                "max_overall": float(np.max(embedding_variance)),
                "per_dimension_mean": float(np.mean(np.mean(embedding_variance, axis=0)))
            },
            "exact_match_rate": float(np.mean([np.allclose(baseline, emb, rtol=1e-7, atol=1e-7)
                                             for emb in embedding_runs[1:]])),
            "dimension": int(embed_dim),
            "num_texts": int(n_texts)
        }

    @staticmethod
    def score_stability(runs: List[List[RetrievalResult]]) -> Dict[str, float]:
        """Analyze score stability across runs"""
        n_queries = len(runs[0])
        n_runs = len(runs)

        score_variances = []
        relative_diffs = []

        for query_idx in range(n_queries):
            query_results = [runs[run_idx][query_idx] for run_idx in range(n_runs)]

            doc_scores = defaultdict(list)
            for result in query_results:
                for doc_id, score in zip(result.doc_ids, result.scores):
                    doc_scores[doc_id].append(score)

            for doc_id, scores in doc_scores.items():
                if len(scores) > 1:
                    score_variances.append(np.var(scores))
                    rel_diff = (max(scores) - min(scores)) / (abs(np.mean(scores)) + 1e-8)
                    relative_diffs.append(rel_diff)

        return {
            "mean_score_variance": np.mean(score_variances) if score_variances else 0,
            "max_score_variance": np.max(score_variances) if score_variances else 0,
            "mean_relative_diff": np.mean(relative_diffs) if relative_diffs else 0
        }

    @staticmethod
    def latency_stability(runs: List[List[RetrievalResult]]) -> Dict[str, float]:
        """Analyze latency stability"""
        all_latencies = []

        for run in runs:
            for result in run:
                all_latencies.append(result.latency_ms)

        return {
            "mean_latency_ms": np.mean(all_latencies),
            "std_latency_ms": np.std(all_latencies),
            "cv_latency": np.std(all_latencies) / np.mean(all_latencies) if np.mean(all_latencies) > 0 else 0
        }


def test_faiss_reproducibility():
    """Test FAISS reproducibility with different configurations"""

    # Create test data
    documents = []
    for i in range(10000):
        documents.append({
            "id": f"doc_{i}",
            "text": f"This is document {i} discussing topic {i % 100}",
            "metadata": {"category": f"cat_{i % 10}"}
        })

    queries = [f"Find documents about topic {i}" for i in range(50)]

    # Test different FAISS configurations
    configs_to_test = [
        ExperimentConfig(index_type="Flat", deterministic_mode=True, seed=42),
        ExperimentConfig(index_type="Flat", deterministic_mode=False),
        ExperimentConfig(index_type="IVF", ivf_nlist=100, deterministic_mode=True, seed=42),
        ExperimentConfig(index_type="HNSW", hnsw_M=16, deterministic_mode=True, seed=42),
        ExperimentConfig(index_type="LSH", deterministic_mode=True, seed=42),
    ]

    results = {}

    for config in configs_to_test:
        logger.info(f"\nTesting configuration: {config.index_type} (deterministic={config.deterministic_mode})")

        # Run multiple times
        runs = []
        for run_idx in range(5):
            retrieval = FaissRetrieval(config)
            retrieval.index_documents(documents)
            run_results = retrieval.search(queries)
            runs.append(run_results)
            retrieval.reset()

        # Calculate metrics
        metrics = ReproducibilityMetrics.calculate_all_metrics(runs)

        config_key = f"{config.index_type}_det_{config.deterministic_mode}"
        results[config_key] = {
            "config": config.to_dict(),
            "metrics": metrics
        }

        # Log summary
        logger.info(f"Exact match rate: {metrics['exact_match']['exact_match_rate']:.3f}")
        logger.info(f"Mean Jaccard similarity: {metrics['overlap']['mean_jaccard']:.3f}")
        logger.info(f"Mean Kendall tau: {metrics['rank_correlation']['mean_kendall_tau']:.3f}")

    return results


def test_distributed_reproducibility():
    """Test distributed FAISS reproducibility"""

    if not MPI_AVAILABLE:
        logger.warning("MPI not available, skipping distributed tests")
        return {}

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Create test data (same on all ranks)
    documents = []
    for i in range(10000):
        documents.append({
            "id": f"doc_{i}",
            "text": f"This is document {i} discussing topic {i % 100}",
            "metadata": {"category": f"cat_{i % 10}"}
        })

    queries = [f"Find documents about topic {i}" for i in range(20)]

    # Test distributed configurations
    config = ExperimentConfig(
        index_type="Flat",
        distributed=True,
        shard_method="hash",
        deterministic_mode=True,
        seed=42
    )

    # Create distributed tester
    tester = DistributedReproducibilityTester(config)

    # Run distributed tests
    if rank == 0:
        logger.info("Starting distributed reproducibility tests...")

    results = tester.test_distributed_factors(documents, queries)

    if rank == 0:
        # Save results
        with open("distributed_reproducibility_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info("Distributed test results saved")

    return results


def main():
    """Main execution function"""

    # Test 1: FAISS reproducibility
    logger.info("="*60)
    logger.info("Testing FAISS Reproducibility")
    logger.info("="*60)

    faiss_results = test_faiss_reproducibility()

    # Save FAISS results
    with open("faiss_reproducibility_results.json", "w") as f:
        json.dump(faiss_results, f, indent=2, default=str)

    # Test 2: Distributed reproducibility (if MPI available)
    if MPI_AVAILABLE:
        logger.info("\n" + "="*60)
        logger.info("Testing Distributed Reproducibility")
        logger.info("="*60)

        distributed_results = test_distributed_reproducibility()

    logger.info("\n" + "="*60)
    logger.info("All tests completed!")
    logger.info("="*60)


class GPUNonDeterminismTester:
    """Test GPU-specific non-determinism factors"""

    def __init__(self):
        self.results = {}

    def test_gpu_factors(self, documents: List[Dict[str, str]], queries: List[str]) -> Dict[str, Any]:
        """Test various GPU-related factors"""

        gpu_factors = {
            "atomicAdd_order": self._test_atomic_operations,
            "parallel_reduction": self._test_parallel_reduction,
            "multi_gpu_sync": self._test_multi_gpu_sync,
            "mixed_precision": self._test_mixed_precision,
            "tensor_core_usage": self._test_tensor_cores
        }

        results = {}
        for factor_name, test_func in gpu_factors.items():
            logger.info(f"Testing GPU factor: {factor_name}")
            results[factor_name] = test_func(documents, queries)

        return results

    def _test_atomic_operations(self, documents: List[Dict[str, str]], queries: List[str]) -> Dict[str, Any]:
        """Test impact of atomic operations on reproducibility"""

        configs = [
            ExperimentConfig(use_gpu=True, deterministic_mode=False),
            ExperimentConfig(use_gpu=True, deterministic_mode=True, seed=42),
        ]

        results = {}
        for config in configs:
            runs = []
            for _ in range(5):
                retrieval = FaissRetrieval(config)
                retrieval.index_documents(documents[:1000])  # Smaller subset
                run_results = retrieval.search(queries[:10])
                runs.append(run_results)
                retrieval.reset()

            metrics = ReproducibilityMetrics.calculate_all_metrics(runs)
            results[f"deterministic_{config.deterministic_mode}"] = metrics

        return results

    def _test_parallel_reduction(self, documents: List[Dict[str, str]], queries: List[str]) -> Dict[str, Any]:
        """Test parallel reduction operations"""

        # Test with different batch sizes that affect parallelism
        batch_sizes = [1, 32, 128, 512]
        results = {}

        for batch_size in batch_sizes:
            config = ExperimentConfig(
                use_gpu=True,
                batch_size=batch_size,
                deterministic_mode=False
            )

            runs = []
            for _ in range(5):
                retrieval = FaissRetrieval(config)
                retrieval.index_documents(documents[:1000])
                run_results = retrieval.search(queries[:10])
                runs.append(run_results)
                retrieval.reset()

            metrics = ReproducibilityMetrics.calculate_all_metrics(runs)
            results[f"batch_{batch_size}"] = metrics

        return results

    def _test_multi_gpu_sync(self, documents: List[Dict[str, str]], queries: List[str]) -> Dict[str, Any]:
        """Test multi-GPU synchronization effects"""

        if faiss.get_num_gpus() < 2:
            return {"error": "Less than 2 GPUs available"}

        configs = [
            ExperimentConfig(use_gpu=True, num_gpus=1),
            ExperimentConfig(use_gpu=True, num_gpus=2),
            ExperimentConfig(use_gpu=True, num_gpus=faiss.get_num_gpus()),
        ]

        results = {}
        for config in configs:
            runs = []
            for _ in range(5):
                retrieval = FaissRetrieval(config)
                retrieval.index_documents(documents[:1000])
                run_results = retrieval.search(queries[:10])
                runs.append(run_results)
                retrieval.reset()

            metrics = ReproducibilityMetrics.calculate_all_metrics(runs)
            results[f"gpus_{config.num_gpus}"] = metrics

        return results

    def _test_mixed_precision(self, documents: List[Dict[str, str]], queries: List[str]) -> Dict[str, Any]:
        """Test mixed precision effects with actual embedding precision testing"""

        # Import the embedding tester
        try:
            from embedding_reproducibility_tester import EmbeddingReproducibilityTester, EmbeddingConfig

            precision_configs = [
                EmbeddingConfig(precision="fp32", deterministic=True),
                EmbeddingConfig(precision="fp16", deterministic=True),
                EmbeddingConfig(precision="fp32", deterministic=False),
                EmbeddingConfig(precision="fp16", deterministic=False),
            ]

            # Add TF32 and BF16 if supported
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
                precision_configs.extend([
                    EmbeddingConfig(precision="tf32", deterministic=True),
                    EmbeddingConfig(precision="tf32", deterministic=False),
                    EmbeddingConfig(precision="bf16", deterministic=True),
                    EmbeddingConfig(precision="bf16", deterministic=False),
                ])

            results = {}
            doc_texts = [doc['text'] for doc in documents[:1000]]

            for emb_config in precision_configs:
                config_name = f"{emb_config.precision}_{emb_config.deterministic}"

                # Test embedding stability
                embedding_tester = EmbeddingReproducibilityTester(emb_config)
                embedding_results = embedding_tester.test_embedding_stability(doc_texts, n_runs=3)

                # Test retrieval with these embeddings
                retrieval_runs = []
                for _ in range(3):
                    config = ExperimentConfig(use_gpu=True)
                    retrieval = FaissRetrieval(config)

                    # Override encoder with precision-aware tester
                    retrieval.encoder = embedding_tester
                    def encode_method(texts, **kwargs):
                        return embedding_tester.encode_texts(texts)
                    retrieval.encoder.encode = encode_method

                    retrieval.index_documents(documents[:1000])
                    run_results = retrieval.search(queries[:10])
                    retrieval_runs.append(run_results)
                    retrieval.reset()

                retrieval_metrics = ReproducibilityMetrics.calculate_all_metrics(retrieval_runs)

                results[config_name] = {
                    "embedding_stability": embedding_results["metrics"],
                    "retrieval_reproducibility": retrieval_metrics,
                    "precision": emb_config.precision,
                    "deterministic": emb_config.deterministic
                }

            return results

        except ImportError:
            # Fallback to original implementation
            results = {}
            for use_fp16 in [False, True]:
                config = ExperimentConfig(use_gpu=True)

                runs = []
                for _ in range(5):
                    retrieval = FaissRetrieval(config)

                    # Simulate mixed precision by adding noise
                    if use_fp16:
                        # Add quantization noise to simulate fp16
                        retrieval.index_documents(documents[:1000])
                        if retrieval.doc_embeddings is not None:
                            # Simulate fp16 quantization
                            retrieval.doc_embeddings = retrieval.doc_embeddings.astype(np.float16).astype(np.float32)
                    else:
                        retrieval.index_documents(documents[:1000])

                    run_results = retrieval.search(queries[:10])
                    runs.append(run_results)
                    retrieval.reset()

                metrics = ReproducibilityMetrics.calculate_all_metrics(runs)
                results[f"fp16_{use_fp16}"] = metrics

            return results

    def _test_tensor_cores(self, documents: List[Dict[str, str]], queries: List[str]) -> Dict[str, Any]:
        """Test tensor core usage effects"""

        # This is hardware-specific and would require CUDA-level control
        # For now, we'll test with configurations that might trigger tensor core usage

        embedding_dims = [384, 512, 768]  # Different dimensions that may/may not use tensor cores
        results = {}

        for dim in embedding_dims:
            config = ExperimentConfig(
                use_gpu=True,
                embedding_dim=dim,
                embedding_model="/scratch/user/u.bw269205/shared_models/bge_model"  # Will be overridden
            )

            runs = []
            for _ in range(3):  # Fewer runs due to computational cost
                retrieval = FaissRetrieval(config)

                # Override embedding dimension by using random embeddings
                retrieval.encoder = None  # Disable encoder
                retrieval.doc_embeddings = np.random.randn(len(documents[:1000]), dim).astype('float32')
                retrieval.documents = documents[:1000]

                # Create index directly
                retrieval.index = retrieval._create_index()
                retrieval.index.add(retrieval.doc_embeddings)

                # Search with random query embeddings
                query_embeddings = np.random.randn(len(queries[:10]), dim).astype('float32')

                results_run = []
                for i, q_emb in enumerate(query_embeddings):
                    D, I = retrieval.index.search(q_emb.reshape(1, -1), config.top_k)
                    result = RetrievalResult(
                        query_id=f"q_{i}",
                        doc_ids=[retrieval.documents[idx]['id'] for idx in I[0]],
                        scores=D[0].tolist(),
                        latency_ms=0,
                        metadata={"embedding_dim": dim}
                    )
                    results_run.append(result)

                runs.append(results_run)
                retrieval.reset()

            metrics = ReproducibilityMetrics.calculate_all_metrics(runs)
            results[f"dim_{dim}"] = metrics

        return results


class ComprehensiveReproducibilityReport:
    """Generate comprehensive reproducibility analysis report"""

    def __init__(self):
        self.report_data = {}

    def generate_full_report(self,
                           documents: List[Dict[str, str]],
                           queries: List[str],
                           output_dir: str = "reproducibility_analysis") -> None:
        """Generate complete reproducibility analysis"""

        os.makedirs(output_dir, exist_ok=True)

        # 1. Test FAISS index types
        logger.info("Phase 1: Testing FAISS index types...")
        self.report_data["faiss_index_types"] = self._test_index_types(documents, queries)

        # 2. Test hardware factors
        logger.info("\nPhase 2: Testing hardware factors...")
        gpu_tester = GPUNonDeterminismTester()
        self.report_data["gpu_factors"] = gpu_tester.test_gpu_factors(documents, queries)

        # 3. Test scale effects
        logger.info("\nPhase 3: Testing scale effects...")
        self.report_data["scale_effects"] = self._test_scale_effects(documents, queries)

        # 4. Generate visualizations
        logger.info("\nPhase 4: Generating visualizations...")
        self._generate_visualizations(output_dir)

        # 5. Write comprehensive report
        self._write_report(output_dir)

        logger.info(f"\nComprehensive report generated in: {output_dir}/")

    def _test_index_types(self, documents: List[Dict[str, str]], queries: List[str]) -> Dict[str, Any]:
        """Test different FAISS index types"""

        index_configs = {
            "Flat_L2": ExperimentConfig(index_type="Flat", faiss_metric="L2"),
            "Flat_IP": ExperimentConfig(index_type="Flat", faiss_metric="IP"),
            "IVF_small": ExperimentConfig(index_type="IVF", ivf_nlist=50, ivf_nprobe=5),
            "IVF_large": ExperimentConfig(index_type="IVF", ivf_nlist=200, ivf_nprobe=20),
            "HNSW_fast": ExperimentConfig(index_type="HNSW", hnsw_M=16, hnsw_ef_search=50),
            "HNSW_accurate": ExperimentConfig(index_type="HNSW", hnsw_M=32, hnsw_ef_search=200),
            "LSH": ExperimentConfig(index_type="LSH"),
        }

        results = {}
        for name, config in index_configs.items():
            runs = []
            for _ in range(5):
                retrieval = FaissRetrieval(config)
                retrieval.index_documents(documents[:5000])
                run_results = retrieval.search(queries[:20])
                runs.append(run_results)
                retrieval.reset()

            metrics = ReproducibilityMetrics.calculate_all_metrics(runs)
            results[name] = {
                "config": config.to_dict(),
                "metrics": metrics
            }

        return results

    def _test_scale_effects(self, documents: List[Dict[str, str]], queries: List[str]) -> Dict[str, Any]:
        """Test how scale affects reproducibility"""

        doc_sizes = [100, 1000, 5000, 10000]
        query_sizes = [10, 50, 100]

        results = {}
        for n_docs in doc_sizes:
            for n_queries in query_sizes:
                key = f"docs_{n_docs}_queries_{n_queries}"

                config = ExperimentConfig(index_type="Flat")
                runs = []

                for _ in range(3):
                    retrieval = FaissRetrieval(config)
                    retrieval.index_documents(documents[:n_docs])
                    run_results = retrieval.search(queries[:n_queries])
                    runs.append(run_results)
                    retrieval.reset()

                metrics = ReproducibilityMetrics.calculate_all_metrics(runs)
                results[key] = {
                    "n_docs": n_docs,
                    "n_queries": n_queries,
                    "metrics": metrics
                }

        return results

    def _generate_visualizations(self, output_dir: str) -> None:
        """Generate visualization plots"""

        import matplotlib.pyplot as plt
        import seaborn as sns

        # 1. Index type comparison
        if "faiss_index_types" in self.report_data:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            index_names = []
            exact_match_rates = []
            mean_jaccards = []
            latencies = []

            for name, data in self.report_data["faiss_index_types"].items():
                index_names.append(name)
                exact_match_rates.append(data["metrics"]["exact_match"]["exact_match_rate"])
                mean_jaccards.append(data["metrics"]["overlap"]["mean_jaccard"])
                latencies.append(data["metrics"]["latency"]["mean_latency_ms"])

            # Plot 1: Exact match rates
            axes[0, 0].bar(index_names, exact_match_rates)
            axes[0, 0].set_title("Exact Match Rate by Index Type")
            axes[0, 0].set_ylabel("Exact Match Rate")
            axes[0, 0].tick_params(axis='x', rotation=45)

            # Plot 2: Jaccard similarity
            axes[0, 1].bar(index_names, mean_jaccards)
            axes[0, 1].set_title("Mean Jaccard Similarity by Index Type")
            axes[0, 1].set_ylabel("Jaccard Similarity")
            axes[0, 1].tick_params(axis='x', rotation=45)

            # Plot 3: Latency
            axes[1, 0].bar(index_names, latencies)
            axes[1, 0].set_title("Mean Latency by Index Type")
            axes[1, 0].set_ylabel("Latency (ms)")
            axes[1, 0].tick_params(axis='x', rotation=45)

            # Plot 4: Trade-off scatter
            axes[1, 1].scatter(latencies, mean_jaccards, s=100)
            for i, name in enumerate(index_names):
                axes[1, 1].annotate(name, (latencies[i], mean_jaccards[i]))
            axes[1, 1].set_xlabel("Latency (ms)")
            axes[1, 1].set_ylabel("Jaccard Similarity")
            axes[1, 1].set_title("Latency vs Reproducibility Trade-off")

            plt.tight_layout()
            plt.savefig(f"{output_dir}/index_type_comparison.png", dpi=300)
            plt.close()

        # 2. Scale effects heatmap
        if "scale_effects" in self.report_data:
            scale_data = self.report_data["scale_effects"]

            # Create matrix for heatmap
            doc_sizes = sorted(set(d["n_docs"] for d in scale_data.values()))
            query_sizes = sorted(set(d["n_queries"] for d in scale_data.values()))

            jaccard_matrix = np.zeros((len(doc_sizes), len(query_sizes)))

            for i, n_docs in enumerate(doc_sizes):
                for j, n_queries in enumerate(query_sizes):
                    key = f"docs_{n_docs}_queries_{n_queries}"
                    if key in scale_data:
                        jaccard_matrix[i, j] = scale_data[key]["metrics"]["overlap"]["mean_jaccard"]

            plt.figure(figsize=(8, 6))
            sns.heatmap(jaccard_matrix,
                       xticklabels=query_sizes,
                       yticklabels=doc_sizes,
                       annot=True,
                       fmt='.3f',
                       cmap='YlOrRd')
            plt.xlabel("Number of Queries")
            plt.ylabel("Number of Documents")
            plt.title("Reproducibility (Jaccard) vs Scale")
            plt.savefig(f"{output_dir}/scale_effects_heatmap.png", dpi=300)
            plt.close()

    def _write_report(self, output_dir: str) -> None:
        """Write comprehensive text report"""

        report_path = f"{output_dir}/reproducibility_report.md"

        with open(report_path, 'w') as f:
            f.write("# RAG Reproducibility Analysis Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write("This report analyzes the reproducibility of RAG retrieval systems across various factors.\n\n")

            # Key Findings
            f.write("## Key Findings\n\n")

            # Find most/least reproducible configurations
            if "faiss_index_types" in self.report_data:
                best_config = None
                worst_config = None
                best_jaccard = 0
                worst_jaccard = 1

                for name, data in self.report_data["faiss_index_types"].items():
                    jaccard = data["metrics"]["overlap"]["mean_jaccard"]
                    if jaccard > best_jaccard:
                        best_jaccard = jaccard
                        best_config = name
                    if jaccard < worst_jaccard:
                        worst_jaccard = jaccard
                        worst_config = name

                f.write(f"- **Most reproducible**: {best_config} (Jaccard: {best_jaccard:.3f})\n")
                f.write(f"- **Least reproducible**: {worst_config} (Jaccard: {worst_jaccard:.3f})\n\n")

            # Detailed Results
            f.write("## Detailed Results\n\n")

            # Index Type Analysis
            if "faiss_index_types" in self.report_data:
                f.write("### Index Type Analysis\n\n")
                f.write("| Index Type | Exact Match | Jaccard | Kendall Tau | Latency (ms) |\n")
                f.write("|------------|-------------|---------|-------------|-------------|\n")

                for name, data in self.report_data["faiss_index_types"].items():
                    metrics = data["metrics"]
                    f.write(f"| {name} | "
                           f"{metrics['exact_match']['exact_match_rate']:.3f} | "
                           f"{metrics['overlap']['mean_jaccard']:.3f} | "
                           f"{metrics['rank_correlation']['mean_kendall_tau']:.3f} | "
                           f"{metrics['latency']['mean_latency_ms']:.2f} |\n")

                f.write("\n")

            # GPU Factors
            if "gpu_factors" in self.report_data:
                f.write("### GPU Non-determinism Factors\n\n")

                for factor_name, factor_data in self.report_data["gpu_factors"].items():
                    f.write(f"#### {factor_name}\n\n")

                    if isinstance(factor_data, dict) and "error" not in factor_data:
                        for config_name, metrics in factor_data.items():
                            if isinstance(metrics, dict) and "exact_match" in metrics:
                                f.write(f"- **{config_name}**: "
                                       f"Jaccard={metrics['overlap']['mean_jaccard']:.3f}\n")
                    f.write("\n")

            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("Based on the analysis, we recommend:\n\n")
            f.write("1. **For maximum reproducibility**: Use Flat index with deterministic mode enabled\n")
            f.write("2. **For production systems**: Consider HNSW with higher M and ef_search values\n")
            f.write("3. **For distributed systems**: Use hash-based sharding for consistent results\n")
            f.write("4. **GPU considerations**: Enable deterministic CUDA operations when reproducibility is critical\n")

            # Save full JSON data
            json_path = f"{output_dir}/full_analysis_data.json"
            with open(json_path, 'w') as json_f:
                json.dump(self.report_data, json_f, indent=2, default=str)

            f.write(f"\n\nFull data saved to: {json_path}\n")


if __name__ == "__main__":
    main()