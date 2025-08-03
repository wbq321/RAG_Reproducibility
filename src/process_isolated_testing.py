"""
Process-Isolated Reproducibility Testing
Ensures each test run executes in a completely separate Python process for maximum isolation
"""

import os
import sys
import json
import tempfile
import subprocess
import multiprocessing as mp
import numpy as np
from typing import List, Dict, Any, Tuple
import logging
import pickle
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class ProcessIsolatedTester:
    """Executes reproducibility tests in separate Python processes"""

    def __init__(self, working_dir: str = None):
        self.working_dir = working_dir or os.getcwd()
        self.temp_dir = tempfile.mkdtemp(prefix="rag_repro_")
        logger.info(f"Process isolated testing temp directory: {self.temp_dir}")

    def __del__(self):
        """Cleanup temp directory"""
        import shutil
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except:
            pass

    def run_embedding_stability_isolated(self,
                                       config_dict: Dict[str, Any],
                                       texts: List[str],
                                       n_runs: int = 5) -> Dict[str, Any]:
        """Run embedding stability test across separate processes"""

        logger.info(f"Running {n_runs} embedding stability tests in separate processes...")

        # Save input data to temporary files
        config_file = os.path.join(self.temp_dir, "embedding_config.json")
        texts_file = os.path.join(self.temp_dir, "texts.json")

        with open(config_file, 'w') as f:
            json.dump(config_dict, f)

        with open(texts_file, 'w') as f:
            json.dump(texts, f)

        # Create script for isolated execution
        script_content = self._create_embedding_test_script()
        script_file = os.path.join(self.temp_dir, "embedding_test_runner.py")

        with open(script_file, 'w') as f:
            f.write(script_content)

        # Run each test in a separate process
        results = []
        for run_idx in range(n_runs):
            logger.info(f"Starting isolated embedding run {run_idx + 1}/{n_runs}")

            result = self._run_single_embedding_process(
                script_file, config_file, texts_file, run_idx
            )

            if result is not None:
                results.append(result)
            else:
                logger.warning(f"Run {run_idx + 1} failed")

        # Calculate metrics from collected results
        if len(results) >= 2:
            metrics = self._calculate_cross_process_metrics(results)
            return {
                "config": config_dict,
                "metrics": metrics,
                "embeddings": [r["embeddings"] for r in results],
                "timings": [r["timing"] for r in results],
                "process_isolation": True,
                "successful_runs": len(results),
                "total_runs": n_runs
            }
        else:
            logger.error("Insufficient successful runs for metrics calculation")
            return {"error": "Insufficient successful runs", "successful_runs": len(results)}

    def run_retrieval_stability_isolated(self,
                                        rag_config_dict: Dict[str, Any],
                                        embedding_config_dict: Dict[str, Any],
                                        documents: List[Dict[str, str]],
                                        queries: List[str],
                                        n_runs: int = 5) -> Dict[str, Any]:
        """Run retrieval stability test across separate processes"""

        logger.info(f"Running {n_runs} retrieval stability tests in separate processes...")

        # Save input data to temporary files
        rag_config_file = os.path.join(self.temp_dir, "rag_config.json")
        embedding_config_file = os.path.join(self.temp_dir, "embedding_config.json")
        documents_file = os.path.join(self.temp_dir, "documents.pkl")
        queries_file = os.path.join(self.temp_dir, "queries.json")

        with open(rag_config_file, 'w') as f:
            json.dump(rag_config_dict, f)

        with open(embedding_config_file, 'w') as f:
            json.dump(embedding_config_dict, f)

        with open(documents_file, 'wb') as f:
            pickle.dump(documents, f)

        with open(queries_file, 'w') as f:
            json.dump(queries, f)

        # Create script for isolated execution
        script_content = self._create_retrieval_test_script()
        script_file = os.path.join(self.temp_dir, "retrieval_test_runner.py")

        with open(script_file, 'w') as f:
            f.write(script_content)

        # Run each test in a separate process
        results = []
        for run_idx in range(n_runs):
            logger.info(f"Starting isolated retrieval run {run_idx + 1}/{n_runs}")

            result = self._run_single_retrieval_process(
                script_file, rag_config_file, embedding_config_file,
                documents_file, queries_file, run_idx
            )

            if result is not None:
                results.append(result)
            else:
                logger.warning(f"Run {run_idx + 1} failed")

        # Calculate retrieval metrics
        if len(results) >= 2:
            from rag_reproducibility_framework import ReproducibilityMetrics
            retrieval_runs = [r["retrieval_results"] for r in results]
            metrics = ReproducibilityMetrics.calculate_all_metrics(retrieval_runs)

            return {
                "config": {"rag": rag_config_dict, "embedding": embedding_config_dict},
                "metrics": metrics,
                "retrieval_runs": retrieval_runs,
                "timings": [r["timing"] for r in results],
                "process_isolation": True,
                "successful_runs": len(results),
                "total_runs": n_runs
            }
        else:
            logger.error("Insufficient successful runs for metrics calculation")
            return {"error": "Insufficient successful runs", "successful_runs": len(results)}

    def _create_embedding_test_script(self) -> str:
        """Create Python script for isolated embedding testing"""
        # Get absolute path to src directory
        src_dir = os.path.dirname(os.path.abspath(__file__))
        return f'''
import sys
import os
import json
import numpy as np
import torch
from typing import List, Dict, Any
import time

# Add src directory to path
sys.path.insert(0, "{src_dir}")

def run_embedding_test(config_file: str, texts_file: str, run_idx: int) -> Dict[str, Any]:
    """Run single embedding test in isolated process"""

    # Import after path setup
    from embedding_reproducibility_tester import EmbeddingReproducibilityTester, EmbeddingConfig

    # Load configuration and data
    with open(config_file, 'r') as f:
        config_dict = json.load(f)

    with open(texts_file, 'r') as f:
        texts = json.load(f)

    # Create config object
    config = EmbeddingConfig(**config_dict)

    # Force fresh environment
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Create tester (fresh instance)
    tester = EmbeddingReproducibilityTester(config)

    # Run single embedding generation
    start_time = time.time()
    embeddings = tester.encode_texts(texts)
    duration = time.time() - start_time

    # Clean up
    del tester.model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {
        "embeddings": embeddings.tolist(),
        "timing": duration,
        "run_idx": run_idx,
        "process_id": os.getpid()
    }

if __name__ == "__main__":
    config_file = sys.argv[1]
    texts_file = sys.argv[2]
    run_idx = int(sys.argv[3])
    output_file = sys.argv[4]

    try:
        result = run_embedding_test(config_file, texts_file, run_idx)

        with open(output_file, 'w') as f:
            json.dump(result, f, default=str)

    except Exception as e:
        error_result = {
            "error": str(e),
            "run_idx": run_idx,
            "process_id": os.getpid()
        }

        with open(output_file, 'w') as f:
            json.dump(error_result, f)
'''

    def _create_retrieval_test_script(self) -> str:
        """Create Python script for isolated retrieval testing"""
        # Get absolute path to src directory
        src_dir = os.path.dirname(os.path.abspath(__file__))
        return f'''
import sys
import os
import json
import pickle
import torch
from typing import List, Dict, Any
import time

# Add src directory to path
sys.path.insert(0, "{src_dir}")

def run_retrieval_test(rag_config_file: str, embedding_config_file: str,
                      documents_file: str, queries_file: str, run_idx: int) -> Dict[str, Any]:
    """Run single retrieval test in isolated process"""

    # Import after path setup
    from rag_reproducibility_framework import FaissRetrieval, ExperimentConfig
    from embedding_reproducibility_tester import EmbeddingReproducibilityTester, EmbeddingConfig

    # Load configuration and data
    with open(rag_config_file, 'r') as f:
        rag_config_dict = json.load(f)

    with open(embedding_config_file, 'r') as f:
        embedding_config_dict = json.load(f)

    with open(documents_file, 'rb') as f:
        documents = pickle.load(f)

    with open(queries_file, 'r') as f:
        queries = json.load(f)

    # Create config objects
    rag_config = ExperimentConfig(**rag_config_dict)
    embedding_config = EmbeddingConfig(**embedding_config_dict)

    # Force fresh environment
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Create retrieval system with custom embedding
    retrieval = FaissRetrieval(rag_config)
    embedding_tester = EmbeddingReproducibilityTester(embedding_config)

    # Override encoder
    def encode_method(texts, **kwargs):
        return embedding_tester.encode_texts(texts)

    retrieval.encoder.encode = encode_method

    # Run retrieval test
    start_time = time.time()

    # Index documents
    retrieval.index_documents(documents)

    # Search queries
    query_strings = [q if isinstance(q, str) else str(q) for q in queries]
    results = retrieval.search(query_strings)

    duration = time.time() - start_time

    # Clean up
    retrieval.reset()
    del embedding_tester.model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Convert results to serializable format
    serializable_results = []
    for result in results:
        serializable_results.append({
            "query_id": result.query_id,
            "doc_ids": result.doc_ids,
            "scores": result.scores,
            "latency_ms": result.latency_ms,
            "metadata": result.metadata
        })

    return {
        "retrieval_results": serializable_results,
        "timing": duration,
        "run_idx": run_idx,
        "process_id": os.getpid()
    }

if __name__ == "__main__":
    rag_config_file = sys.argv[1]
    embedding_config_file = sys.argv[2]
    documents_file = sys.argv[3]
    queries_file = sys.argv[4]
    run_idx = int(sys.argv[5])
    output_file = sys.argv[6]

    try:
        result = run_retrieval_test(rag_config_file, embedding_config_file,
                                  documents_file, queries_file, run_idx)

        with open(output_file, 'w') as f:
            json.dump(result, f, default=str)

    except Exception as e:
        error_result = {
            "error": str(e),
            "run_idx": run_idx,
            "process_id": os.getpid()
        }

        with open(output_file, 'w') as f:
            json.dump(error_result, f)
'''

    def _run_single_embedding_process(self, script_file: str, config_file: str,
                                    texts_file: str, run_idx: int) -> Dict[str, Any]:
        """Run single embedding test in separate process"""

        output_file = os.path.join(self.temp_dir, f"embedding_result_{run_idx}.json")

        cmd = [
            sys.executable, script_file,
            config_file, texts_file, str(run_idx), output_file
        ]

        try:
            # Set environment for deterministic execution
            env = os.environ.copy()

            # Add src directory to PYTHONPATH
            src_dir = os.path.dirname(os.path.abspath(__file__))
            if 'PYTHONPATH' in env:
                env['PYTHONPATH'] = f"{src_dir}{os.pathsep}{env['PYTHONPATH']}"
            else:
                env['PYTHONPATH'] = src_dir

            env.update({
                'CUDA_LAUNCH_BLOCKING': '1',
                'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
                'PYTHONHASHSEED': '0'
            })

            process = subprocess.run(
                cmd,
                cwd=self.working_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if process.returncode == 0:
                # Load result
                with open(output_file, 'r') as f:
                    result = json.load(f)

                if "error" not in result:
                    result["embeddings"] = np.array(result["embeddings"])
                    return result
                else:
                    logger.error(f"Run {run_idx} failed: {result['error']}")
                    return None
            else:
                logger.error(f"Process failed with return code {process.returncode}")
                logger.error(f"STDERR: {process.stderr}")
                return None

        except subprocess.TimeoutExpired:
            logger.error(f"Run {run_idx} timed out")
            return None
        except Exception as e:
            logger.error(f"Error running process {run_idx}: {e}")
            return None

    def _run_single_retrieval_process(self, script_file: str, rag_config_file: str,
                                    embedding_config_file: str, documents_file: str,
                                    queries_file: str, run_idx: int) -> Dict[str, Any]:
        """Run single retrieval test in separate process"""

        output_file = os.path.join(self.temp_dir, f"retrieval_result_{run_idx}.json")

        cmd = [
            sys.executable, script_file,
            rag_config_file, embedding_config_file, documents_file,
            queries_file, str(run_idx), output_file
        ]

        try:
            # Set environment for deterministic execution
            env = os.environ.copy()

            # Add src directory to PYTHONPATH
            src_dir = os.path.dirname(os.path.abspath(__file__))
            if 'PYTHONPATH' in env:
                env['PYTHONPATH'] = f"{src_dir}{os.pathsep}{env['PYTHONPATH']}"
            else:
                env['PYTHONPATH'] = src_dir

            env.update({
                'CUDA_LAUNCH_BLOCKING': '1',
                'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
                'PYTHONHASHSEED': '0'
            })

            process = subprocess.run(
                cmd,
                cwd=self.working_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            if process.returncode == 0:
                # Load result
                with open(output_file, 'r') as f:
                    result = json.load(f)

                if "error" not in result:
                    # Convert back to RetrievalResult objects
                    from rag_reproducibility_framework import RetrievalResult
                    retrieval_results = []
                    for r in result["retrieval_results"]:
                        retrieval_results.append(RetrievalResult(
                            query_id=r["query_id"],
                            doc_ids=r["doc_ids"],
                            scores=r["scores"],
                            latency_ms=r["latency_ms"],
                            metadata=r["metadata"]
                        ))
                    result["retrieval_results"] = retrieval_results
                    return result
                else:
                    logger.error(f"Run {run_idx} failed: {result['error']}")
                    return None
            else:
                logger.error(f"Process failed with return code {process.returncode}")
                logger.error(f"STDERR: {process.stderr}")
                return None

        except subprocess.TimeoutExpired:
            logger.error(f"Run {run_idx} timed out")
            return None
        except Exception as e:
            logger.error(f"Error running process {run_idx}: {e}")
            return None

    def _calculate_cross_process_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate embedding stability metrics across separate processes"""

        embedding_runs = [np.array(r["embeddings"]) for r in results]

        if len(embedding_runs) < 2:
            return {}

        # Use first run as baseline
        baseline = embedding_runs[0]
        n_texts, embed_dim = baseline.shape

        # Calculate pairwise metrics across processes
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

        # Calculate cross-process variance
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
            "num_texts": int(n_texts),
            "cross_process_isolation": True
        }
