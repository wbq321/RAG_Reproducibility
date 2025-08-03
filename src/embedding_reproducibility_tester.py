"""
Embedding Reproducibility Tester - Integration of embedding uncertainty into RAG framework
Combines the standalone embedding uncertainty experiments with the main RAG reproducibility framework
Uses real MS MARCO dataset for realistic reproducibility testing
"""

import os
import sys
import json
import time
import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
import logging
from sentence_transformers import SentenceTransformer

# Add scripts directory to path for dataset loader
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from rag_reproducibility_framework import ExperimentConfig, RetrievalResult, ReproducibilityMetrics

try:
    from dataset_loader import load_dataset_for_reproducibility, load_msmarco_for_reproducibility, MSMARCOLoader
    DATASET_LOADER_AVAILABLE = True
except ImportError:
    DATASET_LOADER_AVAILABLE = False
    print("Warning: Dataset loader not available. Using simulated data as fallback.")

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding reproducibility testing"""
    model_name: str = "/scratch/user/u.bw269205/shared_models/bge_model"
    precision: str = "fp32"  # fp32, fp16, bf16, tf32
    deterministic: bool = True
    device: str = "cuda"
    batch_size: int = 32
    max_length: int = 512
    normalize_embeddings: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def create_local_model_config(model_path: str, **kwargs) -> EmbeddingConfig:
    """Helper function to create EmbeddingConfig with local model path

    Args:
        model_path: Path to local model directory
        **kwargs: Additional configuration parameters to override

    Returns:
        EmbeddingConfig with specified local model path
    """
    config_dict = {
        "model_name": model_path,
        "precision": "fp32",
        "deterministic": True,
        "device": "cuda",
        "batch_size": 32,
        "max_length": 512,
        "normalize_embeddings": True
    }
    config_dict.update(kwargs)
    return EmbeddingConfig(**config_dict)


class EmbeddingReproducibilityTester:
    """Test embedding generation reproducibility across different precisions and configurations"""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model = None
        self._setup_environment()

    def _setup_environment(self):
        """Setup deterministic environment if requested"""
        if self.config.deterministic:
            torch.manual_seed(42)
            np.random.seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        # Setup precision
        if self.config.precision == "fp16":
            if torch.cuda.is_available():
                torch.backends.cudnn.allow_tf32 = False
        elif self.config.precision == "tf32":
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif self.config.precision == "bf16":
            # BF16 setup for both CPU and GPU
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
                # Ampere+ GPU: Enable tensor core BF16
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
            # Note: CPU BF16 support depends on hardware (Intel Ice Lake+, some ARM CPUs)

    def _load_model(self):
        """Load sentence transformer model"""
        if self.model is None:
            # Check if model path exists for local models
            if os.path.exists(self.config.model_name):
                logger.info(f"Loading local model from: {self.config.model_name}")
            else:
                logger.warning(f"Local model path not found: {self.config.model_name}")
                logger.info("Will attempt to load as Hugging Face model name (requires internet)")

            try:
                self.model = SentenceTransformer(self.config.model_name, device=self.device)
                logger.info(f"Model loaded successfully on device: {self.device}")
            except Exception as e:
                logger.error(f"Failed to load model from {self.config.model_name}: {e}")
                raise

            # Apply precision settings
            if self.config.precision == "fp16":
                self.model = self.model.half()
            elif self.config.precision == "bf16":
                # BF16 is supported on both modern CPUs and GPUs
                try:
                    self.model = self.model.to(torch.bfloat16)
                    logger.info(f"BF16 precision enabled on {self.device}")
                except Exception as e:
                    logger.warning(f"BF16 not supported on {self.device}, falling back to FP32: {e}")
                    # Keep model in default precision

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode texts with current configuration"""
        self._load_model()

        # Encode with specified precision
        with torch.no_grad():
            if self.config.precision == "fp16":
                with torch.cuda.amp.autocast():
                    embeddings = self.model.encode(
                        texts,
                        batch_size=self.config.batch_size,
                        normalize_embeddings=self.config.normalize_embeddings,
                        convert_to_numpy=True
                    )
            elif self.config.precision == "bf16":
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    embeddings = self.model.encode(
                        texts,
                        batch_size=self.config.batch_size,
                        normalize_embeddings=self.config.normalize_embeddings,
                        convert_to_numpy=True
                    )
            else:
                embeddings = self.model.encode(
                    texts,
                    batch_size=self.config.batch_size,
                    normalize_embeddings=self.config.normalize_embeddings,
                    convert_to_numpy=True
                )

        return embeddings.astype(np.float32)

    def test_embedding_stability(self, texts: List[str], n_runs: int = 5, use_process_isolation: bool = False) -> Dict[str, Any]:
        """Test embedding stability across multiple runs

        Args:
            texts: List of texts to embed
            n_runs: Number of runs to execute
            use_process_isolation: If True, run each test in a separate Python process
        """

        if use_process_isolation:
            # Use process-isolated testing for maximum reproducibility validation
            from process_isolated_testing import ProcessIsolatedTester

            isolated_tester = ProcessIsolatedTester()
            result = isolated_tester.run_embedding_stability_isolated(
                self.config.to_dict(), texts, n_runs
            )
            return result

        # Original in-process testing
        embeddings_runs = []
        timings = []

        for run_idx in range(n_runs):
            start_time = time.time()
            embeddings = self.encode_texts(texts)
            duration = time.time() - start_time

            embeddings_runs.append(embeddings)
            timings.append(duration)

            logger.info(f"Embedding run {run_idx + 1}/{n_runs} completed in {duration:.3f}s")

        # Calculate stability metrics
        metrics = self._calculate_embedding_metrics(embeddings_runs)
        metrics["timing"] = {
            "mean_duration": np.mean(timings),
            "std_duration": np.std(timings),
            "total_duration": sum(timings)
        }

        return {
            "config": self.config.to_dict(),
            "metrics": metrics,
            "embeddings": embeddings_runs,
            "timings": timings,
            "process_isolation": False
        }

    def _calculate_embedding_metrics(self, embeddings_runs: List[np.ndarray]) -> Dict[str, Any]:
        """Calculate stability metrics for embeddings"""
        if len(embeddings_runs) < 2:
            return {}

        # Use first run as baseline
        baseline = embeddings_runs[0]
        n_texts, embed_dim = baseline.shape

        # Calculate pairwise metrics
        l2_distances = []
        cosine_similarities = []
        max_abs_differences = []

        for i in range(1, len(embeddings_runs)):
            current = embeddings_runs[i]

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
        all_embeddings = np.stack(embeddings_runs, axis=0)  # shape: (n_runs, n_texts, embed_dim)
        embedding_variance = np.var(all_embeddings, axis=0)  # shape: (n_texts, embed_dim)

        return {
            "l2_distance": {
                "mean": np.mean(l2_distances),
                "std": np.std(l2_distances),
                "max": np.max(l2_distances),
                "min": np.min(l2_distances)
            },
            "cosine_similarity": {
                "mean": np.mean(cosine_similarities),
                "std": np.std(cosine_similarities),
                "min": np.min(cosine_similarities)
            },
            "max_abs_difference": {
                "mean": np.mean(max_abs_differences),
                "std": np.std(max_abs_differences),
                "max": np.max(max_abs_differences)
            },
            "embedding_variance": {
                "mean_per_text": np.mean(embedding_variance, axis=1).tolist(),
                "mean_overall": np.mean(embedding_variance),
                "max_overall": np.max(embedding_variance)
            },
            "exact_match_rate": np.mean([np.allclose(baseline, emb, rtol=1e-7, atol=1e-7)
                                       for emb in embeddings_runs[1:]]),
            "dimension": embed_dim,
            "num_texts": n_texts
        }


class IntegratedRAGReproducibilityTester:
    """Integrated tester that combines embedding and retrieval reproducibility"""

    def __init__(self, rag_config: ExperimentConfig, embedding_configs: List[EmbeddingConfig]):
        self.rag_config = rag_config
        self.embedding_configs = embedding_configs
        self.results = {}
        self.data_loader = None

    def load_msmarco_data(self, csv_path: str = None, num_docs: int = 5000, num_queries: int = 100) -> Tuple[List[Dict[str, str]], List[str]]:
        """Load MS MARCO dataset for testing"""

        if not DATASET_LOADER_AVAILABLE:
            logger.warning("Dataset loader not available, using simulated data")
            return self._create_simulated_data(num_docs, num_queries)

        try:
            # Try different possible paths for MS MARCO CSV
            possible_paths = [
                csv_path,
                "data/ms_marco_passages.csv",
                "data/msmarco.csv",
                "data/passages.csv",
                "../data/ms_marco_passages.csv"
            ]

            csv_path_found = None
            for path in possible_paths:
                if path and os.path.exists(path):
                    csv_path_found = path
                    break

            if not csv_path_found:
                logger.warning("MS MARCO CSV file not found in expected locations. Using simulated data.")
                logger.info("Expected locations: data/ms_marco_passages.csv, data/msmarco.csv, data/passages.csv")
                return self._create_simulated_data(num_docs, num_queries)

            logger.info(f"Loading MS MARCO data from: {csv_path_found}")

            # Use the generalized dataset loader for MS MARCO
            documents, queries = load_dataset_for_reproducibility(
                file_path=csv_path_found,
                dataset_type="ms_marco",
                num_docs=num_docs,
                num_queries=num_queries,
                data_dir="data"
            )

            logger.info(f"Successfully loaded {len(documents)} MS MARCO documents and {len(queries)} queries")
            return documents, queries

        except Exception as e:
            logger.error(f"Error loading MS MARCO data: {e}")
            logger.info("Falling back to simulated data")
            return self._create_simulated_data(num_docs, num_queries)

    def _create_simulated_data(self, num_docs: int, num_queries: int) -> Tuple[List[Dict[str, str]], List[str]]:
        """Create simulated data as fallback"""

        logger.info(f"Creating simulated dataset with {num_docs} documents and {num_queries} queries")

        # Create more realistic simulated documents
        topics = [
            "artificial intelligence and machine learning applications",
            "natural language processing and text analysis",
            "computer vision and image recognition systems",
            "data science and statistical analysis methods",
            "deep learning neural network architectures",
            "information retrieval and search algorithms",
            "distributed computing and cloud infrastructure",
            "cybersecurity and network protection protocols",
            "database management and data storage solutions",
            "software engineering and development practices",
            "web development and frontend technologies",
            "mobile application development frameworks",
            "blockchain technology and cryptocurrency systems",
            "quantum computing and quantum algorithms",
            "robotics and autonomous system control",
            "bioinformatics and computational biology",
            "healthcare technology and medical devices",
            "financial technology and trading systems",
            "renewable energy and smart grid technology",
            "transportation systems and autonomous vehicles"
        ]

        domains = [
            "healthcare", "finance", "education", "retail", "manufacturing",
            "transportation", "telecommunications", "government", "research",
            "entertainment", "agriculture", "energy", "logistics", "media"
        ]

        documents = []
        for i in range(num_docs):
            topic = topics[i % len(topics)]
            domain = domains[i % len(domains)]

            # Create more realistic document content
            content_templates = [
                f"Recent research in {topic} has shown significant advances in {domain} applications. "
                f"This comprehensive study examines the implementation challenges and opportunities for {topic} "
                f"within {domain} environments. Key findings indicate that successful deployment requires "
                f"careful consideration of scalability, performance optimization, and integration with existing systems. "
                f"The methodology presented demonstrates practical approaches to addressing common obstacles "
                f"while maintaining high standards for reliability and user experience.",

                f"The integration of {topic} into {domain} systems represents a transformative opportunity "
                f"for organizations seeking competitive advantages. This analysis explores the technical requirements, "
                f"implementation strategies, and potential benefits of adopting {topic} solutions. "
                f"Case studies from leading organizations illustrate successful deployment patterns and lessons learned. "
                f"Future developments in this area are expected to focus on improved efficiency, reduced costs, "
                f"and enhanced user capabilities through advanced {topic} techniques.",

                f"A comprehensive evaluation of {topic} technologies reveals their growing importance in {domain} sectors. "
                f"This investigation covers current market trends, technological capabilities, and implementation considerations. "
                f"The research methodology includes comparative analysis of different approaches, performance benchmarking, "
                f"and expert interviews from industry practitioners. Results demonstrate the significant potential "
                f"for {topic} to revolutionize traditional {domain} processes while addressing contemporary challenges."
            ]

            content = content_templates[i % len(content_templates)]

            documents.append({
                "id": f"sim_doc_{i:06d}",
                "text": content,
                "metadata": {
                    "topic": topic,
                    "domain": domain,
                    "category": f"cat_{i % 10}",
                    "length": len(content),
                    "source": "simulated",
                    "word_count": len(content.split())
                }
            })

        # Generate more realistic queries
        query_templates = [
            f"How does {topic} work in {domain}",
            f"What are the benefits of {topic} for {domain}",
            f"Implementation challenges of {topic} in {domain}",
            f"Latest developments in {topic} for {domain} applications",
            f"Best practices for {topic} deployment in {domain}",
            f"Performance optimization of {topic} systems",
            f"Security considerations for {topic} in {domain}",
            f"Cost analysis of {topic} implementation",
            f"Future trends in {topic} and {domain}",
            f"Comparative study of {topic} approaches"
        ]

        queries = []
        for i in range(num_queries):
            topic = topics[i % len(topics)]
            domain = domains[i % len(domains)]
            template = query_templates[i % len(query_templates)]

            query = template.format(topic=topic, domain=domain)
            queries.append(query)

        return documents, queries

    def test_end_to_end_reproducibility(self,
                                      documents: List[Dict[str, str]],
                                      queries: List[str],
                                      n_runs: int = 5,
                                      use_process_isolation: bool = False) -> Dict[str, Any]:
        """Test reproducibility from embedding generation through retrieval

        Args:
            documents: List of documents to index
            queries: List of queries to search
            n_runs: Number of runs to execute
            use_process_isolation: If True, run each test in a separate Python process
        """

        results = {
            "embedding_stability": {},
            "retrieval_reproducibility": {},
            "end_to_end_analysis": {}
        }

        # Test each embedding configuration
        for emb_config in self.embedding_configs:
            config_name = f"{emb_config.precision}_{emb_config.deterministic}"
            logger.info(f"Testing embedding configuration: {config_name}")

            if use_process_isolation:
                # Use process-isolated testing
                from process_isolated_testing import ProcessIsolatedTester

                isolated_tester = ProcessIsolatedTester()

                # 1. Test embedding stability with process isolation
                doc_texts = [doc['text'] for doc in documents]

                doc_embedding_results = isolated_tester.run_embedding_stability_isolated(
                    emb_config.to_dict(), doc_texts, n_runs
                )

                query_embedding_results = isolated_tester.run_embedding_stability_isolated(
                    emb_config.to_dict(), queries, n_runs
                )

                results["embedding_stability"][config_name] = {
                    "documents": doc_embedding_results,
                    "queries": query_embedding_results
                }

                # 2. Test retrieval reproducibility with process isolation
                retrieval_results = isolated_tester.run_retrieval_stability_isolated(
                    self.rag_config.to_dict(), emb_config.to_dict(),
                    documents, queries, n_runs
                )

                results["retrieval_reproducibility"][config_name] = retrieval_results

            else:
                # Original in-process testing
                # 1. Test embedding stability
                embedding_tester = EmbeddingReproducibilityTester(emb_config)

                # Test on document texts
                doc_texts = [doc['text'] for doc in documents]
                doc_embedding_results = embedding_tester.test_embedding_stability(doc_texts, n_runs)

                # Test on query texts
                query_embedding_results = embedding_tester.test_embedding_stability(queries, n_runs)

                results["embedding_stability"][config_name] = {
                    "documents": doc_embedding_results,
                    "queries": query_embedding_results
                }

                # 2. Test retrieval reproducibility with this embedding configuration
                retrieval_results = self._test_retrieval_with_embeddings(
                    documents, queries, embedding_tester, n_runs
                )

                results["retrieval_reproducibility"][config_name] = retrieval_results

        # 3. Cross-configuration analysis
        results["end_to_end_analysis"] = self._analyze_cross_configuration(results)
        results["process_isolation"] = use_process_isolation

        return results

    def _test_retrieval_with_embeddings(self,
                                      documents: List[Dict[str, str]],
                                      queries: List[str],
                                      embedding_tester: EmbeddingReproducibilityTester,
                                      n_runs: int) -> Dict[str, Any]:
        """Test retrieval reproducibility with specific embedding configuration"""
        from rag_reproducibility_framework import FaissRetrieval

        retrieval_runs = []

        for run_idx in range(n_runs):
            # Create a modified RAG config that uses our embedding tester
            modified_config = ExperimentConfig(**self.rag_config.to_dict())

            # Create retrieval system
            retrieval = FaissRetrieval(modified_config)

            # Override the encoder with our configured one
            retrieval.encoder = embedding_tester

            # Monkey-patch the encode method to use our tester
            def encode_method(texts, **kwargs):
                return embedding_tester.encode_texts(texts)

            retrieval.encoder.encode = encode_method

            # Index and search
            retrieval.index_documents(documents)
            run_results = retrieval.search(queries)
            retrieval_runs.append(run_results)
            retrieval.reset()

        # Calculate retrieval metrics
        retrieval_metrics = ReproducibilityMetrics.calculate_all_metrics(retrieval_runs)

        return {
            "retrieval_metrics": retrieval_metrics,
            "num_runs": n_runs
        }

    def _analyze_cross_configuration(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze relationships between embedding stability and retrieval reproducibility"""

        analysis = {
            "precision_effects": {},
            "deterministic_effects": {},
            "stability_correlation": {}
        }

        # Extract metrics for analysis
        config_metrics = {}
        for config_name in results["embedding_stability"].keys():
            # Handle different result structures (process isolation vs in-process)
            embedding_data = results["embedding_stability"][config_name]["documents"]

            if "metrics" in embedding_data:
                # In-process structure
                emb_stability = embedding_data["metrics"]
            elif "process_isolation" in embedding_data and embedding_data.get("process_isolation"):
                # Process isolation structure
                emb_stability = embedding_data.get("metrics", {})
            else:
                # Fallback - try to find metrics anywhere in the structure
                emb_stability = embedding_data.get("metrics", {})

            # Handle retrieval results similarly
            retrieval_data = results["retrieval_reproducibility"][config_name]
            if "retrieval_metrics" in retrieval_data:
                retrieval_repro = retrieval_data["retrieval_metrics"]
            elif "metrics" in retrieval_data:
                retrieval_repro = retrieval_data["metrics"]
            else:
                retrieval_repro = {}

            config_metrics[config_name] = {
                "embedding_l2_mean": emb_stability.get("l2_distance", {}).get("mean", 0),
                "embedding_cosine_mean": emb_stability.get("cosine_similarity", {}).get("mean", 1),
                "retrieval_jaccard": retrieval_repro.get("overlap", {}).get("mean_jaccard", 0),
                "retrieval_exact_match": retrieval_repro.get("exact_match", {}).get("exact_match_rate", 0)
            }

        # Analyze precision effects
        precisions = set(config.split('_')[0] for config in config_metrics.keys())
        for precision in precisions:
            precision_configs = {k: v for k, v in config_metrics.items() if k.startswith(precision)}
            if precision_configs:
                avg_metrics = {}
                for metric in ["embedding_l2_mean", "embedding_cosine_mean", "retrieval_jaccard", "retrieval_exact_match"]:
                    avg_metrics[metric] = np.mean([config[metric] for config in precision_configs.values()])
                analysis["precision_effects"][precision] = avg_metrics

        # Analyze deterministic vs non-deterministic
        det_configs = {k: v for k, v in config_metrics.items() if k.endswith('True')}
        nondet_configs = {k: v for k, v in config_metrics.items() if k.endswith('False')}

        if det_configs:
            analysis["deterministic_effects"]["deterministic"] = {
                metric: np.mean([config[metric] for config in det_configs.values()])
                for metric in ["embedding_l2_mean", "embedding_cosine_mean", "retrieval_jaccard", "retrieval_exact_match"]
            }

        if nondet_configs:
            analysis["deterministic_effects"]["non_deterministic"] = {
                metric: np.mean([config[metric] for config in nondet_configs.values()])
                for metric in ["embedding_l2_mean", "embedding_cosine_mean", "retrieval_jaccard", "retrieval_exact_match"]
            }

        # Calculate correlations between embedding stability and retrieval reproducibility
        if len(config_metrics) > 2:
            emb_l2_values = [metrics["embedding_l2_mean"] for metrics in config_metrics.values()]
            retrieval_jaccard_values = [metrics["retrieval_jaccard"] for metrics in config_metrics.values()]

            if len(set(emb_l2_values)) > 1 and len(set(retrieval_jaccard_values)) > 1:
                correlation = np.corrcoef(emb_l2_values, retrieval_jaccard_values)[0, 1]
                analysis["stability_correlation"]["l2_vs_jaccard"] = correlation

        return analysis

    def generate_comprehensive_report(self, results: Dict[str, Any], output_dir: str = "integrated_analysis"):
        """Generate comprehensive report combining embedding and retrieval analysis"""

        os.makedirs(output_dir, exist_ok=True)

        # Save raw results
        with open(f"{output_dir}/integrated_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Generate markdown report
        report_path = f"{output_dir}/integrated_reproducibility_report.md"

        with open(report_path, 'w') as f:
            f.write("# Integrated RAG Reproducibility Analysis Report\n\n")
            f.write("This report analyzes reproducibility from embedding generation through retrieval results.\n\n")

            # Executive Summary
            f.write("## Executive Summary\n\n")

            # Find most stable configuration
            best_config = None
            best_stability = float('inf')

            for config_name, data in results["embedding_stability"].items():
                # Handle both process isolation and in-process result structures
                if "metrics" in data:
                    # Process isolation result structure
                    metrics = data["metrics"]
                else:
                    # In-process result structure
                    metrics = data["documents"]["metrics"]

                l2_mean = metrics.get("l2_distance", {}).get("mean", float('inf'))
                if l2_mean < best_stability:
                    best_stability = l2_mean
                    best_config = config_name

            f.write(f"**Most stable embedding configuration**: {best_config} (L2 distance: {best_stability:.2e})\n\n")

            # Embedding Stability Analysis
            f.write("## Embedding Stability Analysis\n\n")
            f.write("| Configuration | L2 Distance | Cosine Similarity | Max Abs Diff | Exact Match Rate |\n")
            f.write("|---------------|-------------|-------------------|--------------|------------------|\n")

            for config_name, data in results["embedding_stability"].items():
                # Handle both process isolation and in-process result structures
                if "metrics" in data:
                    # Process isolation result structure
                    metrics = data["metrics"]
                else:
                    # In-process result structure
                    metrics = data["documents"]["metrics"]

                l2_mean = metrics.get("l2_distance", {}).get("mean", 0)
                cos_mean = metrics.get("cosine_similarity", {}).get("mean", 0)
                max_diff = metrics.get("max_abs_difference", {}).get("mean", 0)
                exact_match = metrics.get("exact_match_rate", 0)

                f.write(f"| {config_name} | {l2_mean:.2e} | {cos_mean:.6f} | {max_diff:.2e} | {exact_match:.3f} |\n")

            f.write("\n")

            # Retrieval Reproducibility Analysis
            f.write("## Retrieval Reproducibility Analysis\n\n")
            f.write("| Configuration | Exact Match | Jaccard | Kendall Tau |\n")
            f.write("|---------------|-------------|---------|-------------|\n")

            for config_name, data in results["retrieval_reproducibility"].items():
                metrics = data["retrieval_metrics"]
                exact_match = metrics.get("exact_match", {}).get("exact_match_rate", 0)
                jaccard = metrics.get("overlap", {}).get("mean_jaccard", 0)
                kendall = metrics.get("rank_correlation", {}).get("mean_kendall_tau", 0)

                f.write(f"| {config_name} | {exact_match:.3f} | {jaccard:.3f} | {kendall:.3f} |\n")

            f.write("\n")

            # Cross-configuration Analysis
            if "end_to_end_analysis" in results:
                f.write("## Cross-Configuration Analysis\n\n")

                analysis = results["end_to_end_analysis"]

                # Precision effects
                if "precision_effects" in analysis:
                    f.write("### Precision Effects\n\n")
                    for precision, metrics in analysis["precision_effects"].items():
                        f.write(f"**{precision.upper()}**:\n")
                        f.write(f"- Embedding L2 distance: {metrics['embedding_l2_mean']:.2e}\n")
                        f.write(f"- Retrieval Jaccard: {metrics['retrieval_jaccard']:.3f}\n\n")

                # Deterministic effects
                if "deterministic_effects" in analysis:
                    f.write("### Deterministic vs Non-Deterministic\n\n")
                    for mode, metrics in analysis["deterministic_effects"].items():
                        f.write(f"**{mode.replace('_', ' ').title()}**:\n")
                        f.write(f"- Embedding L2 distance: {metrics['embedding_l2_mean']:.2e}\n")
                        f.write(f"- Retrieval Jaccard: {metrics['retrieval_jaccard']:.3f}\n\n")

            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("Based on the integrated analysis:\n\n")
            f.write("1. **For maximum stability**: Use FP32 precision with deterministic mode\n")
            f.write("2. **For performance**: FP16 may be acceptable if stability requirements are moderate\n")
            f.write("3. **For production**: Monitor embedding drift alongside retrieval metrics\n")
            f.write("4. **For reproducibility**: Enable deterministic mode at both embedding and retrieval levels\n")

        logger.info(f"Comprehensive report generated: {report_path}")


def test_integrated_reproducibility(csv_path: str = None,
                                   num_docs: int = 1000,
                                   num_queries: int = 50,
                                   n_runs: int = 3,
                                   use_process_isolation: bool = False):
    """Test function for integrated reproducibility analysis using real data

    Args:
        csv_path: Path to CSV dataset file
        num_docs: Number of documents to use
        num_queries: Number of queries to generate
        n_runs: Number of test runs
        use_process_isolation: If True, run each test in a separate Python process for maximum isolation
    """

    logger.info("Starting integrated reproducibility test with real data")

    # Validate local model before proceeding
    local_model_path = "/scratch/user/u.bw269205/shared_models/bge_model"
    if not check_local_model(local_model_path):
        logger.error("Local model validation failed. Please check the model path and files.")
        return None

    # Configure RAG system
    rag_config = ExperimentConfig(
        index_type="Flat",
        deterministic_mode=True,
        use_gpu=True,
        top_k=10
    )

    # Configure embedding tests with local model
    local_model_path = "/scratch/user/u.bw269205/shared_models/bge_model"

    embedding_configs = [
        create_local_model_config(local_model_path, precision="fp32", deterministic=True),
        create_local_model_config(local_model_path, precision="fp32", deterministic=False),
        create_local_model_config(local_model_path, precision="fp16", deterministic=True),
        create_local_model_config(local_model_path, precision="fp16", deterministic=False),
    ]

    # Add TF32 and BF16 if supported
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        embedding_configs.extend([
            create_local_model_config(local_model_path, precision="tf32", deterministic=True),
            create_local_model_config(local_model_path, precision="tf32", deterministic=False),
            create_local_model_config(local_model_path, precision="bf16", deterministic=True),
            create_local_model_config(local_model_path, precision="bf16", deterministic=False),
        ])
        logger.info("Added TF32 and BF16 precision tests (both deterministic modes) (Ampere GPU detected)")

    # Create integrated tester
    tester = IntegratedRAGReproducibilityTester(rag_config, embedding_configs)

    # Load real data
    documents, queries = tester.load_msmarco_data(csv_path, num_docs, num_queries)

    # Print sample data for verification
    logger.info("\n" + "="*60)
    logger.info("DATASET VERIFICATION")
    logger.info("="*60)
    logger.info(f"Total documents: {len(documents)}")
    logger.info(f"Total queries: {len(queries)}")
    logger.info(f"Process isolation: {'Enabled' if use_process_isolation else 'Disabled'}")

    if documents:
        sample_doc = documents[0]
        logger.info(f"\nSample document:")
        logger.info(f"  ID: {sample_doc['id']}")
        logger.info(f"  Text: {sample_doc['text'][:150]}...")
        logger.info(f"  Metadata: {sample_doc['metadata']}")

    if queries:
        logger.info(f"\nSample queries:")
        for i, query in enumerate(queries[:3]):
            logger.info(f"  {i+1}: {query}")

    logger.info("="*60)

    # Run integrated test with optional process isolation
    logger.info(f"Starting integrated reproducibility analysis with {n_runs} runs...")
    results = tester.test_end_to_end_reproducibility(
        documents, queries, n_runs=n_runs, use_process_isolation=use_process_isolation
    )

    # Generate report
    output_dir = f"ms_marco_analysis_{'isolated' if use_process_isolation else 'inprocess'}"
    tester.generate_comprehensive_report(results, output_dir)

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("="*60)

    for config_name, data in results["embedding_stability"].items():
        # Handle both process isolation and in-process result structures
        if "metrics" in data:
            # Process isolation result structure
            doc_metrics = data["metrics"]
        else:
            # In-process result structure
            doc_metrics = data["documents"]["metrics"]

        ret_metrics = results["retrieval_reproducibility"][config_name]["retrieval_metrics"]

        l2_mean = doc_metrics.get("l2_distance", {}).get("mean", 0)
        cos_mean = doc_metrics.get("cosine_similarity", {}).get("mean", 1)
        jaccard = ret_metrics.get("overlap", {}).get("mean_jaccard", 0)
        exact_match = ret_metrics.get("exact_match", {}).get("exact_match_rate", 0)

        logger.info(f"\n{config_name}:")
        logger.info(f"  Embedding L2 distance: {l2_mean:.2e}")
        logger.info(f"  Embedding cosine sim:   {cos_mean:.6f}")
        logger.info(f"  Retrieval Jaccard:      {jaccard:.3f}")
        logger.info(f"  Retrieval exact match:  {exact_match:.3f}")

    logger.info("\n" + "="*60)
    logger.info("✅ Integrated reproducibility test completed!")
    logger.info(f"📁 Results saved to: {output_dir}/")
    logger.info(f"📊 Report: {output_dir}/integrated_reproducibility_report.md")
    logger.info(f"🔬 Process isolation: {'Enabled' if use_process_isolation else 'Disabled'}")
    logger.info("="*60)

    return results


def check_local_model(model_path: str) -> bool:
    """Check if local model exists and contains required files

    Args:
        model_path: Path to local model directory

    Returns:
        True if model appears valid, False otherwise
    """
    if not os.path.exists(model_path):
        logger.error(f"Model directory does not exist: {model_path}")
        return False

    # Check for common SentenceTransformer files
    required_files = [
        "config.json",
        "pytorch_model.bin"  # or model.safetensors
    ]

    optional_files = [
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json"
    ]

    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(model_path, file)):
            missing_files.append(file)

    if missing_files:
        # Check if safetensors exists instead of pytorch_model.bin
        if "pytorch_model.bin" in missing_files:
            if os.path.exists(os.path.join(model_path, "model.safetensors")):
                missing_files.remove("pytorch_model.bin")

    if missing_files:
        logger.error(f"Missing required model files in {model_path}: {missing_files}")
        return False

    logger.info(f"✅ Local model validation passed: {model_path}")

    # List available files for verification
    available_files = [f for f in os.listdir(model_path) if os.path.isfile(os.path.join(model_path, f))]
    logger.info(f"Available model files: {available_files}")

    return True


def test_with_custom_msmarco(csv_path: str):
    """Convenience function to test with a specific MS MARCO CSV file"""

    if not os.path.exists(csv_path):
        logger.error(f"MS MARCO CSV file not found: {csv_path}")
        logger.info("Please ensure the file exists and try again.")
        return None

    logger.info(f"Testing with MS MARCO CSV: {csv_path}")

    return test_integrated_reproducibility(
        csv_path=csv_path,
        num_docs=2000,
        num_queries=100,
        n_runs=3
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_integrated_reproducibility()
