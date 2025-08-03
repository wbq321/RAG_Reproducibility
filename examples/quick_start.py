#!/usr/bin/env python3
"""
Quick Start Script for Integrated RAG Reproducibility Testing
Provides easy-to-use interface for running embedding + retrieval reproducibility tests
"""

import os
import sys
import yaml
import argparse
import logging
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

logger = logging.getLogger(__name__)


def setup_logging(level="INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"quick_start_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        ]
    )


def check_dependencies():
    """Check if all required dependencies are available"""
    missing_deps = []

    try:
        import torch
        if not torch.cuda.is_available():
            logger.warning("CUDA not available - GPU tests will be skipped")
    except ImportError:
        missing_deps.append("torch")

    try:
        import faiss
    except ImportError:
        missing_deps.append("faiss-gpu or faiss-cpu")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        missing_deps.append("sentence-transformers")

    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        missing_deps.append("numpy, pandas, matplotlib, seaborn")

    if missing_deps:
        logger.error("Missing dependencies: " + ", ".join(missing_deps))
        logger.error("Please install missing packages and try again")
        return False

    logger.info("✅ All dependencies available")
    return True


def quick_embedding_test():
    """Quick embedding reproducibility test"""
    from embedding_reproducibility_tester import EmbeddingReproducibilityTester, EmbeddingConfig

    logger.info("Running quick embedding reproducibility test...")

    # Simple test queries
    test_texts = [
        "This is a test sentence for embedding reproducibility.",
        "Another example text to check consistency.",
        "Machine learning models should produce stable outputs.",
        "Reproducibility is crucial for scientific research.",
        "Testing different precision modes in deep learning."
    ]

    # Test FP32 vs FP16
    configs = [
        EmbeddingConfig(precision="fp32", deterministic=True),
        EmbeddingConfig(precision="fp16", deterministic=True),
    ]

    results = {}
    for config in configs:
        config_name = f"{config.precision}_det"
        logger.info(f"Testing {config_name}...")

        tester = EmbeddingReproducibilityTester(config)
        result = tester.test_embedding_stability(test_texts, n_runs=3)
        results[config_name] = result

        # Log key metrics
        metrics = result["metrics"]
        l2_mean = metrics.get("l2_distance", {}).get("mean", 0)
        cos_mean = metrics.get("cosine_similarity", {}).get("mean", 1)

        logger.info(f"  L2 distance: {l2_mean:.2e}")
        logger.info(f"  Cosine similarity: {cos_mean:.6f}")

    return results


def quick_retrieval_test():
    """Quick retrieval reproducibility test"""
    from rag_reproducibility_framework import ExperimentConfig, FaissRetrieval, ReproducibilityMetrics

    logger.info("Running quick retrieval reproducibility test...")

    # Create simple test documents
    documents = []
    for i in range(100):
        documents.append({
            "id": f"doc_{i}",
            "text": f"This is test document {i} about topic {i % 10}.",
            "metadata": {"topic": i % 10}
        })

    queries = [
        "Find documents about topic 1",
        "Search for information on topic 5",
        "Look for content related to topic 8"
    ]

    # Test deterministic vs non-deterministic
    configs = [
        ExperimentConfig(index_type="Flat", deterministic_mode=True),
        ExperimentConfig(index_type="Flat", deterministic_mode=False),
    ]

    results = {}
    for config in configs:
        config_name = f"flat_det_{config.deterministic_mode}"
        logger.info(f"Testing {config_name}...")

        runs = []
        for run_idx in range(3):
            retrieval = FaissRetrieval(config)
            retrieval.index_documents(documents)
            run_results = retrieval.search(queries)
            runs.append(run_results)
            retrieval.reset()

        metrics = ReproducibilityMetrics.calculate_all_metrics(runs)
        results[config_name] = metrics

        # Log key metrics
        exact_match = metrics.get("exact_match", {}).get("exact_match_rate", 0)
        jaccard = metrics.get("overlap", {}).get("mean_jaccard", 0)

        logger.info(f"  Exact match rate: {exact_match:.3f}")
        logger.info(f"  Mean Jaccard: {jaccard:.3f}")

    return results


def quick_integrated_test():
    """Quick integrated embedding + retrieval test"""
    from embedding_reproducibility_tester import EmbeddingConfig, IntegratedRAGReproducibilityTester
    from rag_reproducibility_framework import ExperimentConfig

    logger.info("Running quick integrated test...")

    # Small test dataset
    documents = []
    for i in range(50):
        documents.append({
            "id": f"doc_{i}",
            "text": f"Document {i} discusses artificial intelligence and machine learning applications in domain {i % 5}.",
            "metadata": {"domain": i % 5}
        })

    queries = [
        "artificial intelligence applications",
        "machine learning in various domains"
    ]

    # Configure systems
    rag_config = ExperimentConfig(index_type="Flat", deterministic_mode=True)
    embedding_configs = [
        EmbeddingConfig(precision="fp32", deterministic=True),
        EmbeddingConfig(precision="fp16", deterministic=True),
    ]

    # Run integrated test
    tester = IntegratedRAGReproducibilityTester(rag_config, embedding_configs)
    results = tester.test_end_to_end_reproducibility(documents, queries, n_runs=2)

    # Log summary
    for config_name in results["embedding_stability"].keys():
        emb_metrics = results["embedding_stability"][config_name]["documents"]["metrics"]
        ret_metrics = results["retrieval_reproducibility"][config_name]["retrieval_metrics"]

        l2_mean = emb_metrics.get("l2_distance", {}).get("mean", 0)
        jaccard = ret_metrics.get("overlap", {}).get("mean_jaccard", 0)

        logger.info(f"{config_name}: L2={l2_mean:.2e}, Jaccard={jaccard:.3f}")

    return results


def run_diagnostic():
    """Run system diagnostics"""
    logger.info("Running system diagnostics...")

    # Check GPU availability
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU count: {torch.cuda.device_count()}")
            logger.info(f"Current GPU: {torch.cuda.get_device_name()}")

            # Check GPU capabilities
            capability = torch.cuda.get_device_capability()
            logger.info(f"GPU compute capability: {capability}")
            if capability[0] >= 8:
                logger.info("✅ Ampere GPU detected - TF32/BF16 precision tests available")
            else:
                logger.info("⚠️  Pre-Ampere GPU - TF32/BF16 precision tests not available")
    except Exception as e:
        logger.error(f"GPU check failed: {e}")

    # Check FAISS
    try:
        import faiss
        logger.info(f"FAISS version: {faiss.__version__ if hasattr(faiss, '__version__') else 'unknown'}")
        logger.info(f"FAISS GPU support: {faiss.get_num_gpus() > 0}")
        if faiss.get_num_gpus() > 0:
            logger.info(f"FAISS GPU count: {faiss.get_num_gpus()}")
    except Exception as e:
        logger.error(f"FAISS check failed: {e}")

    # Check model availability
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        logger.info("✅ Default sentence transformer model loaded successfully")
    except Exception as e:
        logger.error(f"Model loading failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Quick Start for RAG Reproducibility Testing")
    parser.add_argument("--test", choices=["embedding", "retrieval", "integrated", "all"],
                       default="all", help="Which test to run")
    parser.add_argument("--diagnostic", action="store_true",
                       help="Run system diagnostic checks")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Logging level")
    parser.add_argument("--output-dir", default="quick_test_results",
                       help="Output directory for results")

    args = parser.parse_args()

    # Setup
    setup_logging(args.log_level)
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("🚀 Quick Start - RAG Reproducibility Testing")
    logger.info("=" * 50)

    # Run diagnostics if requested
    if args.diagnostic:
        run_diagnostic()
        return

    # Check dependencies
    if not check_dependencies():
        return

    # Run tests
    try:
        if args.test in ["embedding", "all"]:
            embedding_results = quick_embedding_test()

        if args.test in ["retrieval", "all"]:
            retrieval_results = quick_retrieval_test()

        if args.test in ["integrated", "all"]:
            integrated_results = quick_integrated_test()

        logger.info("=" * 50)
        logger.info("✅ Quick tests completed successfully!")
        logger.info("🔬 For comprehensive testing, run: python run_comprehensive_tests.py")
        logger.info("📁 For full integration: see embedding_uncertainty/ folder")

    except Exception as e:
        logger.error(f"Quick test failed: {str(e)}")
        logger.error("Try running with --diagnostic to check system configuration")
        raise


if __name__ == "__main__":
    main()
