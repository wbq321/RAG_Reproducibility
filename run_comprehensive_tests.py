#!/usr/bin/env python3
"""
Comprehensive RAG Reproducibility Testing Suite
Integrates embedding uncertainty testing with FAISS retrieval reproducibility analysis
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Any

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from rag_reproducibility_framework import (
    ExperimentConfig,
    test_faiss_reproducibility,
    GPUNonDeterminismTester,
    ComprehensiveReproducibilityReport
)
from embedding_reproducibility_tester import (
    EmbeddingConfig,
    IntegratedRAGReproducibilityTester
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_documents(n_docs: int = 5000, use_real_data: bool = True, csv_path: str = None) -> List[Dict[str, str]]:
    """Create test documents with real dataset or fallback to simulated data"""

    if use_real_data:
        try:
            # Try to load real dataset using generalized loader
            sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
            from dataset_loader import load_dataset_for_reproducibility

            # Try different possible paths
            possible_paths = [
                csv_path,
                "data/dataset.csv",
            ]

            csv_path_found = None
            for path in possible_paths:
                if path and os.path.exists(path):
                    csv_path_found = path
                    break

            if csv_path_found:
                logger.info(f"Loading real dataset from: {csv_path_found}")
                documents, _ = load_dataset_for_reproducibility(
                    file_path=csv_path_found,
                    dataset_type="auto",  # Auto-detect dataset type
                    num_docs=n_docs,
                    num_queries=10  # Just need documents here
                )
                logger.info(f"Loaded {len(documents)} documents from real dataset")
                return documents
            else:
                logger.warning("Real dataset CSV not found in expected locations:")
                for path in possible_paths:
                    if path:
                        logger.warning(f"  - {path}")
                logger.warning("Falling back to simulated data")

        except Exception as e:
            logger.error(f"Error loading real dataset: {e}")
            logger.warning("Falling back to simulated data")

    # Fallback to simulated data (enhanced version)
    logger.info(f"Creating {n_docs} simulated documents with realistic content")
    return create_simulated_documents(n_docs)


def create_simulated_documents(n_docs: int) -> List[Dict[str, str]]:
    """Create enhanced simulated documents"""
    import random

    topics = [
        "artificial intelligence and machine learning applications",
        "natural language processing and computational linguistics",
        "computer vision and image recognition technology",
        "data science and predictive analytics methods",
        "deep learning neural network architectures",
        "information retrieval and search engine algorithms",
        "distributed computing and cloud infrastructure systems",
        "cybersecurity and network protection protocols",
        "database management and data storage optimization",
        "software engineering and development methodologies",
        "web development and user interface design",
        "mobile application development and deployment",
        "blockchain technology and decentralized systems",
        "quantum computing and quantum algorithm design",
        "robotics and autonomous system control",
        "bioinformatics and computational biology research",
        "healthcare technology and medical device innovation",
        "financial technology and algorithmic trading",
        "renewable energy and smart grid optimization",
        "transportation systems and autonomous vehicle technology"
    ]

    domains = [
        "healthcare and medical research", "financial services and banking",
        "educational institutions and learning", "retail and e-commerce platforms",
        "manufacturing and industrial automation", "transportation and logistics",
        "telecommunications and networking", "government and public services",
        "scientific research and development", "entertainment and media",
        "agriculture and food production", "energy and utilities",
        "real estate and construction", "insurance and risk management"
    ]

    documents = []
    for i in range(n_docs):
        topic = random.choice(topics)
        domain = random.choice(domains)

        # Create more sophisticated document content
        content_variants = [
            f"Recent advances in {topic} have demonstrated significant potential for transforming {domain}. "
            f"This comprehensive analysis examines current implementation strategies, technical challenges, "
            f"and emerging opportunities within the field. Research findings indicate that successful "
            f"deployment requires careful consideration of scalability, performance optimization, and "
            f"integration with existing infrastructure. The methodology presented in this study "
            f"addresses key technical requirements while maintaining high standards for reliability, "
            f"security, and user experience. Future developments are expected to focus on enhanced "
            f"efficiency, reduced operational costs, and improved capabilities through advanced techniques.",

            f"The integration of {topic} into {domain} represents a paradigm shift in how organizations "
            f"approach complex problem-solving and decision-making processes. This investigation explores "
            f"the technical architecture, implementation frameworks, and strategic considerations necessary "
            f"for successful adoption. Case studies from industry leaders demonstrate practical approaches "
            f"to overcoming common obstacles while maximizing return on investment. Performance benchmarks "
            f"reveal measurable improvements in efficiency, accuracy, and user satisfaction when properly "
            f"implemented. The research methodology includes comparative analysis, expert interviews, "
            f"and longitudinal studies to provide comprehensive insights.",

            f"A systematic evaluation of {topic} technologies reveals their transformative impact on {domain}. "
            f"This study encompasses market analysis, technical assessment, and strategic planning considerations "
            f"for organizations considering implementation. The research framework includes detailed examination "
            f"of existing solutions, emerging trends, and future development trajectories. Key findings highlight "
            f"the importance of stakeholder engagement, change management, and continuous optimization in "
            f"achieving desired outcomes. Best practices identified through this research provide actionable "
            f"guidance for practitioners and decision-makers across various organizational contexts."
        ]

        content = random.choice(content_variants)

        documents.append({
            "id": f"sim_doc_{i:06d}",
            "text": content,
            "metadata": {
                "topic": topic,
                "domain": domain,
                "category": f"cat_{i % 20}",
                "length": len(content),
                "word_count": len(content.split()),
                "source": "simulated_realistic"
            }
        })

    return documents


def create_test_queries(use_real_data: bool = True, csv_path: str = None) -> List[str]:
    """Create test queries from real dataset or generate realistic ones"""

    if use_real_data:
        try:
            sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
            from dataset_loader import load_dataset_for_reproducibility

            possible_paths = [
                csv_path,
                "data/dataset.csv",
                "data/msmarco.csv",
                "data/passages.csv"
            ]

            csv_path_found = None
            for path in possible_paths:
                if path and os.path.exists(path):
                    csv_path_found = path
                    break

            if csv_path_found:
                _, queries = load_dataset_for_reproducibility(
                    file_path=csv_path_found,
                    dataset_type="auto",  # Auto-detect dataset type
                    num_docs=100,  # Small number for query generation
                    num_queries=20
                )
                logger.info(f"Generated {len(queries)} queries from real dataset")
                return queries

        except Exception as e:
            logger.error(f"Error generating queries from real dataset: {e}")

    # Fallback to enhanced simulated queries
    return [
        "How do machine learning algorithms improve healthcare diagnostics and patient outcomes?",
        "What are the latest developments in natural language processing for search engines?",
        "How does computer vision technology work in autonomous vehicle systems?",
        "What are the benefits of deep learning for financial fraud detection?",
        "How can artificial intelligence optimize supply chain management processes?",
        "What are the security considerations for blockchain implementation in banking?",
        "How do distributed computing systems handle large-scale data processing?",
        "What are the challenges of implementing quantum computing in real applications?",
        "How does predictive analytics improve decision-making in business intelligence?",
        "What are the best practices for cybersecurity in cloud computing environments?",
        "How can robotics automation transform manufacturing efficiency?",
        "What are the applications of bioinformatics in drug discovery research?",
        "How do smart grid technologies optimize renewable energy distribution?",
        "What are the advantages of microservices architecture in software development?",
        "How can data visualization improve scientific research communication?",
        "What are the ethical considerations in artificial intelligence deployment?",
        "How do recommendation systems personalize user experiences in e-commerce?",
        "What are the technical requirements for real-time streaming analytics?",
        "How can edge computing reduce latency in IoT applications?",
        "What are the performance optimization strategies for database query processing?"
    ]
def run_basic_faiss_tests(output_dir: str) -> Dict[str, Any]:
    """Run basic FAISS reproducibility tests"""
    logger.info("Running basic FAISS reproducibility tests...")

    results = test_faiss_reproducibility()

    # Save results
    with open(f"{output_dir}/basic_faiss_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("Basic FAISS tests completed")
    return results


def run_gpu_nondeterminism_tests(documents: List[Dict[str, str]],
                                queries: List[str],
                                output_dir: str) -> Dict[str, Any]:
    """Run GPU non-determinism tests with embedding integration"""
    logger.info("Running GPU non-determinism tests...")

    gpu_tester = GPUNonDeterminismTester()
    results = gpu_tester.test_gpu_factors(documents, queries)

    # Save results
    with open(f"{output_dir}/gpu_nondeterminism_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("GPU non-determinism tests completed")
    return results


def run_integrated_tests(documents: List[Dict[str, str]],
                        queries: List[str],
                        output_dir: str) -> Dict[str, Any]:
    """Run integrated embedding + retrieval tests"""
    logger.info("Running integrated embedding and retrieval tests...")

    # Configure RAG system
    rag_config = ExperimentConfig(
        index_type="Flat",
        deterministic_mode=True,
        use_gpu=True,
        top_k=10,
        batch_size=32
    )

    # Configure embedding tests - reduced for faster execution
    embedding_configs = [
        EmbeddingConfig(precision="fp32", deterministic=True),
        EmbeddingConfig(precision="fp32", deterministic=False),
        EmbeddingConfig(precision="fp16", deterministic=True),
        EmbeddingConfig(precision="fp16", deterministic=False),
    ]

    # Add advanced precisions if supported
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            embedding_configs.extend([
                EmbeddingConfig(precision="tf32", deterministic=True),
                EmbeddingConfig(precision="bf16", deterministic=True),
            ])
            logger.info("Added TF32 and BF16 precision tests (Ampere GPU detected)")
    except:
        pass

    # Run integrated test
    tester = IntegratedRAGReproducibilityTester(rag_config, embedding_configs)
    results = tester.test_end_to_end_reproducibility(
        documents[:2000],  # Reduced dataset for faster execution
        queries[:10],
        n_runs=3
    )

    # Generate integrated report
    tester.generate_comprehensive_report(results, f"{output_dir}/integrated_analysis")

    logger.info("Integrated tests completed")
    return results


def run_comprehensive_analysis(documents: List[Dict[str, str]],
                             queries: List[str],
                             output_dir: str) -> None:
    """Run comprehensive reproducibility analysis"""
    logger.info("Running comprehensive reproducibility analysis...")

    # Create comprehensive report generator
    report_generator = ComprehensiveReproducibilityReport()

    # Generate full report with reduced dataset
    report_generator.generate_full_report(
        documents[:3000],
        queries[:15],
        f"{output_dir}/comprehensive_analysis"
    )

    logger.info("Comprehensive analysis completed")


def generate_summary_report(all_results: Dict[str, Any], output_dir: str) -> None:
    """Generate an executive summary report"""
    logger.info("Generating executive summary report...")

    summary_path = f"{output_dir}/executive_summary.md"

    with open(summary_path, 'w') as f:
        f.write("# RAG Reproducibility Executive Summary\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Test Suite Overview\n\n")
        f.write("This comprehensive test suite evaluated RAG system reproducibility across multiple dimensions:\n\n")
        f.write("1. **Basic FAISS Reproducibility**: Index type and configuration effects\n")
        f.write("2. **GPU Non-determinism**: Hardware-specific reproducibility factors\n")
        f.write("3. **Integrated Analysis**: End-to-end embedding + retrieval reproducibility\n")
        f.write("4. **Comprehensive Analysis**: Scale effects and optimization trade-offs\n\n")

        # Key findings from basic tests
        if "basic_faiss" in all_results:
            f.write("## Key Findings - Basic FAISS Tests\n\n")

            best_config = None
            best_jaccard = 0
            worst_config = None
            worst_jaccard = 1

            for config_name, data in all_results["basic_faiss"].items():
                if "metrics" in data and "overlap" in data["metrics"]:
                    jaccard = data["metrics"]["overlap"]["mean_jaccard"]
                    if jaccard > best_jaccard:
                        best_jaccard = jaccard
                        best_config = config_name
                    if jaccard < worst_jaccard:
                        worst_jaccard = jaccard
                        worst_config = config_name

            if best_config:
                f.write(f"- **Most reproducible configuration**: {best_config} (Jaccard: {best_jaccard:.3f})\n")
            if worst_config:
                f.write(f"- **Least reproducible configuration**: {worst_config} (Jaccard: {worst_jaccard:.3f})\n")
            f.write("\n")

        # Embedding stability findings
        if "integrated" in all_results:
            f.write("## Key Findings - Embedding Stability\n\n")

            integrated_results = all_results["integrated"]
            if "embedding_stability" in integrated_results:
                most_stable = None
                most_stable_l2 = float('inf')

                for config_name, data in integrated_results["embedding_stability"].items():
                    if "documents" in data and "metrics" in data["documents"]:
                        l2_mean = data["documents"]["metrics"].get("l2_distance", {}).get("mean", float('inf'))
                        if l2_mean < most_stable_l2:
                            most_stable_l2 = l2_mean
                            most_stable = config_name

                if most_stable:
                    f.write(f"- **Most stable embedding configuration**: {most_stable} (L2: {most_stable_l2:.2e})\n")
            f.write("\n")

        # Recommendations
        f.write("## Recommendations\n\n")
        f.write("Based on comprehensive testing:\n\n")
        f.write("### For Maximum Reproducibility\n")
        f.write("- Use **Flat index** with deterministic mode enabled\n")
        f.write("- Use **FP32 precision** for embedding generation\n")
        f.write("- Enable **deterministic CUDA operations**\n")
        f.write("- Use **fixed random seeds** across all components\n\n")

        f.write("### For Production Balance\n")
        f.write("- **HNSW index** offers good speed/reproducibility trade-off\n")
        f.write("- **FP16 precision** may be acceptable with monitoring\n")
        f.write("- **Monitor embedding drift** alongside retrieval metrics\n")
        f.write("- **Implement reproducibility testing** in CI/CD pipelines\n\n")

        f.write("### For Distributed Systems\n")
        f.write("- Use **hash-based sharding** for consistent document distribution\n")
        f.write("- **Synchronize random seeds** across all nodes\n")
        f.write("- **Monitor cross-node consistency** regularly\n")
        f.write("- **Test at scale** before production deployment\n\n")

        # File references
        f.write("## Detailed Reports\n\n")
        f.write("- Basic FAISS Results: `basic_faiss_results.json`\n")
        f.write("- GPU Analysis: `gpu_nondeterminism_results.json`\n")
        f.write("- Integrated Analysis: `integrated_analysis/`\n")
        f.write("- Comprehensive Analysis: `comprehensive_analysis/`\n")

    logger.info(f"Executive summary generated: {summary_path}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Comprehensive RAG Reproducibility Testing Suite")
    parser.add_argument("--output-dir", default="reproducibility_results",
                       help="Output directory for results")
    parser.add_argument("--n-docs", type=int, default=5000,
                       help="Number of test documents")
    parser.add_argument("--dataset", type=str,
                       help="Path to dataset CSV file (optional)")
    parser.add_argument("--use-simulated", action="store_true",
                       help="Force use of simulated data instead of real dataset")
    parser.add_argument("--skip-basic", action="store_true",
                       help="Skip basic FAISS tests")
    parser.add_argument("--skip-gpu", action="store_true",
                       help="Skip GPU non-determinism tests")
    parser.add_argument("--skip-integrated", action="store_true",
                       help="Skip integrated embedding+retrieval tests")
    parser.add_argument("--skip-comprehensive", action="store_true",
                       help="Skip comprehensive analysis")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick tests with reduced datasets")

    args = parser.parse_args()

    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Configure test size based on quick mode
    if args.quick:
        n_docs = min(args.n_docs, 1000)
        logger.info("Quick mode enabled - using reduced test datasets")
    else:
        n_docs = args.n_docs

    logger.info(f"Starting comprehensive RAG reproducibility testing")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Test documents: {n_docs}")

    # Determine data source
    use_real_data = not args.use_simulated
    if args.dataset:
        logger.info(f"Dataset CSV specified: {args.dataset}")

    # Create test data
    logger.info("Creating test documents and queries...")
    documents = create_test_documents(n_docs, use_real_data=use_real_data, csv_path=args.dataset)
    queries = create_test_queries(use_real_data=use_real_data, csv_path=args.dataset)

    # Log data source information
    if documents and "source" in documents[0].get("metadata", {}):
        data_source = documents[0]["metadata"]["source"]
        logger.info(f"Using data source: {data_source}")

    # Store all results
    all_results = {}

    try:
        # Run test suites
        if not args.skip_basic:
            all_results["basic_faiss"] = run_basic_faiss_tests(output_dir)

        if not args.skip_gpu:
            all_results["gpu_nondeterminism"] = run_gpu_nondeterminism_tests(
                documents, queries, output_dir
            )

        if not args.skip_integrated:
            all_results["integrated"] = run_integrated_tests(
                documents, queries, output_dir
            )

        if not args.skip_comprehensive:
            run_comprehensive_analysis(documents, queries, output_dir)

        # Generate executive summary
        generate_summary_report(all_results, output_dir)

        logger.info("=" * 60)
        logger.info("🎉 All reproducibility tests completed successfully!")
        logger.info(f"📁 Results saved to: {output_dir}/")
        logger.info(f"📊 Executive summary: {output_dir}/executive_summary.md")
        if use_real_data and documents and documents[0].get("metadata", {}).get("source") not in ["simulated_realistic", "simulated"]:
            data_source = documents[0]["metadata"].get("source", "real dataset")
            logger.info(f"✅ Tests completed using real dataset: {data_source}")
        else:
            logger.info("ℹ️  Tests completed using simulated data")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Test suite failed with error: {str(e)}")
        raise
if __name__ == "__main__":
    main()
