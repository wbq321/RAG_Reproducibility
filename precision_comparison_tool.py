"""
Precision Comparison Tool for Embedding Analysis
Calculates embedding differences between different float precision settings
"""

import numpy as np
import torch
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PrecisionConfig:
    """Configuration for precision testing"""
    name: str
    dtype: str  # "fp32", "fp16", "bf16", "tf32"
    torch_dtype: torch.dtype
    autocast: bool = False
    description: str = ""

class EmbeddingPrecisionAnalyzer:
    """Analyze embedding differences across different precision settings"""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.precision_configs = self._create_precision_configs()

    def _create_precision_configs(self) -> List[PrecisionConfig]:
        """Create precision configurations to test"""
        configs = [
            PrecisionConfig(
                name="FP32",
                dtype="fp32",
                torch_dtype=torch.float32,
                autocast=False,
                description="32-bit floating point (baseline)"
            ),
            PrecisionConfig(
                name="FP16",
                dtype="fp16",
                torch_dtype=torch.float16,
                autocast=True,
                description="16-bit floating point with autocast"
            )
        ]

        # Add BF16 if supported
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            configs.append(PrecisionConfig(
                name="BF16",
                dtype="bf16",
                torch_dtype=torch.bfloat16,
                autocast=True,
                description="Brain floating point 16-bit (Ampere+ GPUs)"
            ))

        # Add TF32 configuration (default on Ampere+ but can be controlled)
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            configs.append(PrecisionConfig(
                name="TF32",
                dtype="tf32",
                torch_dtype=torch.float32,  # TF32 uses FP32 tensors but TF32 ops
                autocast=False,
                description="TensorFloat-32 (enabled by default on Ampere+)"
            ))

        return configs

    def encode_with_precision(self, texts: List[str], config: PrecisionConfig) -> np.ndarray:
        """Encode texts with specific precision configuration"""
        logger.info(f"Encoding with {config.name} precision...")

        # Load fresh model instance to avoid state carryover
        model = SentenceTransformer(self.model_path, device=self.device)

        # Configure precision settings
        if config.name == "TF32":
            # TF32 control
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        else:
            # Disable TF32 for other precisions for clean comparison
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

        # Convert model to target precision
        if config.torch_dtype != torch.float32:
            model = model.to(config.torch_dtype)

        # Encode with appropriate autocast
        with torch.no_grad():
            if config.autocast:
                with torch.cuda.amp.autocast(dtype=config.torch_dtype):
                    embeddings = model.encode(
                        texts,
                        batch_size=32,
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                        show_progress_bar=False
                    )
            else:
                embeddings = model.encode(
                    texts,
                    batch_size=32,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )

        # Clean up
        del model
        torch.cuda.empty_cache()

        return embeddings.astype(np.float32)  # Convert back to FP32 for analysis

    def calculate_embedding_differences(self, embeddings1: np.ndarray, embeddings2: np.ndarray,
                                      config1: str, config2: str) -> Dict[str, Any]:
        """Calculate comprehensive differences between two embedding sets"""

        # L2 (Euclidean) distance
        l2_distances = np.linalg.norm(embeddings1 - embeddings2, axis=1)

        # Cosine similarity
        cosine_similarities = []
        for i in range(len(embeddings1)):
            cos_sim = np.dot(embeddings1[i], embeddings2[i]) / (
                np.linalg.norm(embeddings1[i]) * np.linalg.norm(embeddings2[i])
            )
            cosine_similarities.append(cos_sim)
        cosine_similarities = np.array(cosine_similarities)

        # Element-wise absolute differences
        abs_differences = np.abs(embeddings1 - embeddings2)

        # Relative error (where embeddings1 != 0)
        relative_errors = []
        for i in range(len(embeddings1)):
            mask = embeddings1[i] != 0
            if np.any(mask):
                rel_err = np.abs((embeddings1[i][mask] - embeddings2[i][mask]) / embeddings1[i][mask])
                relative_errors.extend(rel_err)
        relative_errors = np.array(relative_errors)

        # Statistical analysis
        results = {
            "comparison": f"{config1} vs {config2}",
            "num_embeddings": len(embeddings1),
            "embedding_dim": embeddings1.shape[1],

            # L2 distance statistics
            "l2_distance": {
                "mean": float(np.mean(l2_distances)),
                "std": float(np.std(l2_distances)),
                "min": float(np.min(l2_distances)),
                "max": float(np.max(l2_distances)),
                "median": float(np.median(l2_distances)),
                "percentile_95": float(np.percentile(l2_distances, 95)),
                "percentile_99": float(np.percentile(l2_distances, 99))
            },

            # Cosine similarity statistics
            "cosine_similarity": {
                "mean": float(np.mean(cosine_similarities)),
                "std": float(np.std(cosine_similarities)),
                "min": float(np.min(cosine_similarities)),
                "max": float(np.max(cosine_similarities)),
                "median": float(np.median(cosine_similarities))
            },

            # Element-wise difference statistics
            "element_wise_diff": {
                "mean_abs_diff": float(np.mean(abs_differences)),
                "max_abs_diff": float(np.max(abs_differences)),
                "std_abs_diff": float(np.std(abs_differences)),
                "fraction_zero_diff": float(np.mean(abs_differences == 0)),
                "fraction_small_diff": float(np.mean(abs_differences < 1e-6))
            },

            # Relative error statistics (if applicable)
            "relative_error": {
                "mean": float(np.mean(relative_errors)) if len(relative_errors) > 0 else 0,
                "max": float(np.max(relative_errors)) if len(relative_errors) > 0 else 0,
                "std": float(np.std(relative_errors)) if len(relative_errors) > 0 else 0,
                "median": float(np.median(relative_errors)) if len(relative_errors) > 0 else 0
            }
        }

        return results

    def run_comprehensive_analysis(self, texts: List[str], output_dir: str = "precision_analysis") -> Dict[str, Any]:
        """Run comprehensive precision comparison analysis"""

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        logger.info(f"Starting precision analysis with {len(texts)} texts")
        logger.info(f"Available precisions: {[c.name for c in self.precision_configs]}")

        # Generate embeddings for each precision
        embeddings_dict = {}
        for config in self.precision_configs:
            try:
                embeddings = self.encode_with_precision(texts, config)
                embeddings_dict[config.name] = embeddings
                logger.info(f"✅ {config.name}: Shape {embeddings.shape}")
            except Exception as e:
                logger.error(f"❌ Failed to generate {config.name} embeddings: {e}")
                continue

        if len(embeddings_dict) < 2:
            raise ValueError("Need at least 2 successful precision configurations for comparison")

        # Compare all pairs
        results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model_path": self.model_path,
                "num_texts": len(texts),
                "device": self.device,
                "precision_configs": [
                    {
                        "name": c.name,
                        "description": c.description,
                        "dtype": c.dtype
                    } for c in self.precision_configs if c.name in embeddings_dict
                ]
            },
            "pairwise_comparisons": {},
            "summary": {}
        }

        # Pairwise comparisons
        precision_names = list(embeddings_dict.keys())
        for i, prec1 in enumerate(precision_names):
            for j, prec2 in enumerate(precision_names):
                if i < j:  # Avoid duplicates
                    comparison_key = f"{prec1}_vs_{prec2}"
                    logger.info(f"Comparing {prec1} vs {prec2}...")

                    diff_results = self.calculate_embedding_differences(
                        embeddings_dict[prec1],
                        embeddings_dict[prec2],
                        prec1,
                        prec2
                    )

                    results["pairwise_comparisons"][comparison_key] = diff_results

        # Generate summary
        results["summary"] = self._generate_summary(results["pairwise_comparisons"])

        # Save results
        results_file = output_path / "precision_analysis_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {results_file}")

        # Generate visualizations
        self._create_visualizations(embeddings_dict, results, output_path)

        # Generate report
        self._generate_report(results, output_path)

        return results

    def _generate_summary(self, comparisons: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics across all comparisons"""
        summary = {
            "most_similar_pair": None,
            "least_similar_pair": None,
            "l2_distance_stats": {},
            "cosine_similarity_stats": {}
        }

        l2_means = []
        cosine_means = []

        best_cosine = -1
        worst_cosine = 2

        for comp_name, comp_data in comparisons.items():
            l2_mean = comp_data["l2_distance"]["mean"]
            cosine_mean = comp_data["cosine_similarity"]["mean"]

            l2_means.append(l2_mean)
            cosine_means.append(cosine_mean)

            if cosine_mean > best_cosine:
                best_cosine = cosine_mean
                summary["most_similar_pair"] = comp_name

            if cosine_mean < worst_cosine:
                worst_cosine = cosine_mean
                summary["least_similar_pair"] = comp_name

        summary["l2_distance_stats"] = {
            "mean": float(np.mean(l2_means)),
            "std": float(np.std(l2_means)),
            "min": float(np.min(l2_means)),
            "max": float(np.max(l2_means))
        }

        summary["cosine_similarity_stats"] = {
            "mean": float(np.mean(cosine_means)),
            "std": float(np.std(cosine_means)),
            "min": float(np.min(cosine_means)),
            "max": float(np.max(cosine_means))
        }

        return summary

    def _create_visualizations(self, embeddings_dict: Dict[str, np.ndarray],
                             results: Dict[str, Any], output_path: Path):
        """Create visualization plots"""

        try:
            # Set up the plotting style
            plt.style.use('default')
            sns.set_palette("husl")

            # 1. L2 Distance Heatmap
            precision_names = list(embeddings_dict.keys())
            n_prec = len(precision_names)
            l2_matrix = np.zeros((n_prec, n_prec))

            for i, prec1 in enumerate(precision_names):
                for j, prec2 in enumerate(precision_names):
                    if i == j:
                        l2_matrix[i, j] = 0
                    elif i < j:
                        comp_key = f"{prec1}_vs_{prec2}"
                        if comp_key in results["pairwise_comparisons"]:
                            l2_dist = results["pairwise_comparisons"][comp_key]["l2_distance"]["mean"]
                            l2_matrix[i, j] = l2_dist
                            l2_matrix[j, i] = l2_dist

            plt.figure(figsize=(10, 8))
            sns.heatmap(l2_matrix,
                       xticklabels=precision_names,
                       yticklabels=precision_names,
                       annot=True,
                       fmt='.2e',
                       cmap='YlOrRd')
            plt.title('Mean L2 Distance Between Precision Settings')
            plt.tight_layout()
            plt.savefig(output_path / "l2_distance_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()

            # 2. Cosine Similarity Heatmap
            cosine_matrix = np.ones((n_prec, n_prec))

            for i, prec1 in enumerate(precision_names):
                for j, prec2 in enumerate(precision_names):
                    if i != j and i < j:
                        comp_key = f"{prec1}_vs_{prec2}"
                        if comp_key in results["pairwise_comparisons"]:
                            cosine_sim = results["pairwise_comparisons"][comp_key]["cosine_similarity"]["mean"]
                            cosine_matrix[i, j] = cosine_sim
                            cosine_matrix[j, i] = cosine_sim

            plt.figure(figsize=(10, 8))
            sns.heatmap(cosine_matrix,
                       xticklabels=precision_names,
                       yticklabels=precision_names,
                       annot=True,
                       fmt='.6f',
                       cmap='RdYlGn',
                       vmin=0.99, vmax=1.0)
            plt.title('Mean Cosine Similarity Between Precision Settings')
            plt.tight_layout()
            plt.savefig(output_path / "cosine_similarity_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()

            # 3. Distribution plots for key comparisons
            if "FP32_vs_FP16" in results["pairwise_comparisons"]:
                comp_data = results["pairwise_comparisons"]["FP32_vs_FP16"]

                fig, axes = plt.subplots(2, 2, figsize=(15, 12))

                # L2 distance distribution (would need raw data)
                axes[0,0].text(0.5, 0.5, f"L2 Distance\nMean: {comp_data['l2_distance']['mean']:.2e}\nStd: {comp_data['l2_distance']['std']:.2e}",
                              ha='center', va='center', transform=axes[0,0].transAxes, fontsize=12)
                axes[0,0].set_title("L2 Distance: FP32 vs FP16")

                # Cosine similarity
                axes[0,1].text(0.5, 0.5, f"Cosine Similarity\nMean: {comp_data['cosine_similarity']['mean']:.6f}\nStd: {comp_data['cosine_similarity']['std']:.6f}",
                              ha='center', va='center', transform=axes[0,1].transAxes, fontsize=12)
                axes[0,1].set_title("Cosine Similarity: FP32 vs FP16")

                # Element-wise differences
                axes[1,0].text(0.5, 0.5, f"Element-wise Differences\nMean: {comp_data['element_wise_diff']['mean_abs_diff']:.2e}\nMax: {comp_data['element_wise_diff']['max_abs_diff']:.2e}",
                              ha='center', va='center', transform=axes[1,0].transAxes, fontsize=12)
                axes[1,0].set_title("Element-wise Absolute Differences")

                # Relative error
                axes[1,1].text(0.5, 0.5, f"Relative Error\nMean: {comp_data['relative_error']['mean']:.2e}\nMax: {comp_data['relative_error']['max']:.2e}",
                              ha='center', va='center', transform=axes[1,1].transAxes, fontsize=12)
                axes[1,1].set_title("Relative Error")

                plt.tight_layout()
                plt.savefig(output_path / "fp32_vs_fp16_analysis.png", dpi=300, bbox_inches='tight')
                plt.close()

            logger.info("✅ Visualizations saved successfully")

        except Exception as e:
            logger.warning(f"⚠️ Could not create visualizations: {e}")

    def _generate_report(self, results: Dict[str, Any], output_path: Path):
        """Generate a comprehensive text report"""

        report_file = output_path / "precision_analysis_report.md"

        with open(report_file, 'w') as f:
            f.write("# 🔬 Embedding Precision Analysis Report\n\n")
            f.write(f"**Generated**: {results['metadata']['timestamp']}\n")
            f.write(f"**Model**: {results['metadata']['model_path']}\n")
            f.write(f"**Device**: {results['metadata']['device']}\n")
            f.write(f"**Number of texts**: {results['metadata']['num_texts']}\n\n")

            f.write("## 📊 Precision Configurations Tested\n\n")
            for config in results['metadata']['precision_configs']:
                f.write(f"- **{config['name']}** ({config['dtype']}): {config['description']}\n")
            f.write("\n")

            f.write("## 🔍 Summary Results\n\n")
            summary = results['summary']
            f.write(f"- **Most similar precision pair**: {summary['most_similar_pair']}\n")
            f.write(f"- **Least similar precision pair**: {summary['least_similar_pair']}\n")
            f.write(f"- **Average L2 distance across comparisons**: {summary['l2_distance_stats']['mean']:.2e}\n")
            f.write(f"- **Average cosine similarity across comparisons**: {summary['cosine_similarity_stats']['mean']:.6f}\n\n")

            f.write("## 📈 Detailed Pairwise Comparisons\n\n")

            for comp_name, comp_data in results['pairwise_comparisons'].items():
                f.write(f"### {comp_name.replace('_', ' ')}\n\n")
                f.write(f"**L2 Distance Statistics:**\n")
                l2_stats = comp_data['l2_distance']
                f.write(f"- Mean: {l2_stats['mean']:.2e}\n")
                f.write(f"- Std: {l2_stats['std']:.2e}\n")
                f.write(f"- Min: {l2_stats['min']:.2e}\n")
                f.write(f"- Max: {l2_stats['max']:.2e}\n")
                f.write(f"- 95th percentile: {l2_stats['percentile_95']:.2e}\n\n")

                f.write(f"**Cosine Similarity Statistics:**\n")
                cos_stats = comp_data['cosine_similarity']
                f.write(f"- Mean: {cos_stats['mean']:.6f}\n")
                f.write(f"- Std: {cos_stats['std']:.6f}\n")
                f.write(f"- Min: {cos_stats['min']:.6f}\n")
                f.write(f"- Max: {cos_stats['max']:.6f}\n\n")

                f.write(f"**Element-wise Difference Statistics:**\n")
                elem_stats = comp_data['element_wise_diff']
                f.write(f"- Mean absolute difference: {elem_stats['mean_abs_diff']:.2e}\n")
                f.write(f"- Max absolute difference: {elem_stats['max_abs_diff']:.2e}\n")
                f.write(f"- Fraction of zero differences: {elem_stats['fraction_zero_diff']:.4f}\n")
                f.write(f"- Fraction of small differences (<1e-6): {elem_stats['fraction_small_diff']:.4f}\n\n")

                if comp_data['relative_error']['mean'] > 0:
                    f.write(f"**Relative Error Statistics:**\n")
                    rel_stats = comp_data['relative_error']
                    f.write(f"- Mean: {rel_stats['mean']:.2e}\n")
                    f.write(f"- Max: {rel_stats['max']:.2e}\n")
                    f.write(f"- Median: {rel_stats['median']:.2e}\n\n")

                f.write("---\n\n")

        logger.info(f"📄 Report saved to {report_file}")

def main():
    """Example usage"""

    # Configuration
    model_path = "/scratch/user/u.bw269205/shared_models/bge_model"  # Your local model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Test texts (you can customize these)
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning models require large amounts of training data.",
        "Transformer architectures have revolutionized NLP tasks.",
        "Vector embeddings capture semantic meaning in high-dimensional space.",
        "Reproducibility is crucial for scientific research.",
        "GPU acceleration significantly speeds up neural network training.",
        "Precision formats affect both performance and accuracy in computation.",
        "RAG systems combine retrieval and generation for enhanced performance."
    ]

    # Run analysis
    analyzer = EmbeddingPrecisionAnalyzer(model_path, device)

    try:
        results = analyzer.run_comprehensive_analysis(
            texts=test_texts,
            output_dir="precision_analysis_results"
        )

        print("\n🎉 Analysis completed successfully!")
        print(f"📁 Results saved to: precision_analysis_results/")
        print(f"📊 Most similar pair: {results['summary']['most_similar_pair']}")
        print(f"📊 Least similar pair: {results['summary']['least_similar_pair']}")

    except Exception as e:
        print(f"❌ Analysis failed: {e}")

if __name__ == "__main__":
    main()
