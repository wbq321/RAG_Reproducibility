#!/usr/bin/env python3
"""
Cross-Model Retrieval Ranking Correlation Analysis

This script analyzes the ranking consistency across different embedding models (BGE, E5, QW)
using FAISS retrieval as a downstream task. It focuses on comparing how different models
rank the same documents for the same queries, rather than comparing precision impact.

Features:
- FAISS Flat index retrieval for consistent comparison
- MSMARCO dataset integration
- Top-50 retrieval analysis
- Four ranking correlation metrics
- Fixed precision (FP32) to isolate model differences
- Publication-ready visualizations with 30pt fonts
"""

import os
import sys

# Set environment for offline mode (cluster compatibility)
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

import json
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Add the necessary directories to the path
script_dir = Path(__file__).parent / "scripts"
src_dir = Path(__file__).parent / "src"
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

# Import existing modules
from embedding_reproducibility_tester import EmbeddingReproducibilityTester, EmbeddingConfig
from dataset_loader import load_dataset_for_reproducibility

# PyTorch for device detection
try:
    import torch
    def get_device():
        return "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    print("PyTorch not found. Defaulting to CPU.")
    def get_device():
        return "cpu"

# FAISS for retrieval
try:
    import faiss
except ImportError:
    print("FAISS not found. Please install with: pip install faiss-cpu")
    sys.exit(1)

# Scipy for correlation calculations
try:
    from scipy.stats import kendalltau, spearmanr
    from scipy.spatial.distance import cosine
except ImportError:
    print("SciPy not found. Please install with: pip install scipy")
    sys.exit(1)

# Set publication-ready 30pt fonts
plt.rcParams.update({
    'font.size': 30,
    'axes.titlesize': 30,
    'axes.labelsize': 30,
    'xtick.labelsize': 30,
    'ytick.labelsize': 30,
    'legend.fontsize': 30,
    'figure.titlesize': 30,
    'font.family': 'DejaVu Sans',
    'figure.figsize': (28, 14),
    'axes.linewidth': 2,
    'grid.linewidth': 1.5,
    'lines.linewidth': 3,
    'patch.linewidth': 2,
    'xtick.major.width': 2,
    'ytick.major.width': 2,
    'xtick.minor.width': 1.5,
    'ytick.minor.width': 1.5
})

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cross_model_retrieval_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Container for retrieval results"""
    query_id: str
    model_name: str
    retrieved_doc_ids: List[str]
    retrieval_scores: List[float]
    retrieval_time: float

@dataclass
class RankingCorrelation:
    """Container for ranking correlation results"""
    model_pair: Tuple[str, str]
    kendall_tau: float
    kendall_p_value: float
    spearman_rho: float
    spearman_p_value: float
    rank_biased_overlap: float
    overlap_coefficient: float
    overlap_origin: Dict[str, float]

class CrossModelRetrievalAnalyzer:
    """
    Analyzes ranking consistency across different embedding models using FAISS retrieval
    """
    
    def __init__(self, 
                 models_to_test: List[str] = ["bge", "e5", "qw"],
                 top_k: int = 50,
                 data_dir: str = "data",
                 output_dir: str = "results",
                 model_base_path: str = "/scratch/user/u.bw269205/shared_models"):
        """
        Initialize the cross-model retrieval analyzer
        
        Args:
            models_to_test: List of model names to compare
            top_k: Number of top documents to retrieve for ranking comparison
            data_dir: Directory containing MSMARCO data
            output_dir: Directory to save results
            model_base_path: Base path where local models are stored
        """
        self.models_to_test = models_to_test
        self.top_k = top_k
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.model_base_path = model_base_path
        
        # Model name mapping to local paths
        self.model_paths = {
            "bge": f"{model_base_path}/bge_model",
            "e5": f"{model_base_path}/intfloat_e5-base-v2", 
            "qw": f"{model_base_path}/Qwen_Qwen3-Embedding-0.6B"
        }
        
        # Initialize components
        # Create a default embedding config for the tester
        default_config = EmbeddingConfig(
            model_name="bge",  # Default model, will be overridden per model
            precision="fp32",
            deterministic=True,
            device=get_device(),
            batch_size=32,
            max_length=512,
            normalize_embeddings=True
        )
        self.tester = EmbeddingReproducibilityTester(default_config)
        
        # Storage for results
        self.documents = []
        self.queries = []
        self.embeddings = {}  # model_name -> embeddings
        self.faiss_indices = {}  # model_name -> faiss index
        self.retrieval_results = {}  # query_id -> {model_name -> RetrievalResult}
        self.correlations = []  # List of RankingCorrelation objects
        
        logger.info(f"Initialized CrossModelRetrievalAnalyzer")
        logger.info(f"Models to test: {models_to_test}")
        logger.info(f"Top-K retrieval: {top_k}")
        logger.info(f"Fixed precision: FP32")
    
    def load_msmarco_data(self, max_documents: int = 5000, max_queries: int = 100) -> Tuple[List[Dict], List[str]]:
        """
        Load MSMARCO dataset using existing infrastructure
        """
        logger.info("Loading MSMARCO dataset...")
        
        # Try multiple possible MSMARCO file locations
        possible_paths = [
            self.data_dir / "ms_marco_passages.csv",
            self.data_dir / "msmarco.csv",
            self.data_dir / "passages.csv",
            Path("data/ms_marco_passages.csv"),
            Path("data/msmarco.csv"),
            Path("data/passages.csv")
        ]
        
        msmarco_path = None
        for path in possible_paths:
            if path.exists():
                msmarco_path = path
                break
        
        if msmarco_path is None:
            raise FileNotFoundError(
                f"MSMARCO dataset not found. Please place it in one of: {possible_paths}"
            )
        
        logger.info(f"Found MSMARCO data at: {msmarco_path}")
        
        # Load using existing dataset loader
        documents, queries = load_dataset_for_reproducibility(
            file_path=str(msmarco_path),
            dataset_type="ms_marco",
            num_docs=max_documents,
            num_queries=max_queries,
            data_dir=str(self.data_dir),
            doc_sampling="diverse",
            query_strategy="first_sentence"
        )
        
        self.documents = documents
        self.queries = queries
        
        logger.info(f"Loaded {len(documents)} documents and {len(queries)} queries")
        return documents, queries
    
    def generate_embeddings_for_all_models(self) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for all models with fixed FP32 precision
        """
        logger.info("Generating embeddings for all models...")
        
        # Prepare document texts
        doc_texts = [doc["text"] for doc in self.documents]
        
        for model_name in self.models_to_test:
            logger.info(f"Generating embeddings with {model_name}...")
            start_time = time.time()
            
            # Get local model path
            model_path = self.model_paths.get(model_name, model_name)
            logger.info(f"Using model path: {model_path}")
            
            # Create model-specific config
            model_config = EmbeddingConfig(
                model_name=model_path,  # Use local path instead of model name
                precision="fp32",  # Fixed precision for consistency
                deterministic=True,
                device=get_device(),
                batch_size=32,
                max_length=512,
                normalize_embeddings=True
            )
            
            # Create a new tester instance for this model
            model_tester = EmbeddingReproducibilityTester(model_config)
            
            # Generate embeddings using the tester
            embeddings = model_tester.encode_texts(doc_texts)
            
            # Ensure FP32 and normalize for FAISS
            embeddings = embeddings.astype(np.float32)
            # Normalize for cosine similarity in FAISS
            faiss.normalize_L2(embeddings)
            
            self.embeddings[model_name] = embeddings
            
            end_time = time.time()
            logger.info(f"Generated {model_name} embeddings: shape={embeddings.shape}, "
                       f"dtype={embeddings.dtype}, time={end_time-start_time:.2f}s")
        
        return self.embeddings
    
    def build_faiss_indices(self) -> Dict[str, faiss.Index]:
        """
        Build FAISS Flat indices for all models
        """
        logger.info("Building FAISS Flat indices...")
        
        for model_name, embeddings in self.embeddings.items():
            logger.info(f"Building Flat index for {model_name}...")
            
            # Use Flat index for exact search (no approximation)
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)  # Inner Product for normalized vectors
            
            # Add embeddings to index
            index.add(embeddings)
            
            self.faiss_indices[model_name] = index
            
            logger.info(f"Built {model_name} index: {index.ntotal} vectors, dimension={dimension}")
        
        return self.faiss_indices
    
    def perform_retrieval_for_all_models(self) -> Dict[str, Dict[str, RetrievalResult]]:
        """
        Perform retrieval for all queries across all models
        """
        logger.info(f"Performing top-{self.top_k} retrieval for all models...")
        
        # Generate query embeddings for each model
        query_embeddings = {}
        for model_name in self.models_to_test:
            logger.info(f"Generating query embeddings for {model_name}...")
            
            # Get local model path
            model_path = self.model_paths.get(model_name, model_name)
            logger.info(f"Using model path: {model_path}")
            
            # Create model-specific config
            model_config = EmbeddingConfig(
                model_name=model_path,  # Use local path instead of model name
                precision="fp32",  # Fixed precision for consistency
                deterministic=True,
                device=get_device(),
                batch_size=32,
                max_length=512,
                normalize_embeddings=True
            )
            
            # Create a new tester instance for this model
            model_tester = EmbeddingReproducibilityTester(model_config)
            
            # Generate query embeddings
            query_embs = model_tester.encode_texts(self.queries)
            
            # Ensure FP32 and normalize
            query_embs = query_embs.astype(np.float32)
            faiss.normalize_L2(query_embs)
            
            query_embeddings[model_name] = query_embs
        
        # Perform retrieval
        for query_idx, query in enumerate(self.queries):
            query_id = f"query_{query_idx:05d}"
            self.retrieval_results[query_id] = {}
            
            for model_name in self.models_to_test:
                start_time = time.time()
                
                # Get query embedding
                query_embedding = query_embeddings[model_name][query_idx:query_idx+1]
                
                # Search using FAISS
                scores, indices = self.faiss_indices[model_name].search(query_embedding, self.top_k)
                
                # Convert to document IDs
                retrieved_doc_ids = [self.documents[idx]["id"] for idx in indices[0]]
                retrieval_scores = scores[0].tolist()
                
                end_time = time.time()
                
                # Store result
                result = RetrievalResult(
                    query_id=query_id,
                    model_name=model_name,
                    retrieved_doc_ids=retrieved_doc_ids,
                    retrieval_scores=retrieval_scores,
                    retrieval_time=end_time - start_time
                )
                
                self.retrieval_results[query_id][model_name] = result
        
        logger.info(f"Completed retrieval for {len(self.queries)} queries across {len(self.models_to_test)} models")
        return self.retrieval_results
    
    def calculate_rank_biased_overlap(self, list1: List[str], list2: List[str], p: float = 0.9) -> float:
        """
        Calculate Rank-Biased Overlap (RBO) between two ranked lists
        
        Args:
            list1, list2: Ranked lists of document IDs
            p: Persistence parameter (0 < p < 1)
        """
        if not list1 or not list2:
            return 0.0
        
        # Convert to sets for faster lookup
        set1 = set(list1)
        set2 = set(list2)
        
        # Calculate overlap at each rank
        overlap_sum = 0.0
        for k in range(1, min(len(list1), len(list2)) + 1):
            # Items in top-k of both lists
            top_k_1 = set(list1[:k])
            top_k_2 = set(list2[:k])
            
            # Intersection size
            intersection_size = len(top_k_1.intersection(top_k_2))
            
            # Add to weighted sum
            overlap_sum += (p ** (k-1)) * (intersection_size / k)
        
        # Normalize
        normalizer = sum(p ** (k-1) for k in range(1, min(len(list1), len(list2)) + 1))
        
        if normalizer > 0:
            return overlap_sum / normalizer
        else:
            return 0.0
    
    def calculate_overlap_coefficient(self, list1: List[str], list2: List[str]) -> float:
        """
        Calculate overlap coefficient between two ranked lists at top-k
        """
        if not list1 or not list2:
            return 0.0
        
        set1 = set(list1)
        set2 = set(list2)
        
        intersection_size = len(set1.intersection(set2))
        min_size = min(len(set1), len(set2))
        
        if min_size > 0:
            return intersection_size / min_size
        else:
            return 0.0
    
    def calculate_kendall_tau(self, list1: List[str], list2: List[str]) -> Tuple[float, float]:
        """
        Calculate Kendall's Tau correlation for common items
        """
        if not list1 or not list2:
            return 0.0, 1.0
        
        # Find common items
        set1 = set(list1)
        set2 = set(list2)
        common_items = set1.intersection(set2)
        
        if len(common_items) < 2:
            return 0.0, 1.0
        
        # Create rank mappings for common items
        rank1 = {item: idx for idx, item in enumerate(list1) if item in common_items}
        rank2 = {item: idx for idx, item in enumerate(list2) if item in common_items}
        
        # Get ranks for common items
        ranks1 = [rank1[item] for item in common_items]
        ranks2 = [rank2[item] for item in common_items]
        
        # Calculate Kendall's Tau
        tau, p_value = kendalltau(ranks1, ranks2)
        return tau, p_value
    
    def analyze_overlap_origin(self, list1: List[str], list2: List[str]) -> Dict[str, float]:
        """
        Analyze where overlapping items come from in the rankings
        """
        if not list1 or not list2:
            return {"top_10": 0.0, "top_20": 0.0, "top_50": 0.0}
        
        set1 = set(list1)
        set2 = set(list2)
        common_items = set1.intersection(set2)
        
        if not common_items:
            return {"top_10": 0.0, "top_20": 0.0, "top_50": 0.0}
        
        # Count overlaps in different rank bands
        overlaps = {"top_10": 0, "top_20": 0, "top_50": 0}
        
        for item in common_items:
            # Find ranks in both lists
            rank1 = list1.index(item) if item in list1 else float('inf')
            rank2 = list2.index(item) if item in list2 else float('inf')
            
            # Count if both are in top-10
            if rank1 < 10 and rank2 < 10:
                overlaps["top_10"] += 1
            
            # Count if both are in top-20
            if rank1 < 20 and rank2 < 20:
                overlaps["top_20"] += 1
            
            # Count if both are in top-50
            if rank1 < 50 and rank2 < 50:
                overlaps["top_50"] += 1
        
        # Convert to percentages
        total_common = len(common_items)
        return {
            "top_10": overlaps["top_10"] / total_common if total_common > 0 else 0.0,
            "top_20": overlaps["top_20"] / total_common if total_common > 0 else 0.0,
            "top_50": overlaps["top_50"] / total_common if total_common > 0 else 0.0
        }
    
    def calculate_ranking_correlations(self) -> List[RankingCorrelation]:
        """
        Calculate all ranking correlation metrics between model pairs
        """
        logger.info("Calculating ranking correlations...")
        
        correlations = []
        
        # Compare all pairs of models
        for i, model1 in enumerate(self.models_to_test):
            for j, model2 in enumerate(self.models_to_test):
                if i >= j:  # Skip duplicate and self-comparisons
                    continue
                
                logger.info(f"Calculating correlations between {model1} and {model2}...")
                
                # Collect correlation data across all queries
                kendall_taus = []
                kendall_p_values = []
                spearman_rhos = []
                spearman_p_values = []
                rbos = []
                overlap_coeffs = []
                overlap_origins = {"top_10": [], "top_20": [], "top_50": []}
                
                for query_id in self.retrieval_results:
                    result1 = self.retrieval_results[query_id][model1]
                    result2 = self.retrieval_results[query_id][model2]
                    
                    list1 = result1.retrieved_doc_ids
                    list2 = result2.retrieved_doc_ids
                    
                    # Calculate metrics
                    tau, tau_p = self.calculate_kendall_tau(list1, list2)
                    rbo = self.calculate_rank_biased_overlap(list1, list2)
                    overlap_coeff = self.calculate_overlap_coefficient(list1, list2)
                    overlap_origin = self.analyze_overlap_origin(list1, list2)
                    
                    # Store values
                    kendall_taus.append(tau)
                    kendall_p_values.append(tau_p)
                    rbos.append(rbo)
                    overlap_coeffs.append(overlap_coeff)
                    
                    for key in overlap_origins:
                        overlap_origins[key].append(overlap_origin[key])
                
                # Average the metrics across queries
                avg_correlation = RankingCorrelation(
                    model_pair=(model1, model2),
                    kendall_tau=np.mean(kendall_taus),
                    kendall_p_value=np.mean(kendall_p_values),
                    spearman_rho=0.0,  # Placeholder - complex to calculate across queries
                    spearman_p_value=1.0,
                    rank_biased_overlap=np.mean(rbos),
                    overlap_coefficient=np.mean(overlap_coeffs),
                    overlap_origin={key: np.mean(values) for key, values in overlap_origins.items()}
                )
                
                correlations.append(avg_correlation)
                
                logger.info(f"{model1} vs {model2}: "
                           f"Kendall's τ={avg_correlation.kendall_tau:.4f}, "
                           f"RBO={avg_correlation.rank_biased_overlap:.4f}, "
                           f"Overlap={avg_correlation.overlap_coefficient:.4f}")
        
        self.correlations = correlations
        return correlations
    
    def create_correlation_visualizations(self):
        """
        Create publication-ready visualizations of ranking correlations
        """
        logger.info("Creating correlation visualizations...")
        
        if not self.correlations:
            logger.warning("No correlations to visualize")
            return
        
        # Prepare data for visualization
        model_pairs = [f"{corr.model_pair[0]} vs {corr.model_pair[1]}" for corr in self.correlations]
        kendall_taus = [corr.kendall_tau for corr in self.correlations]
        rbos = [corr.rank_biased_overlap for corr in self.correlations]
        overlap_coeffs = [corr.overlap_coefficient for corr in self.correlations]
        
        # Create comprehensive correlation plot
        fig, axes = plt.subplots(2, 2, figsize=(40, 28))
        fig.suptitle('Cross-Model Retrieval Ranking Correlations', fontsize=36, y=0.95)
        
        # 1. Kendall's Tau
        axes[0, 0].bar(model_pairs, kendall_taus, color='steelblue', alpha=0.8)
        axes[0, 0].set_title("Kendall's Tau Correlation", fontsize=32)
        axes[0, 0].set_ylabel("Kendall's τ", fontsize=30)
        axes[0, 0].set_ylim(-1, 1)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for i, v in enumerate(kendall_taus):
            axes[0, 0].text(i, v + 0.05, f'{v:.3f}', ha='center', va='bottom', fontsize=24)
        
        # 2. Rank-Biased Overlap
        axes[0, 1].bar(model_pairs, rbos, color='forestgreen', alpha=0.8)
        axes[0, 1].set_title("Rank-Biased Overlap", fontsize=32)
        axes[0, 1].set_ylabel("RBO", fontsize=30)
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for i, v in enumerate(rbos):
            axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=24)
        
        # 3. Overlap Coefficient
        axes[1, 0].bar(model_pairs, overlap_coeffs, color='darkorange', alpha=0.8)
        axes[1, 0].set_title("Overlap Coefficient", fontsize=32)
        axes[1, 0].set_ylabel("Overlap Coefficient", fontsize=30)
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for i, v in enumerate(overlap_coeffs):
            axes[1, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=24)
        
        # 4. Overlap Origin Analysis
        top_10_overlaps = [corr.overlap_origin["top_10"] for corr in self.correlations]
        top_20_overlaps = [corr.overlap_origin["top_20"] for corr in self.correlations]
        top_50_overlaps = [corr.overlap_origin["top_50"] for corr in self.correlations]
        
        x = np.arange(len(model_pairs))
        width = 0.25
        
        axes[1, 1].bar(x - width, top_10_overlaps, width, label='Top-10', color='crimson', alpha=0.8)
        axes[1, 1].bar(x, top_20_overlaps, width, label='Top-20', color='gold', alpha=0.8)
        axes[1, 1].bar(x + width, top_50_overlaps, width, label='Top-50', color='purple', alpha=0.8)
        
        axes[1, 1].set_title("Overlap Origin Analysis", fontsize=32)
        axes[1, 1].set_ylabel("Proportion of Common Items", fontsize=30)
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(model_pairs, rotation=45)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].legend(fontsize=26)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "cross_model_ranking_correlations.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved correlation visualization to {plot_path}")
        
        # Create distribution plots
        self._create_correlation_distribution_plots()
        
        # Create correlation matrix heatmap
        self._create_correlation_matrix_heatmap()
    
    def _create_correlation_distribution_plots(self):
        """
        Create violin plots showing the distribution of correlation metrics across queries
        """
        logger.info("Creating correlation distribution plots...")
        
        # Collect per-query correlation data for all model pairs
        all_kendall_taus = []
        all_rbos = []
        all_overlap_coeffs = []
        pair_labels = []
        
        # Recalculate individual query metrics for distribution analysis
        for i, model1 in enumerate(self.models_to_test):
            for j, model2 in enumerate(self.models_to_test):
                if i >= j:  # Skip duplicate and self-comparisons
                    continue
                
                pair_label = f"{model1} vs {model2}"
                kendall_taus = []
                rbos = []
                overlap_coeffs = []
                
                for query_id in self.retrieval_results:
                    result1 = self.retrieval_results[query_id][model1]
                    result2 = self.retrieval_results[query_id][model2]
                    
                    list1 = result1.retrieved_doc_ids
                    list2 = result2.retrieved_doc_ids
                    
                    # Calculate metrics for this query
                    tau, _ = self.calculate_kendall_tau(list1, list2)
                    rbo = self.calculate_rank_biased_overlap(list1, list2)
                    overlap_coeff = self.calculate_overlap_coefficient(list1, list2)
                    
                    kendall_taus.append(tau)
                    rbos.append(rbo)
                    overlap_coeffs.append(overlap_coeff)
                
                # Store all values for this pair
                all_kendall_taus.extend(kendall_taus)
                all_rbos.extend(rbos)
                all_overlap_coeffs.extend(overlap_coeffs)
                pair_labels.extend([pair_label] * len(kendall_taus))
        
        # Prepare data for plotting
        # Create DataFrame for easier plotting
        df = pd.DataFrame({
            'Kendall_Tau': all_kendall_taus,
            'RBO': all_rbos,
            'Overlap_Coefficient': all_overlap_coeffs,
            'Model_Pair': pair_labels
        })
        
        # 1. Kendall's Tau Violin Plot
        plt.figure(figsize=(28, 14))
        sns.violinplot(data=df, x='Model_Pair', y='Kendall_Tau', 
                      palette="viridis", inner="box")
        plt.title("Kendall's Tau Distribution Across Queries", fontsize=32)
        plt.xlabel("Model Pairs", fontsize=30)
        plt.ylabel("Kendall's τ", fontsize=30)
        plt.ylim(-1, 1)
        plt.xticks(rotation=45, fontsize=28)
        plt.yticks(fontsize=28)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save Kendall's Tau violin plot
        kendall_path = self.output_dir / "kendall_tau_violin.png"
        plt.savefig(kendall_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved Kendall's Tau violin plot to {kendall_path}")
        
        # 2. RBO Violin Plot
        plt.figure(figsize=(28, 14))
        sns.violinplot(data=df, x='Model_Pair', y='RBO', 
                      palette="viridis", inner="box")
        plt.title("Rank-Biased Overlap Distribution Across Queries", fontsize=32)
        plt.xlabel("Model Pairs", fontsize=30)
        plt.ylabel("RBO", fontsize=30)
        plt.ylim(0, 1)
        plt.xticks(rotation=45, fontsize=28)
        plt.yticks(fontsize=28)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save RBO violin plot
        rbo_path = self.output_dir / "rbo_violin.png"
        plt.savefig(rbo_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved RBO violin plot to {rbo_path}")
        
        # 3. Overlap Coefficient Violin Plot
        plt.figure(figsize=(28, 14))
        sns.violinplot(data=df, x='Model_Pair', y='Overlap_Coefficient', 
                      palette="viridis", inner="box")
        plt.title("Overlap Coefficient Distribution Across Queries", fontsize=32)
        plt.xlabel("Model Pairs", fontsize=30)
        plt.ylabel("Overlap Coefficient", fontsize=30)
        plt.ylim(0, 1)
        plt.xticks(rotation=45, fontsize=28)
        plt.yticks(fontsize=28)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save Overlap Coefficient violin plot
        overlap_path = self.output_dir / "overlap_coefficient_violin.png"
        plt.savefig(overlap_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved Overlap Coefficient violin plot to {overlap_path}")
        
        # Create box plots for clearer statistical comparison
        self._create_correlation_boxplots(df)
    
    def _create_correlation_boxplots(self, df):
        """
        Create separate box plots showing statistical distribution of correlation metrics
        """
        logger.info("Creating correlation box plots...")
        
        # 1. Kendall's Tau Box Plot
        plt.figure(figsize=(28, 14))
        box1 = plt.boxplot([df[df['Model_Pair'] == pair]['Kendall_Tau'] for pair in df['Model_Pair'].unique()],
                           labels=df['Model_Pair'].unique(),
                           patch_artist=True,
                           boxprops=dict(linewidth=2),
                           whiskerprops=dict(linewidth=2),
                           capprops=dict(linewidth=2),
                           medianprops=dict(linewidth=3, color='red'))
        
        # Color the boxes
        colors = ['steelblue', 'forestgreen', 'darkorange']
        for patch, color in zip(box1['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        plt.title("Kendall's Tau Statistical Distribution", fontsize=32)
        plt.ylabel("Kendall's τ", fontsize=30)
        plt.ylim(-1, 1)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45, fontsize=28)
        plt.yticks(fontsize=28)
        plt.tight_layout()
        
        # Save Kendall's Tau box plot
        kendall_box_path = self.output_dir / "kendall_tau_boxplot.png"
        plt.savefig(kendall_box_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved Kendall's Tau box plot to {kendall_box_path}")
        
        # 2. RBO Box Plot
        plt.figure(figsize=(28, 14))
        box2 = plt.boxplot([df[df['Model_Pair'] == pair]['RBO'] for pair in df['Model_Pair'].unique()],
                           labels=df['Model_Pair'].unique(),
                           patch_artist=True,
                           boxprops=dict(linewidth=2),
                           whiskerprops=dict(linewidth=2),
                           capprops=dict(linewidth=2),
                           medianprops=dict(linewidth=3, color='red'))
        
        for patch, color in zip(box2['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        plt.title("Rank-Biased Overlap Statistical Distribution", fontsize=32)
        plt.ylabel("RBO", fontsize=30)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45, fontsize=28)
        plt.yticks(fontsize=28)
        plt.tight_layout()
        
        # Save RBO box plot
        rbo_box_path = self.output_dir / "rbo_boxplot.png"
        plt.savefig(rbo_box_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved RBO box plot to {rbo_box_path}")
        
        # 3. Overlap Coefficient Box Plot
        plt.figure(figsize=(28, 14))
        box3 = plt.boxplot([df[df['Model_Pair'] == pair]['Overlap_Coefficient'] for pair in df['Model_Pair'].unique()],
                           labels=df['Model_Pair'].unique(),
                           patch_artist=True,
                           boxprops=dict(linewidth=2),
                           whiskerprops=dict(linewidth=2),
                           capprops=dict(linewidth=2),
                           medianprops=dict(linewidth=3, color='red'))
        
        for patch, color in zip(box3['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        plt.title("Overlap Coefficient Statistical Distribution", fontsize=32)
        plt.ylabel("Overlap Coefficient", fontsize=30)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45, fontsize=28)
        plt.yticks(fontsize=28)
        plt.tight_layout()
        
        # Save Overlap Coefficient box plot
        overlap_box_path = self.output_dir / "overlap_coefficient_boxplot.png"
        plt.savefig(overlap_box_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved Overlap Coefficient box plot to {overlap_box_path}")

    def _create_correlation_matrix_heatmap(self):
        """
        Create a correlation matrix heatmap
        """
        # Create correlation matrices
        n_models = len(self.models_to_test)
        kendall_matrix = np.eye(n_models)
        rbo_matrix = np.eye(n_models)
        overlap_matrix = np.eye(n_models)
        
        # Fill matrices
        for corr in self.correlations:
            model1, model2 = corr.model_pair
            i = self.models_to_test.index(model1)
            j = self.models_to_test.index(model2)
            
            kendall_matrix[i, j] = corr.kendall_tau
            kendall_matrix[j, i] = corr.kendall_tau
            
            rbo_matrix[i, j] = corr.rank_biased_overlap
            rbo_matrix[j, i] = corr.rank_biased_overlap
            
            overlap_matrix[i, j] = corr.overlap_coefficient
            overlap_matrix[j, i] = corr.overlap_coefficient
        
        # Create heatmaps
        fig, axes = plt.subplots(1, 3, figsize=(45, 12))
        
        # Kendall's Tau heatmap
        sns.heatmap(kendall_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                   xticklabels=self.models_to_test, yticklabels=self.models_to_test,
                   ax=axes[0], square=True, cbar_kws={'shrink': 0.8})
        axes[0].set_title("Kendall's Tau Matrix", fontsize=32)
        
        # RBO heatmap
        sns.heatmap(rbo_matrix, annot=True, fmt='.3f', cmap='Greens',
                   xticklabels=self.models_to_test, yticklabels=self.models_to_test,
                   ax=axes[1], square=True, cbar_kws={'shrink': 0.8})
        axes[1].set_title("Rank-Biased Overlap Matrix", fontsize=32)
        
        # Overlap Coefficient heatmap
        sns.heatmap(overlap_matrix, annot=True, fmt='.3f', cmap='Oranges',
                   xticklabels=self.models_to_test, yticklabels=self.models_to_test,
                   ax=axes[2], square=True, cbar_kws={'shrink': 0.8})
        axes[2].set_title("Overlap Coefficient Matrix", fontsize=32)
        
        plt.tight_layout()
        
        # Save heatmap
        heatmap_path = self.output_dir / "correlation_matrix_heatmaps.png"
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved correlation heatmaps to {heatmap_path}")
    
    def save_detailed_results(self):
        """
        Save detailed results to JSON files
        """
        logger.info("Saving detailed results...")
        
        # Save correlation results
        correlations_data = []
        for corr in self.correlations:
            correlations_data.append({
                "model_pair": list(corr.model_pair),
                "kendall_tau": corr.kendall_tau,
                "kendall_p_value": corr.kendall_p_value,
                "spearman_rho": corr.spearman_rho,
                "spearman_p_value": corr.spearman_p_value,
                "rank_biased_overlap": corr.rank_biased_overlap,
                "overlap_coefficient": corr.overlap_coefficient,
                "overlap_origin": corr.overlap_origin
            })
        
        correlations_path = self.output_dir / "ranking_correlations.json"
        with open(correlations_path, 'w') as f:
            json.dump(correlations_data, f, indent=2)
        
        # Save summary statistics
        summary = {
            "analysis_config": {
                "models_tested": self.models_to_test,
                "top_k": self.top_k,
                "num_documents": len(self.documents),
                "num_queries": len(self.queries),
                "precision": "fp32"
            },
            "correlation_summary": {
                "avg_kendall_tau": np.mean([corr.kendall_tau for corr in self.correlations]),
                "avg_rank_biased_overlap": np.mean([corr.rank_biased_overlap for corr in self.correlations]),
                "avg_overlap_coefficient": np.mean([corr.overlap_coefficient for corr in self.correlations]),
                "std_kendall_tau": np.std([corr.kendall_tau for corr in self.correlations]),
                "std_rank_biased_overlap": np.std([corr.rank_biased_overlap for corr in self.correlations]),
                "std_overlap_coefficient": np.std([corr.overlap_coefficient for corr in self.correlations])
            }
        }
        
        summary_path = self.output_dir / "analysis_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved detailed results to {correlations_path} and {summary_path}")
    
    def run_complete_analysis(self, max_documents: int = 5000, max_queries: int = 100):
        """
        Run the complete cross-model retrieval ranking correlation analysis
        """
        logger.info("Starting complete cross-model retrieval analysis...")
        start_time = time.time()
        
        try:
            # Step 1: Load MSMARCO data
            self.load_msmarco_data(max_documents, max_queries)
            
            # Step 2: Generate embeddings for all models
            self.generate_embeddings_for_all_models()
            
            # Step 3: Build FAISS indices
            self.build_faiss_indices()
            
            # Step 4: Perform retrieval
            self.perform_retrieval_for_all_models()
            
            # Step 5: Calculate correlations
            self.calculate_ranking_correlations()
            
            # Step 6: Create visualizations
            self.create_correlation_visualizations()
            
            # Step 7: Save results
            self.save_detailed_results()
            
            end_time = time.time()
            total_time = end_time - start_time
            
            logger.info(f"Analysis completed successfully in {total_time:.2f} seconds")
            
            # Print summary
            self._print_analysis_summary()
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise
    
    def _print_analysis_summary(self):
        """
        Print a summary of the analysis results
        """
        print("\n" + "="*80)
        print("CROSS-MODEL RETRIEVAL RANKING CORRELATION ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"Models tested: {', '.join(self.models_to_test)}")
        print(f"Documents: {len(self.documents)}")
        print(f"Queries: {len(self.queries)}")
        print(f"Top-K retrieval: {self.top_k}")
        print(f"Precision: FP32")
        
        print("\nRanking Correlation Results:")
        print("-" * 40)
        
        for corr in self.correlations:
            model1, model2 = corr.model_pair
            print(f"{model1.upper()} vs {model2.upper()}:")
            print(f"  Kendall's τ:       {corr.kendall_tau:.4f}")
            print(f"  RBO:               {corr.rank_biased_overlap:.4f}")
            print(f"  Overlap Coeff:     {corr.overlap_coefficient:.4f}")
            print(f"  Top-10 Overlap:    {corr.overlap_origin['top_10']:.4f}")
            print(f"  Top-20 Overlap:    {corr.overlap_origin['top_20']:.4f}")
            print(f"  Top-50 Overlap:    {corr.overlap_origin['top_50']:.4f}")
            print()
        
        # Overall statistics
        avg_kendall = np.mean([corr.kendall_tau for corr in self.correlations])
        avg_rbo = np.mean([corr.rank_biased_overlap for corr in self.correlations])
        avg_overlap = np.mean([corr.overlap_coefficient for corr in self.correlations])
        
        print("Overall Average Correlations:")
        print(f"  Kendall's τ:       {avg_kendall:.4f}")
        print(f"  RBO:               {avg_rbo:.4f}")
        print(f"  Overlap Coeff:     {avg_overlap:.4f}")
        
        print(f"\nResults saved to: {self.output_dir}")
        print("="*80)


def main():
    """
    Main function to run the cross-model retrieval analysis
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Cross-Model Retrieval Ranking Correlation Analysis"
    )
    parser.add_argument("--models", nargs="+", default=["bge", "e5", "qw"],
                       help="Models to test (default: bge e5 qw)")
    parser.add_argument("--top-k", type=int, default=50,
                       help="Number of top documents to retrieve (default: 50)")
    parser.add_argument("--max-docs", type=int, default=5000,
                       help="Maximum documents to load (default: 5000)")
    parser.add_argument("--max-queries", type=int, default=100,
                       help="Maximum queries to generate (default: 100)")
    parser.add_argument("--data-dir", default="data",
                       help="Directory containing MSMARCO data (default: data)")
    parser.add_argument("--output-dir", default="results",
                       help="Output directory for results (default: results)")
    parser.add_argument("--model-base-path", default="/scratch/user/u.bw269205/shared_models",
                       help="Base path for local models (default: /scratch/user/u.bw269205/shared_models)")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = CrossModelRetrievalAnalyzer(
        models_to_test=args.models,
        top_k=args.top_k,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_base_path=args.model_base_path
    )
    
    # Run analysis
    analyzer.run_complete_analysis(
        max_documents=args.max_docs,
        max_queries=args.max_queries
    )


if __name__ == "__main__":
    main()
