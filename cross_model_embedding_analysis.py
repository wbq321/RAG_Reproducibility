#!/usr/bin/env python3
"""
Cross-Model Embedding Certainty Analysis
Analyze embedding consistency and uncertainty across different models
"""

import json
import numpy as np
from pathlib import Path
import os
from typing import Dict, List, Tuple, Any
from collections import defaultdict

# Plotting imports (optional)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    PLOTTING_AVAILABLE = True
    
    # Set publication-quality plot parameters
    plt.rcParams.update({
        'font.size': 30,
        'font.weight': 'bold',
        'axes.titlesize': 30,
        'axes.labelsize': 30,
        'xtick.labelsize': 30,
        'ytick.labelsize': 30,
        'legend.fontsize': 30,
        'figure.titlesize': 30,
        'lines.linewidth': 2,
        'axes.linewidth': 2,
        'xtick.major.width': 2,
        'ytick.major.width': 2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })
except ImportError:
    PLOTTING_AVAILABLE = False
    print("⚠️ Plotting libraries not available. Analysis will proceed without visualization.")

class CrossModelEmbeddingAnalyzer:
    """Analyze embedding certainty and consistency across multiple models"""
    
    def __init__(self, results_base_dir: str = "results"):
        self.results_base = Path(results_base_dir)
        self.models = ["bge", "e5", "qw"]
        self.output_dir = self.results_base / "cross_model_analysis"
        self.output_dir.mkdir(exist_ok=True)
        
        self.cross_model_data = {}
        self.model_embeddings = {}
        self.analysis_results = {}
        
    def load_model_data(self) -> Dict[str, Any]:
        """Load embedding data from all available models"""
        print("🔍 Loading cross-model embedding data...")
        
        loaded_models = []
        
        for model in self.models:
            model_dir = self.results_base / model
            results_file = model_dir / "embedding_reproducibility_results.json"
            
            if not results_file.exists():
                print(f"⚠️ Results file not found for {model.upper()}: {results_file}")
                continue
                
            try:
                with open(results_file, 'r') as f:
                    model_data = json.load(f)
                    
                self.cross_model_data[model] = model_data
                loaded_models.append(model)
                print(f"✅ Loaded {model.upper()} data with {len(model_data)} configurations")
                
            except Exception as e:
                print(f"❌ Error loading {model.upper()} data: {e}")
                continue
        
        if not loaded_models:
            print("❌ No model data could be loaded!")
            return {}
            
        print(f"📊 Successfully loaded data for {len(loaded_models)} models: {', '.join([m.upper() for m in loaded_models])}")
        return self.cross_model_data
    
    def extract_embeddings_by_config(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Extract embeddings organized by configuration across models"""
        print("🔗 Extracting embeddings by configuration...")
        
        # Structure: {config_name: {model_name: embedding_array}}
        config_embeddings = defaultdict(dict)
        
        for model_name, model_data in self.cross_model_data.items():
            for config_name, config_data in model_data.items():
                if 'embeddings' in config_data and config_data['embeddings']:
                    # Parse the first embedding (assuming single text input)
                    embedding_str = config_data['embeddings'][0]
                    
                    try:
                        # Parse the first embedding (assuming single text input)
                        embedding_str = config_data['embeddings'][0]
                        
                        if isinstance(embedding_str, str):
                            # Remove any extra whitespace and handle different formats
                            embedding_str = embedding_str.strip()
                            
                            # Method 1: Direct numpy array parsing for the specific format we have
                            if embedding_str.startswith('[[') and embedding_str.endswith(']]'):
                                # This is a 2D array string representation - use ast.literal_eval
                                try:
                                    import ast
                                    # Remove the string quotes and parse as literal
                                    array_data = ast.literal_eval(embedding_str)
                                    embedding = np.array(array_data).flatten()
                                except:
                                    # Fallback: manually parse the scientific notation
                                    import re
                                    # Extract all numbers including scientific notation
                                    pattern = r'-?\d+\.?\d*[eE]?[+-]?\d*'
                                    numbers = re.findall(pattern, embedding_str)
                                    # Filter out empty strings and invalid numbers
                                    valid_numbers = []
                                    for num_str in numbers:
                                        if num_str and num_str != 'e' and num_str != 'E':
                                            try:
                                                val = float(num_str)
                                                valid_numbers.append(val)
                                            except ValueError:
                                                continue
                                    embedding = np.array(valid_numbers)
                            else:
                                # Try other parsing methods
                                try:
                                    import json
                                    embedding = np.array(json.loads(embedding_str))
                                except:
                                    # Last resort: try eval
                                    embedding = np.array(eval(embedding_str))
                        else:
                            # If it's already a list or array
                            embedding = np.array(embedding_str)
                        
                        # Ensure it's flattened
                        if embedding.ndim > 1:
                            embedding = embedding.flatten()
                        
                        # Validate embedding dimension
                        if len(embedding) < 100:  # Embeddings should be much larger
                            print(f"⚠️ Warning: {model_name}/{config_name} has unusually small embedding dimension: {len(embedding)}")
                            print(f"   First few values: {embedding[:10]}")
                            continue
                        
                        config_embeddings[config_name][model_name] = embedding
                        print(f"✅ Parsed {model_name}/{config_name}: {len(embedding)} dimensions")
                        
                    except Exception as e:
                        print(f"⚠️ Failed to parse embedding for {model_name}/{config_name}: {e}")
                        print(f"   Embedding string preview: {str(embedding_str)[:200]}...")
                        continue
                    
        # Filter to only include configs present in multiple models
        filtered_configs = {}
        for config, models in config_embeddings.items():
            if len(models) >= 2:  # At least 2 models must have this config
                # Check that all embeddings have the same dimension
                dimensions = [len(emb) for emb in models.values()]
                if len(set(dimensions)) == 1:  # All same dimension
                    filtered_configs[config] = models
                else:
                    print(f"⚠️ Skipping {config}: inconsistent dimensions {dimensions}")
        
        self.model_embeddings = filtered_configs
        print(f"📈 Found {len(filtered_configs)} configurations shared across multiple models")
        
        for config, models in filtered_configs.items():
            dimensions = [len(emb) for emb in models.values()]
            print(f"   {config}: {', '.join(models.keys())} (dim: {dimensions[0]})")
            
        return filtered_configs
    
    def calculate_cross_model_distances(self) -> Dict[str, Any]:
        """Calculate distances between embeddings across models for each configuration"""
        print("📏 Calculating cross-model embedding distances...")
        
        distance_results = {}
        
        for config_name, model_embeddings in self.model_embeddings.items():
            models = list(model_embeddings.keys())
            config_results = {
                'models': models,
                'pairwise_distances': {},
                'statistics': {}
            }
            
            # Calculate pairwise L2 distances
            distances = []
            for i, model1 in enumerate(models):
                for j, model2 in enumerate(models):
                    if i < j:  # Avoid duplicates
                        emb1 = model_embeddings[model1]
                        emb2 = model_embeddings[model2]
                        
                        l2_distance = np.linalg.norm(emb1 - emb2)
                        cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                        
                        pair_key = f"{model1}_vs_{model2}"
                        config_results['pairwise_distances'][pair_key] = {
                            'l2_distance': float(l2_distance),
                            'cosine_similarity': float(cosine_sim),
                            'cosine_distance': float(1 - cosine_sim)
                        }
                        distances.append(l2_distance)
            
            # Calculate statistics
            if distances:
                config_results['statistics'] = {
                    'mean_l2_distance': float(np.mean(distances)),
                    'std_l2_distance': float(np.std(distances)),
                    'max_l2_distance': float(np.max(distances)),
                    'min_l2_distance': float(np.min(distances)),
                    'num_model_pairs': len(distances)
                }
            
            distance_results[config_name] = config_results
            
        self.analysis_results = distance_results
        return distance_results
    
    def analyze_model_consistency(self) -> Dict[str, Any]:
        """Analyze which models are most consistent with each other"""
        print("🎯 Analyzing model consistency patterns...")
        
        model_pair_stats = defaultdict(list)
        
        # Collect all distances for each model pair across configurations
        for config_name, config_data in self.analysis_results.items():
            for pair_name, distances in config_data['pairwise_distances'].items():
                model_pair_stats[pair_name].append(distances['l2_distance'])
        
        # Calculate overall statistics for each model pair
        consistency_analysis = {}
        for pair_name, distances in model_pair_stats.items():
            if distances:
                consistency_analysis[pair_name] = {
                    'mean_distance': float(np.mean(distances)),
                    'std_distance': float(np.std(distances)),
                    'min_distance': float(np.min(distances)),
                    'max_distance': float(np.max(distances)),
                    'num_configurations': len(distances),
                    'consistency_score': float(1.0 / (1.0 + np.mean(distances)))  # Higher = more consistent
                }
        
        # Rank model pairs by consistency
        sorted_pairs = sorted(consistency_analysis.items(), 
                            key=lambda x: x[1]['consistency_score'], 
                            reverse=True)
        
        print("\n🏆 Model Pair Consistency Ranking (Higher score = more consistent):")
        for i, (pair_name, stats) in enumerate(sorted_pairs, 1):
            print(f"{i}. {pair_name}: {stats['consistency_score']:.4f} "
                  f"(avg L2: {stats['mean_distance']:.4f})")
        
        return {
            'pair_statistics': consistency_analysis,
            'consistency_ranking': sorted_pairs
        }
    
    def generate_cross_model_heatmap(self):
        """Generate heatmap showing cross-model distances"""
        if not PLOTTING_AVAILABLE:
            print("⚠️ Skipping heatmap - plotting libraries not available")
            return
            
        print("🎨 Generating cross-model distance heatmap...")
        
        # Create a comprehensive distance matrix
        all_models = set()
        for config_data in self.analysis_results.values():
            all_models.update(config_data['models'])
        
        all_models = sorted(list(all_models))
        n_models = len(all_models)
        
        if n_models < 2:
            print("⚠️ Need at least 2 models for cross-model analysis")
            return
        
        # Create separate heatmaps for each configuration
        for config_name, config_data in self.analysis_results.items():
            if len(config_data['models']) < 2:
                continue
                
            # Create distance matrix
            models_in_config = config_data['models']
            n_models_config = len(models_in_config)
            distance_matrix = np.zeros((n_models_config, n_models_config))
            
            # Fill the matrix
            for i, model1 in enumerate(models_in_config):
                for j, model2 in enumerate(models_in_config):
                    if i == j:
                        distance_matrix[i, j] = 0.0
                    else:
                        # Find the distance in pairwise_distances
                        pair_key1 = f"{model1}_vs_{model2}"
                        pair_key2 = f"{model2}_vs_{model1}"
                        
                        if pair_key1 in config_data['pairwise_distances']:
                            distance_matrix[i, j] = config_data['pairwise_distances'][pair_key1]['l2_distance']
                        elif pair_key2 in config_data['pairwise_distances']:
                            distance_matrix[i, j] = config_data['pairwise_distances'][pair_key2]['l2_distance']
            
            # Create heatmap
            plt.figure(figsize=(12, 10))
            
            # Create annotations
            annot_matrix = np.zeros_like(distance_matrix, dtype=object)
            for i in range(n_models_config):
                for j in range(n_models_config):
                    if i == j:
                        annot_matrix[i, j] = '0.000'
                    else:
                        annot_matrix[i, j] = f'{distance_matrix[i, j]:.3f}'
            
            sns.heatmap(distance_matrix, 
                       annot=annot_matrix, 
                       fmt='',
                       xticklabels=[m.upper() for m in models_in_config],
                       yticklabels=[m.upper() for m in models_in_config],
                       cmap='YlOrRd',
                       square=True,
                       linewidths=2,
                       cbar_kws={'label': 'L2 Distance'},
                       annot_kws={'fontsize': 28, 'fontweight': 'bold'})
            
            plt.title(f'Cross-Model Embedding Distance\n{config_name}', 
                     fontsize=30, fontweight='bold', pad=40)
            plt.xlabel('Model', fontsize=30, fontweight='bold')
            plt.ylabel('Model', fontsize=30, fontweight='bold')
            plt.xticks(fontsize=30, fontweight='bold', rotation=45)
            plt.yticks(fontsize=30, fontweight='bold', rotation=0)
            
            # Adjust colorbar
            cbar = plt.gca().collections[0].colorbar
            cbar.ax.tick_params(labelsize=26)
            cbar.set_label('L2 Distance', fontsize=30, fontweight='bold')
            
            plt.tight_layout()
            
            # Save the plot
            safe_config_name = config_name.replace(' ', '_').replace('/', '_')
            output_path = self.output_dir / f'cross_model_distance_{safe_config_name}.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"   💾 Saved: {output_path}")
    
    def generate_consistency_comparison(self):
        """Generate a comparison plot of model consistency across configurations"""
        if not PLOTTING_AVAILABLE:
            print("⚠️ Skipping consistency comparison - plotting libraries not available")
            return
            
        print("📊 Generating model consistency comparison...")
        
        # Prepare data for plotting
        config_names = []
        mean_distances = []
        std_distances = []
        
        for config_name, config_data in self.analysis_results.items():
            if 'statistics' in config_data and config_data['statistics']:
                config_names.append(config_name.replace(' ', '\n'))
                mean_distances.append(config_data['statistics']['mean_l2_distance'])
                std_distances.append(config_data['statistics']['std_l2_distance'])
        
        if not config_names:
            print("⚠️ No data available for consistency comparison")
            return
        
        # Create bar plot
        fig, ax = plt.subplots(1, 1, figsize=(20, 14))
        
        # Color bars by distance (lower = more consistent = green)
        colors = []
        for distance in mean_distances:
            if distance < 0.1:
                colors.append('#2ca02c')  # Green - very consistent
            elif distance < 0.5:
                colors.append('#ff7f0e')  # Orange - moderately consistent
            else:
                colors.append('#d62728')  # Red - less consistent
        
        bars = ax.bar(range(len(config_names)), mean_distances, 
                     yerr=std_distances, capsize=15, 
                     color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        
        ax.set_xlabel('Configuration', fontsize=30, fontweight='bold')
        ax.set_ylabel('Mean Cross-Model L2 Distance', fontsize=30, fontweight='bold')
        ax.set_title('Cross-Model Embedding Consistency\n(Lower = More Consistent)', 
                    fontsize=30, fontweight='bold', pad=40)
        
        ax.set_xticks(range(len(config_names)))
        ax.set_xticklabels(config_names, rotation=45, ha='right', fontsize=26, fontweight='bold')
        ax.tick_params(axis='y', labelsize=30)
        ax.grid(True, alpha=0.3, axis='y', linewidth=1.5)
        
        # Add value labels on bars
        for i, (bar, mean_dist, std_dist) in enumerate(zip(bars, mean_distances, std_distances)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std_dist + max(mean_distances) * 0.02,
                   f'{mean_dist:.3f}', ha='center', va='bottom',
                   fontsize=24, fontweight='bold')
        
        # Clean up plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        
        plt.tight_layout()
        
        # Save the plot
        output_path = self.output_dir / 'cross_model_consistency_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   💾 Saved: {output_path}")
    
    def save_analysis_results(self):
        """Save detailed analysis results to JSON"""
        print("💾 Saving cross-model analysis results...")
        
        # Prepare comprehensive results
        comprehensive_results = {
            'cross_model_distances': self.analysis_results,
            'model_consistency': self.analyze_model_consistency(),
            'summary': {
                'models_analyzed': list(self.cross_model_data.keys()),
                'configurations_analyzed': list(self.analysis_results.keys()),
                'total_model_pairs': sum(len(config['pairwise_distances']) 
                                       for config in self.analysis_results.values())
            }
        }
        
        # Save detailed results
        results_file = self.output_dir / 'cross_model_analysis_results.json'
        with open(results_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2)
        
        print(f"   💾 Detailed results: {results_file}")
        
        # Save summary report
        self.generate_summary_report(comprehensive_results)
    
    def generate_summary_report(self, results: Dict[str, Any]):
        """Generate a human-readable summary report"""
        report_lines = [
            "# Cross-Model Embedding Certainty Analysis Report",
            "=" * 60,
            "",
            f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Models Analyzed:** {', '.join([m.upper() for m in results['summary']['models_analyzed']])}",
            f"**Configurations:** {len(results['summary']['configurations_analyzed'])}",
            f"**Total Model Pairs:** {results['summary']['total_model_pairs']}",
            "",
            "## Model Consistency Ranking",
            "",
        ]
        
        # Add consistency ranking
        for i, (pair_name, stats) in enumerate(results['model_consistency']['consistency_ranking'], 1):
            report_lines.append(
                f"{i}. **{pair_name}**: Consistency Score = {stats['consistency_score']:.4f} "
                f"(Avg L2 Distance: {stats['mean_distance']:.4f})"
            )
        
        report_lines.extend([
            "",
            "## Configuration Analysis",
            "",
        ])
        
        # Add per-configuration analysis
        for config_name, config_data in results['cross_model_distances'].items():
            if 'statistics' in config_data:
                stats = config_data['statistics']
                report_lines.extend([
                    f"### {config_name}",
                    f"- **Models:** {', '.join([m.upper() for m in config_data['models']])}",
                    f"- **Mean L2 Distance:** {stats['mean_l2_distance']:.4f}",
                    f"- **Std L2 Distance:** {stats['std_l2_distance']:.4f}",
                    f"- **Range:** {stats['min_l2_distance']:.4f} - {stats['max_l2_distance']:.4f}",
                    f"- **Model Pairs:** {stats['num_model_pairs']}",
                    "",
                ])
        
        # Save report
        report_file = self.output_dir / 'cross_model_analysis_report.md'
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"   📄 Summary report: {report_file}")
    
    def run_cross_model_analysis(self) -> bool:
        """Run the complete cross-model embedding certainty analysis"""
        print("🚀 Starting Cross-Model Embedding Certainty Analysis")
        print("=" * 60)
        
        try:
            # Load data
            if not self.load_model_data():
                return False
            
            # Extract embeddings
            if not self.extract_embeddings_by_config():
                print("❌ No shared configurations found across models")
                return False
            
            # Calculate distances
            self.calculate_cross_model_distances()
            
            # Generate visualizations
            if PLOTTING_AVAILABLE:
                self.generate_cross_model_heatmap()
                self.generate_consistency_comparison()
            
            # Save results
            self.save_analysis_results()
            
            print("\n" + "=" * 60)
            print("✅ Cross-Model Analysis Complete!")
            print(f"📁 Results saved in: {self.output_dir}")
            print("📊 Key findings:")
            
            # Quick summary
            consistency_data = self.analyze_model_consistency()
            if consistency_data['consistency_ranking']:
                most_consistent = consistency_data['consistency_ranking'][0]
                least_consistent = consistency_data['consistency_ranking'][-1]
                
                print(f"   🏆 Most consistent pair: {most_consistent[0]} "
                      f"(score: {most_consistent[1]['consistency_score']:.4f})")
                print(f"   ⚠️ Least consistent pair: {least_consistent[0]} "
                      f"(score: {least_consistent[1]['consistency_score']:.4f})")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during cross-model analysis: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main function to run cross-model analysis"""
    analyzer = CrossModelEmbeddingAnalyzer()
    success = analyzer.run_cross_model_analysis()
    
    if not success:
        print("\n❌ Cross-model analysis failed. Please check the error messages above.")
        return False
    
    return True

if __name__ == "__main__":
    main()
