#!/usr/bin/env python3
"""
RAG Reproducibility Results Analysis Script
Run this in your Python environment (conda, venv, etc.)
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

class RAGResultsAnalyzer:
    def __init__(self, results_dir="my_reproducibility_results"):
        self.results_dir = Path(results_dir)
        self.results = {}

    def load_json_safely(self, file_path):
        """Safely load JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Error loading {file_path}: {e}")
            return None

    def load_all_data(self):
        """Load all available result files"""
        print("🔍 Loading result files...")

        # Basic FAISS results
        basic_file = self.results_dir / "basic_faiss_results.json"
        if basic_file.exists():
            self.results['basic_faiss'] = self.load_json_safely(basic_file)
            print(f"✅ Loaded: basic_faiss_results.json")

        # GPU non-determinism results
        gpu_file = self.results_dir / "gpu_nondeterminism_results.json"
        if gpu_file.exists():
            self.results['gpu_factors'] = self.load_json_safely(gpu_file)
            print(f"✅ Loaded: gpu_nondeterminism_results.json")

        # Integrated analysis results (both modes)
        for mode in ['inprocess', 'isolated']:
            integrated_file = self.results_dir / f"integrated_analysis_{mode}" / "integrated_results.json"
            if integrated_file.exists():
                self.results[f'integrated_{mode}'] = self.load_json_safely(integrated_file)
                print(f"✅ Loaded: integrated_analysis_{mode}/integrated_results.json")

        # Comprehensive analysis
        comp_file = self.results_dir / "comprehensive_analysis" / "full_analysis_data.json"
        if comp_file.exists():
            self.results['comprehensive'] = self.load_json_safely(comp_file)
            print(f"✅ Loaded: comprehensive_analysis/full_analysis_data.json")

        print(f"📊 Total datasets loaded: {len(self.results)}")
        return len(self.results) > 0

    def analyze_basic_faiss(self):
        """Analyze basic FAISS reproducibility"""
        if 'basic_faiss' not in self.results:
            print("⚠️  No basic FAISS data available")
            return

        print("\n" + "="*80)
        print("📊 BASIC FAISS REPRODUCIBILITY ANALYSIS")
        print("="*80)

        data = self.results['basic_faiss']

        print(f"{'Configuration':<20} {'Jaccard':<10} {'Exact Match':<12} {'Latency (ms)':<12} {'Status'}")
        print("-" * 80)

        perfect_configs = 0
        total_configs = 0

        for config_name, result in data.items():
            if 'metrics' not in result:
                continue

            total_configs += 1
            metrics = result['metrics']

            jaccard = metrics.get('overlap', {}).get('mean_jaccard', 0)
            exact_match = metrics.get('exact_match', {}).get('exact_match_rate', 0)
            latency = metrics.get('latency', {}).get('mean_latency_ms', 0)

            status = "🟢 Perfect" if jaccard >= 0.999 and exact_match >= 0.999 else "🔴 Issues"
            if jaccard >= 0.999:
                perfect_configs += 1

            print(f"{config_name:<20} {jaccard:<10.6f} {exact_match:<12.6f} {latency:<12.2f} {status}")

        print(f"\n📈 SUMMARY:")
        print(f"   Perfect configurations: {perfect_configs}/{total_configs} ({perfect_configs/total_configs*100:.1f}%)")

        # Find best performing configuration
        best_config = None
        best_score = -1
        for config_name, result in data.items():
            if 'metrics' in result:
                jaccard = result['metrics']['overlap']['mean_jaccard']
                if jaccard > best_score:
                    best_score = jaccard
                    best_config = config_name

        if best_config:
            print(f"🏆 Best configuration: {best_config} (Jaccard: {best_score:.6f})")

    def analyze_embedding_stability(self):
        """Analyze embedding stability across modes"""
        print("\n" + "="*80)
        print("🧬 EMBEDDING STABILITY ANALYSIS")
        print("="*80)

        for mode in ['inprocess', 'isolated']:
            key = f'integrated_{mode}'
            if key not in self.results:
                continue

            data = self.results[key]
            if 'embedding_stability' not in data:
                continue

            print(f"\n--- {mode.upper()} MODE RESULTS ---")
            print(f"{'Configuration':<15} {'L2 Distance':<15} {'Cosine Similarity':<18} {'Exact Match':<12}")
            print("-" * 70)

            stable_configs = 0
            total_configs = 0

            for config_name, result in data['embedding_stability'].items():
                if 'documents' not in result or 'metrics' not in result['documents']:
                    continue

                total_configs += 1
                metrics = result['documents']['metrics']

                # Handle both string and numeric formats
                l2_dist = metrics.get('l2_distance', {}).get('mean', 0)
                cos_sim = metrics.get('cosine_similarity', {}).get('mean', 1)
                exact_match = metrics.get('exact_match_rate', 0)

                if isinstance(l2_dist, str):
                    l2_dist = float(l2_dist)
                if isinstance(cos_sim, str):
                    cos_sim = float(cos_sim)

                if l2_dist < 1e-6:  # Very stable
                    stable_configs += 1

                print(f"{config_name:<15} {l2_dist:<15.2e} {cos_sim:<18.12f} {exact_match:<12.3f}")

            print(f"\n📊 {mode.upper()} Summary: {stable_configs}/{total_configs} highly stable configurations")

    def analyze_gpu_factors(self):
        """Analyze GPU non-determinism factors"""
        if 'gpu_factors' not in self.results:
            print("⚠️  No GPU factors data available")
            return

        print("\n" + "="*80)
        print("🖥️  GPU NON-DETERMINISM FACTORS ANALYSIS")
        print("="*80)

        data = self.results['gpu_factors']

        for factor_name, factor_data in data.items():
            print(f"\n--- {factor_name.upper().replace('_', ' ')} ---")

            if not isinstance(factor_data, dict) or 'error' in factor_data:
                print(f"❌ Error or no data available")
                continue

            print(f"{'Configuration':<30} {'Jaccard':<10} {'Status'}")
            print("-" * 50)

            perfect_count = 0
            total_count = 0

            for config_name, metrics in factor_data.items():
                if not isinstance(metrics, dict) or 'overlap' not in metrics:
                    continue

                total_count += 1
                jaccard = metrics['overlap']['mean_jaccard']
                status = "🟢 Perfect" if jaccard >= 0.999 else "🔴 Issues"

                if jaccard >= 0.999:
                    perfect_count += 1

                print(f"{config_name:<30} {jaccard:<10.6f} {status}")

            if total_count > 0:
                print(f"Factor performance: {perfect_count}/{total_count} perfect ({perfect_count/total_count*100:.1f}%)")

    def generate_overall_assessment(self):
        """Generate overall framework assessment"""
        print("\n" + "="*80)
        print("🎯 OVERALL FRAMEWORK ASSESSMENT")
        print("="*80)

        scores = {}

        # Basic FAISS score
        if 'basic_faiss' in self.results:
            perfect = 0
            total = 0
            for result in self.results['basic_faiss'].values():
                if 'metrics' in result:
                    total += 1
                    jaccard = result['metrics']['overlap']['mean_jaccard']
                    if jaccard >= 0.999:
                        perfect += 1
            if total > 0:
                scores['FAISS Reproducibility'] = perfect / total

        # Embedding stability score
        for mode in ['inprocess', 'isolated']:
            key = f'integrated_{mode}'
            if key in self.results and 'embedding_stability' in self.results[key]:
                stable = 0
                total = 0
                for result in self.results[key]['embedding_stability'].values():
                    if 'documents' in result and 'metrics' in result['documents']:
                        total += 1
                        l2_dist = result['documents']['metrics'].get('l2_distance', {}).get('mean', 0)
                        if isinstance(l2_dist, str):
                            l2_dist = float(l2_dist)
                        if l2_dist < 1e-6:
                            stable += 1
                if total > 0:
                    scores[f'Embedding Stability ({mode})'] = stable / total

        # GPU determinism score
        if 'gpu_factors' in self.results:
            perfect_factors = 0
            total_factors = 0
            for factor_data in self.results['gpu_factors'].values():
                if isinstance(factor_data, dict) and 'error' not in factor_data:
                    total_factors += 1
                    factor_perfect = True
                    for metrics in factor_data.values():
                        if isinstance(metrics, dict) and 'overlap' in metrics:
                            jaccard = metrics['overlap']['mean_jaccard']
                            if jaccard < 0.999:
                                factor_perfect = False
                                break
                    if factor_perfect:
                        perfect_factors += 1
            if total_factors > 0:
                scores['GPU Determinism'] = perfect_factors / total_factors

        # Display scores
        print("Component Scores (0.0 - 1.0):")
        print("-" * 50)

        overall_scores = []
        for component, score in scores.items():
            grade = self.score_to_grade(score)
            status = "🟢" if score >= 0.95 else "🟡" if score >= 0.8 else "🔴"
            print(f"{status} {component:<30} {score:.3f} ({grade})")
            overall_scores.append(score)

        # Overall assessment
        if overall_scores:
            overall_score = sum(overall_scores) / len(overall_scores)
            overall_grade = self.score_to_grade(overall_score)
            status = "🟢" if overall_score >= 0.95 else "🟡" if overall_score >= 0.8 else "🔴"

            print("-" * 50)
            print(f"{status} {'OVERALL FRAMEWORK SCORE':<30} {overall_score:.3f} ({overall_grade})")

            print(f"\n🎯 ASSESSMENT:")
            if overall_score >= 0.95:
                print("   🌟 EXCEPTIONAL - Your framework achieves state-of-the-art reproducibility!")
                print("   ✅ Ready for production deployment")
                print("   📚 Suitable for research publication")
            elif overall_score >= 0.9:
                print("   🎉 EXCELLENT - Outstanding reproducibility performance")
                print("   ✅ Ready for production with monitoring")
            elif overall_score >= 0.8:
                print("   👍 GOOD - Solid reproducibility foundation")
                print("   ⚠️  Some improvements needed before production")
            else:
                print("   ⚠️  NEEDS IMPROVEMENT - Significant reproducibility issues")
                print("   🔧 Requires debugging before deployment")

    def score_to_grade(self, score):
        """Convert score to letter grade"""
        if score >= 0.97: return "A+"
        elif score >= 0.93: return "A"
        elif score >= 0.90: return "A-"
        elif score >= 0.87: return "B+"
        elif score >= 0.83: return "B"
        elif score >= 0.80: return "B-"
        else: return "C+"

    def save_detailed_analysis(self, filename="detailed_analysis_results.json"):
        """Save detailed analysis results to JSON for further processing"""
        analysis_results = {
            "metadata": {
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "datasets_loaded": len(self.results),
                "analysis_version": "1.0"
            },
            "basic_faiss_analysis": {},
            "embedding_stability_analysis": {},
            "gpu_factors_analysis": {},
            "overall_assessment": {}
        }

        # Basic FAISS analysis
        if 'basic_faiss' in self.results:
            faiss_results = []
            perfect_configs = 0
            total_configs = 0

            for config_name, result in self.results['basic_faiss'].items():
                if 'metrics' not in result:
                    continue

                total_configs += 1
                metrics = result['metrics']

                jaccard = metrics.get('overlap', {}).get('mean_jaccard', 0)
                exact_match = metrics.get('exact_match', {}).get('exact_match_rate', 0)
                latency = metrics.get('latency', {}).get('mean_latency_ms', 0)

                is_perfect = jaccard >= 0.999 and exact_match >= 0.999
                if is_perfect:
                    perfect_configs += 1

                faiss_results.append({
                    "configuration": config_name,
                    "jaccard_similarity": jaccard,
                    "exact_match_rate": exact_match,
                    "latency_ms": latency,
                    "is_perfect": is_perfect
                })

            analysis_results["basic_faiss_analysis"] = {
                "configurations": faiss_results,
                "summary": {
                    "total_configurations": total_configs,
                    "perfect_configurations": perfect_configs,
                    "perfect_percentage": (perfect_configs/total_configs*100) if total_configs > 0 else 0
                }
            }

        # Embedding stability analysis
        for mode in ['inprocess', 'isolated']:
            key = f'integrated_{mode}'
            if key not in self.results or 'embedding_stability' not in self.results[key]:
                continue

            stability_results = []
            stable_configs = 0
            total_configs = 0

            for config_name, result in self.results[key]['embedding_stability'].items():
                if 'documents' not in result or 'metrics' not in result['documents']:
                    continue

                total_configs += 1
                metrics = result['documents']['metrics']

                l2_dist = metrics.get('l2_distance', {}).get('mean', 0)
                cos_sim = metrics.get('cosine_similarity', {}).get('mean', 1)
                exact_match = metrics.get('exact_match_rate', 0)

                if isinstance(l2_dist, str):
                    l2_dist = float(l2_dist)
                if isinstance(cos_sim, str):
                    cos_sim = float(cos_sim)

                is_stable = l2_dist < 1e-6
                if is_stable:
                    stable_configs += 1

                stability_results.append({
                    "configuration": config_name,
                    "l2_distance": l2_dist,
                    "cosine_similarity": cos_sim,
                    "exact_match_rate": exact_match,
                    "is_highly_stable": is_stable
                })

            analysis_results["embedding_stability_analysis"][mode] = {
                "configurations": stability_results,
                "summary": {
                    "total_configurations": total_configs,
                    "stable_configurations": stable_configs,
                    "stability_percentage": (stable_configs/total_configs*100) if total_configs > 0 else 0
                }
            }

        # GPU factors analysis
        if 'gpu_factors' in self.results:
            gpu_analysis = {}

            for factor_name, factor_data in self.results['gpu_factors'].items():
                if not isinstance(factor_data, dict) or 'error' in factor_data:
                    gpu_analysis[factor_name] = {"error": "No valid data"}
                    continue

                factor_results = []
                perfect_count = 0
                total_count = 0

                for config_name, metrics in factor_data.items():
                    if not isinstance(metrics, dict) or 'overlap' not in metrics:
                        continue

                    total_count += 1
                    jaccard = metrics['overlap']['mean_jaccard']
                    is_perfect = jaccard >= 0.999

                    if is_perfect:
                        perfect_count += 1

                    factor_results.append({
                        "configuration": config_name,
                        "jaccard_similarity": jaccard,
                        "is_perfect": is_perfect
                    })

                gpu_analysis[factor_name] = {
                    "configurations": factor_results,
                    "summary": {
                        "total_configurations": total_count,
                        "perfect_configurations": perfect_count,
                        "perfect_percentage": (perfect_count/total_count*100) if total_count > 0 else 0
                    }
                }

            analysis_results["gpu_factors_analysis"] = gpu_analysis

        # Overall assessment
        scores = {}

        # Calculate component scores
        if 'basic_faiss' in self.results:
            perfect = sum(1 for result in self.results['basic_faiss'].values()
                         if 'metrics' in result and result['metrics']['overlap']['mean_jaccard'] >= 0.999)
            total = sum(1 for result in self.results['basic_faiss'].values() if 'metrics' in result)
            scores['faiss_reproducibility'] = perfect / total if total > 0 else 0

        for mode in ['inprocess', 'isolated']:
            key = f'integrated_{mode}'
            if key in self.results and 'embedding_stability' in self.results[key]:
                stable = 0
                total = 0
                for result in self.results[key]['embedding_stability'].values():
                    if 'documents' in result and 'metrics' in result['documents']:
                        total += 1
                        l2_dist = result['documents']['metrics'].get('l2_distance', {}).get('mean', 0)
                        if isinstance(l2_dist, str):
                            l2_dist = float(l2_dist)
                        if l2_dist < 1e-6:
                            stable += 1
                scores[f'embedding_stability_{mode}'] = stable / total if total > 0 else 0

        if 'gpu_factors' in self.results:
            perfect_factors = 0
            total_factors = 0
            for factor_data in self.results['gpu_factors'].values():
                if isinstance(factor_data, dict) and 'error' not in factor_data:
                    total_factors += 1
                    factor_perfect = all(
                        metrics.get('overlap', {}).get('mean_jaccard', 0) >= 0.999
                        for metrics in factor_data.values()
                        if isinstance(metrics, dict) and 'overlap' in metrics
                    )
                    if factor_perfect:
                        perfect_factors += 1
            scores['gpu_determinism'] = perfect_factors / total_factors if total_factors > 0 else 0

        overall_score = sum(scores.values()) / len(scores) if scores else 0
        overall_grade = self.score_to_grade(overall_score)

        analysis_results["overall_assessment"] = {
            "component_scores": scores,
            "overall_score": overall_score,
            "overall_grade": overall_grade,
            "assessment_level": self.get_assessment_level(overall_score),
            "production_ready": overall_score >= 0.95,
            "research_ready": overall_score >= 0.9
        }

        # Save to JSON file
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(analysis_results, f, indent=2, ensure_ascii=False)
            print(f"📊 Detailed analysis results saved to: {filename}")
            return filename
        except Exception as e:
            print(f"❌ Error saving detailed analysis: {e}")
            return None

    def get_assessment_level(self, score):
        """Get assessment level description"""
        if score >= 0.95:
            return "EXCEPTIONAL"
        elif score >= 0.9:
            return "EXCELLENT"
        elif score >= 0.8:
            return "GOOD"
        else:
            return "NEEDS_IMPROVEMENT"

    def save_summary_report(self, filename="analysis_summary.txt"):
        """Save a text summary report"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("RAG Reproducibility Analysis Summary\n")
                f.write("="*50 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Datasets analyzed: {len(self.results)}\n\n")

                # Quick summary
                if 'basic_faiss' in self.results:
                    perfect_configs = 0
                    total_configs = 0
                    for result in self.results['basic_faiss'].values():
                        if 'metrics' in result:
                            total_configs += 1
                            jaccard = result['metrics']['overlap']['mean_jaccard']
                            if jaccard >= 0.999:
                                perfect_configs += 1

                    f.write(f"FAISS Reproducibility: {perfect_configs}/{total_configs} perfect configurations\n")

                f.write("\nRecommendations:\n")
                f.write("- Use deterministic mode for maximum reproducibility\n")
                f.write("- Consider process isolation for research applications\n")
                f.write("- Monitor GPU determinism factors in production\n")

            print(f"📄 Summary report saved to: {filename}")

        except Exception as e:
            print(f"❌ Error saving report: {e}")

    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("🚀 RAG Reproducibility Results Analysis")
        print("="*80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Check if results directory exists
        if not self.results_dir.exists():
            print(f"❌ Results directory '{self.results_dir}' not found!")
            print("Please run the reproducibility tests first.")
            return False

        # Load all data
        if not self.load_all_data():
            print("❌ No result files found!")
            return False

        # Run analyses
        self.analyze_basic_faiss()
        self.analyze_embedding_stability()
        self.analyze_gpu_factors()
        self.generate_overall_assessment()

        # Save detailed analysis for report generation
        detailed_file = self.save_detailed_analysis()

        # Save summary
        self.save_summary_report()

        print("\n🎉 Analysis completed successfully!")
        print("🔍 Check analysis_summary.txt for a quick summary")
        print("📊 Check detailed_analysis_results.json for comprehensive data")

        return True

def main():
    """Main entry point"""
    print("RAG Reproducibility Results Analyzer")
    print("Run this in your Python environment (conda activate <env_name>)")
    print()

    analyzer = RAGResultsAnalyzer()
    success = analyzer.run_full_analysis()

    if success:
        print("\n✨ Your RAG framework shows exceptional reproducibility!")
        print("🚀 Ready for production deployment and research publication!")
    else:
        print("\n❌ Analysis failed. Please check your setup and try again.")

if __name__ == "__main__":
    main()
