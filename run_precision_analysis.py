"""
Simple script to run precision comparison analysis
Integrates with your existing RAG reproducibility framework
"""

import sys
import os

# Add src directory to path
sys.path.append('src')

from embedding_reproducibility_tester import PrecisionComparisonAnalyzer

def run_precision_analysis():
    """Run precision analysis with your local model"""

    # Your model path
    model_path = "/scratch/user/u.bw269205/shared_models/bge_model"

    # Test texts - you can modify these or load from your test datasets
    test_texts = [
        "Document about machine learning and artificial intelligence research.",
        "Natural language processing enables semantic understanding of text.",
        "Vector embeddings capture meaning in high-dimensional space.",
        "Transformer models have revolutionized language understanding.",
        "Reproducibility testing ensures consistent model behavior.",
        "GPU acceleration improves neural network training speed.",
        "Different precision formats affect computational accuracy.",
        "RAG systems combine retrieval with generation capabilities.",
        "FAISS provides efficient similarity search for embeddings.",
        "Distributed computing scales machine learning workloads."
    ]

    print("🔬 Starting Embedding Precision Analysis")
    print(f"📍 Model: {model_path}")
    print(f"📝 Testing with {len(test_texts)} sample texts")

    # Create results directory
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    print(f"📁 Results will be saved to: {results_dir}/")

    # Create analyzer
    analyzer = PrecisionComparisonAnalyzer(model_path, device="cuda")

    # Run analysis - save to results directory
    results = analyzer.run_precision_comparison_analysis(
        texts=test_texts,
        output_dir=os.path.join(results_dir, "precision_analysis_results")
    )

    # Print key findings
    print("\n📊 Analysis Results:")
    if results and results.get('summary'):
        print(f"✅ Most similar precision pair: {results['summary'].get('most_similar_pair', 'N/A')}")
        print(f"⚠️  Least similar precision pair: {results['summary'].get('least_similar_pair', 'N/A')}")

    # Print grouped analysis results
    if results and results.get('grouped_comparisons'):
        grouped = results['grouped_comparisons']

        print("\n🔍 Group 1: Deterministic Cross-Precision Analysis")
        for comp_name, comp_data in grouped.get('deterministic_cross_precision', {}).items():
            simplified_name = comp_name.replace('_deterministic', '').replace('_vs_', ' vs ')
            print(f"   📊 {simplified_name}: L2={comp_data['l2_distance']['mean']:.2e}, Cosine={comp_data['cosine_similarity']['mean']:.6f}")

        print("\n🔍 Group 2: Non-Deterministic Cross-Precision Analysis")
        for comp_name, comp_data in grouped.get('nondeterministic_cross_precision', {}).items():
            simplified_name = comp_name.replace('_nondeterministic', '').replace('_vs_', ' vs ')
            print(f"   📊 {simplified_name}: L2={comp_data['l2_distance']['mean']:.2e}, Cosine={comp_data['cosine_similarity']['mean']:.6f}")

        print("\n🔍 Group 3: Within-Precision (Deterministic vs Non-Deterministic)")
        for comp_name, comp_data in grouped.get('within_precision_det_vs_nondet', {}).items():
            precision_name = comp_name.split('_')[0].upper()
            print(f"   📊 {precision_name} Det vs NonDet: L2={comp_data['l2_distance']['mean']:.2e}, Cosine={comp_data['cosine_similarity']['mean']:.6f}")

        # Show detailed FP32 deterministic vs non-deterministic if available
        fp32_det_vs_nondet_key = "fp32_deterministic_vs_fp32_nondeterministic"
        if results.get('pairwise_comparisons') and fp32_det_vs_nondet_key in results['pairwise_comparisons']:
            fp32_comparison = results['pairwise_comparisons'][fp32_det_vs_nondet_key]
            print(f"\n🎯 Detailed FP32 Deterministic vs Non-Deterministic Analysis:")
            print(f"   L2 Distance: {fp32_comparison['l2_distance']['mean']:.2e} ± {fp32_comparison['l2_distance']['std']:.2e}")
            print(f"   Cosine Similarity: {fp32_comparison['cosine_similarity']['mean']:.6f}")
            print(f"   Max Element Difference: {fp32_comparison['element_wise_diff']['max_abs_diff']:.2e}")

    # Save additional summary analysis to results directory
    if results:
        from datetime import datetime
        import json

        summary_analysis = {
            "timestamp": datetime.now().isoformat(),
            "model_path": model_path,
            "num_texts": len(test_texts),
            "summary": results.get('summary', {}),
            "group_analysis": {
                "deterministic_cross_precision_count": len(results.get('grouped_comparisons', {}).get('deterministic_cross_precision', {})),
                "nondeterministic_cross_precision_count": len(results.get('grouped_comparisons', {}).get('nondeterministic_cross_precision', {})),
                "within_precision_count": len(results.get('grouped_comparisons', {}).get('within_precision_det_vs_nondet', {})),
                "total_comparisons": len(results.get('pairwise_comparisons', {}))
            }
        }

        summary_file = os.path.join(results_dir, "precision_analysis_summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary_analysis, f, indent=2, default=str)

        print(f"\n📁 Summary analysis saved to: {summary_file}")

    print(f"\n📁 Detailed results saved to: {os.path.join(results_dir, 'precision_analysis_results')}/")
    print("📄 Check precision_comparison_report.md for full analysis")

    return results

if __name__ == "__main__":
    run_precision_analysis()
