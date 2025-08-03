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

    # Create analyzer
    analyzer = PrecisionComparisonAnalyzer(model_path, device="cuda")

    # Run analysis
    results = analyzer.run_precision_comparison_analysis(
        texts=test_texts,
        output_dir="precision_analysis_results"
    )

    # Print key findings
    print("\n📊 Key Findings:")
    if results and results.get('summary'):
        print(f"✅ Most similar precision pair: {results['summary'].get('most_similar_pair', 'N/A')}")
        print(f"⚠️  Least similar precision pair: {results['summary'].get('least_similar_pair', 'N/A')}")

        # Print specific comparison if FP32 vs FP16 exists
        if results.get('pairwise_comparisons') and "fp32_vs_fp16" in results['pairwise_comparisons']:
            fp32_fp16 = results['pairwise_comparisons']['fp32_vs_fp16']
            print(f"\n🔍 FP32 vs FP16 Analysis:")
            print(f"   L2 Distance: {fp32_fp16['l2_distance']['mean']:.2e} ± {fp32_fp16['l2_distance']['std']:.2e}")
            print(f"   Cosine Similarity: {fp32_fp16['cosine_similarity']['mean']:.6f}")
            print(f"   Max Element Difference: {fp32_fp16['element_wise_diff']['max_abs_diff']:.2e}")

    print(f"\n📁 Detailed results saved to: precision_analysis_results/")
    print("📄 Check precision_comparison_report.md for full analysis")

    return results

if __name__ == "__main__":
    run_precision_analysis()
