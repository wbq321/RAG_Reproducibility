"""
Test script for the integrated precision comparison functionality
Uses the enhanced EmbeddingReproducibilityTester with precision comparison
"""

import sys
import os
import logging

# Add src directory to path
sys.path.append('src')

from embedding_reproducibility_tester import PrecisionComparisonAnalyzer

def test_precision_comparison():
    """Test the integrated precision comparison functionality"""

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Your model path
    model_path = "/scratch/user/u.bw269205/shared_models/bge_model"

    # Test texts
    test_texts = [
        "Machine learning models require careful evaluation for reproducibility.",
        "Vector embeddings encode semantic information in high-dimensional space.",
        "Natural language processing systems benefit from precision optimization.",
        "GPU acceleration can introduce numerical precision variations.",
        "Reproducibility testing ensures consistent model behavior across runs.",
        "Different floating point formats affect computational accuracy.",
        "RAG systems combine retrieval and generation for enhanced performance.",
        "FAISS provides efficient similarity search capabilities.",
        "Distributed computing enables scalable machine learning workloads.",
        "Precision formats like FP16 and BF16 optimize memory usage."
    ]

    print("🔬 Testing Integrated Precision Comparison Analysis")
    print(f"📍 Model: {model_path}")
    print(f"📝 Using {len(test_texts)} test texts")

    try:
        # Create analyzer
        analyzer = PrecisionComparisonAnalyzer(model_path, device="cuda")

        # Run precision comparison
        results = analyzer.run_precision_comparison_analysis(
            texts=test_texts,
            output_dir="integrated_precision_results"
        )

        # Print key findings
        print("\n✅ Analysis completed successfully!")

        if results['summary']:
            summary = results['summary']
            print(f"📊 Most similar pair: {summary.get('most_similar_pair', 'N/A')}")
            print(f"📊 Least similar pair: {summary.get('least_similar_pair', 'N/A')}")

            # Show specific comparison details
            for comp_name, comp_data in results['pairwise_comparisons'].items():
                print(f"\n🔍 {comp_name.replace('_', ' ').upper()}:")
                print(f"   L2 Distance: {comp_data['l2_distance']['mean']:.2e}")
                print(f"   Cosine Similarity: {comp_data['cosine_similarity']['mean']:.6f}")
                print(f"   Max Element Diff: {comp_data['element_wise_diff']['max_abs_diff']:.2e}")

        print(f"\n📁 Results saved to: integrated_precision_results/")
        print("📄 Check precision_comparison_report.md for detailed analysis")

        return results

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_precision_comparison()
