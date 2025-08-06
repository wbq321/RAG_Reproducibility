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

    # Debug GPU capability
    import torch
    print(f"\n🔍 GPU Capability Check:")
    print(f"   CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"   Number of GPUs: {torch.cuda.device_count()}")
        print(f"   Current GPU: {torch.cuda.current_device()}")

        # Check capability of current device
        current_device = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(current_device)
        print(f"   GPU {current_device} Capability: {capability[0]}.{capability[1]}")

        # Check if advanced precisions should be supported
        supports_advanced = capability[0] >= 8
        print(f"   Supports Advanced Precisions (TF32/BF16): {supports_advanced}")

        if supports_advanced:
            print(f"   ✅ Will test FP32, FP16, BF16, TF32 precisions")
        else:
            print(f"   ⚠️  Will test FP32, FP16 only - GPU compute capability {capability[0]}.{capability[1]} < 8.0")
    else:
        print(f"   ❌ CUDA not available - precision testing may be limited")

    # Create results directory
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    print(f"📁 Results will be saved to: {results_dir}/")

    # Create analyzer
    print(f"\n🔧 Initializing precision analyzer...")

    try:
        analyzer = PrecisionComparisonAnalyzer(model_path, device="cuda")
        print(f"✅ Analyzer created successfully")
    except Exception as e:
        print(f"❌ Failed to create analyzer: {e}")
        print(f"📝 Error type: {type(e).__name__}")
        return None

    # Run analysis - save to results directory
    print(f"\n🏃 Running precision comparison analysis...")
    print(f"📊 This will test 8 configurations (4 precisions × 2 deterministic modes)")

    try:
        results = analyzer.run_precision_comparison_analysis(
            texts=test_texts,
            output_dir=os.path.join(results_dir, "precision_analysis_results")
        )
        print(f"✅ Precision analysis completed successfully")
    except Exception as e:
        print(f"❌ Precision analysis failed: {e}")
        print(f"📝 Error type: {type(e).__name__}")

        # More detailed error info for precision-related issues
        if "precision" in str(e).lower() or "bf16" in str(e).lower() or "tf32" in str(e).lower():
            print(f"💡 This might be a precision-specific error")
            print(f"💡 Consider checking GPU compute capability and model compatibility")

        return None

    # Print key findings
    print("\n" + "="*60)
    print("📊 PRECISION ANALYSIS RESULTS")
    print("="*60)

    if results and results.get('summary'):
        print(f"✅ Most similar precision pair: {results['summary'].get('most_similar_pair', 'N/A')}")
        print(f"⚠️  Least similar precision pair: {results['summary'].get('least_similar_pair', 'N/A')}")
    else:
        print("❌ No summary results available")
        return results

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
            print(f"\n🎯 Key Finding - FP32 Deterministic vs Non-Deterministic:")
            l2_mean = fp32_comparison['l2_distance']['mean']
            l2_std = fp32_comparison['l2_distance']['std']
            cosine_mean = fp32_comparison['cosine_similarity']['mean']
            max_diff = fp32_comparison['element_wise_diff']['max_abs_diff']

            print(f"   L2 Distance: {l2_mean:.2e} ± {l2_std:.2e}")
            print(f"   Cosine Similarity: {cosine_mean:.6f}")
            print(f"   Max Element Difference: {max_diff:.2e}")

            # Interpretation
            if isinstance(l2_mean, str):
                l2_val = float(l2_mean.replace('e-', 'E-').replace('e+', 'E+'))
            else:
                l2_val = float(l2_mean)

            if l2_val == 0:
                print(f"   🚨 CRITICAL: Deterministic and non-deterministic modes are identical!")
                print(f"   💡 This suggests non-deterministic mode is not working properly")
            elif l2_val < 1e-6:
                print(f"   ✅ Very small differences detected")
            else:
                print(f"   ✅ Expected differences detected between modes")

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

    # Final summary
    print(f"\n" + "="*60)
    print("✅ PRECISION ANALYSIS COMPLETE!")
    print("="*60)

    if results:
        total_comparisons = len(results.get('pairwise_comparisons', {}))
        print(f"📊 Total comparisons performed: {total_comparisons}")

        # Check for deterministic vs non-deterministic issues
        det_vs_nondet_comps = {k: v for k, v in results.get('pairwise_comparisons', {}).items()
                              if 'deterministic_vs_' in k and 'nondeterministic' in k}

        zero_diff_count = 0
        for comp_name, comp_data in det_vs_nondet_comps.items():
            l2_mean_str = comp_data.get('l2_distance', {}).get('mean', '0')
            try:
                # Handle both string and float types
                if isinstance(l2_mean_str, str):
                    l2_mean = float(l2_mean_str.replace('e-', 'E-').replace('e+', 'E+'))
                else:
                    l2_mean = float(l2_mean_str)

                if l2_mean == 0:
                    zero_diff_count += 1
            except (ValueError, TypeError, AttributeError):
                continue

        if zero_diff_count > 0:
            print(f"⚠️  {zero_diff_count}/{len(det_vs_nondet_comps)} precision types show identical deterministic/non-deterministic results")
            print(f"💡 Consider checking the non-deterministic implementation")
        else:
            print(f"✅ All precision types show expected differences between deterministic/non-deterministic modes")

    print("="*60)

    return results

if __name__ == "__main__":
    run_precision_analysis()
