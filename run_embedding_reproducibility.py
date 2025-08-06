"""
Test embedding reproducibility for the same configuration across multiple runs
This tests if the same embedding configuration produces identical results when run multiple times
"""

import sys
import os
import time
import statistics

# Add src directory to path
sys.path.append('src')

from embedding_reproducibility_tester import EmbeddingReproducibilityTester, EmbeddingConfig

def create_timing_plot(timing_results, output_dir="results"):
    """Create a bar plot comparing embedding generation times across configurations"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        # Filter successful timings
        successful_timings = {k: v for k, v in timing_results.items() if "error" not in v}

        if not successful_timings:
            print("⚠️  No successful timing results to plot")
            return

        # Prepare data
        config_names = list(successful_timings.keys())
        avg_times = [timing['avg_embedding_time_s'] for timing in successful_timings.values()]
        std_times = [timing['std_embedding_time_s'] for timing in successful_timings.values()]

        # Create figure
        plt.figure(figsize=(12, 8))

        # Create bar plot with error bars
        bars = plt.bar(range(len(config_names)), avg_times, yerr=std_times,
                      capsize=5, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])

        # Customize plot
        plt.xlabel('Configuration', fontsize=12, fontweight='bold')
        plt.ylabel('Average Embedding Time (seconds)', fontsize=12, fontweight='bold')
        plt.title('Embedding Generation Time Comparison\nAcross Precision Configurations',
                 fontsize=14, fontweight='bold')

        # Set x-axis labels
        plt.xticks(range(len(config_names)), [name.replace(' ', '\n') for name in config_names],
                  rotation=45, ha='right')

        # Add value labels on bars
        for i, (bar, time, std) in enumerate(zip(bars, avg_times, std_times)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.001,
                    f'{time:.3f}s', ha='center', va='bottom', fontweight='bold')

        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(output_dir, "embedding_timing_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"📊 Timing comparison plot saved to: {plot_path}")

    except ImportError:
        print("⚠️  matplotlib not available - skipping timing plot generation")
    except Exception as e:
        print(f"❌ Error creating timing plot: {e}")

def run_embedding_reproducibility():
    """Test embedding reproducibility within same configurations"""

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

    print("🔬 Starting Embedding Reproducibility Testing")
    print(f"📍 Model: {model_path}")
    print(f"📝 Testing with {len(test_texts)} sample texts")
    print("🔄 Testing same config multiple times for reproducibility")

    # Create results directory
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    print(f"📁 Results will be saved to: {results_dir}/")

    # Define configurations to test
    configs_to_test = [
        ("FP32 Deterministic", EmbeddingConfig(precision="fp32", deterministic=True, model_name=model_path)),
        ("FP32 Non-Deterministic", EmbeddingConfig(precision="fp32", deterministic=False, model_name=model_path)),
        ("FP16 Deterministic", EmbeddingConfig(precision="fp16", deterministic=True, model_name=model_path)),
        ("FP16 Non-Deterministic", EmbeddingConfig(precision="fp16", deterministic=False, model_name=model_path)),
    ]

    # Add advanced precisions if supported
    import torch

    # Debug GPU capability
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
            configs_to_test.extend([
                ("BF16 Deterministic", EmbeddingConfig(precision="bf16", deterministic=True, model_name=model_path)),
                ("BF16 Non-Deterministic", EmbeddingConfig(precision="bf16", deterministic=False, model_name=model_path)),
                ("TF32 Deterministic", EmbeddingConfig(precision="tf32", deterministic=True, model_name=model_path)),
                ("TF32 Non-Deterministic", EmbeddingConfig(precision="tf32", deterministic=False, model_name=model_path)),
            ])
            print(f"   ✅ Added TF32 and BF16 configurations")
        else:
            print(f"   ❌ TF32/BF16 not supported - GPU compute capability {capability[0]}.{capability[1]} < 8.0")
    else:
        print(f"   ❌ CUDA not available")

    print(f"\n📋 Total configurations to test: {len(configs_to_test)}")
    for i, (name, config) in enumerate(configs_to_test, 1):
        print(f"   {i}. {name} (precision: {config.precision})")

    all_results = {}
    timing_results = {}  # Store timing information separately
    n_runs = 5  # Number of runs per configuration

    for config_name, config in configs_to_test:
        print(f"\n🧪 Testing {config_name} (precision: {config.precision}, deterministic: {config.deterministic})")
        print(f"   Running {n_runs} times to check reproducibility...")

        try:
            # Create tester for this configuration
            print(f"   🔧 Initializing {config_name}...")

            # Time the initialization
            init_start = time.perf_counter()
            tester = EmbeddingReproducibilityTester(config)
            init_time = time.perf_counter() - init_start

            print(f"   ✅ Tester created successfully (init time: {init_time:.3f}s)")

            # Test stability across multiple runs WITH TIMING
            print(f"   🏃 Running stability test with timing measurements...")

            # Collect timing data for each run
            run_times = []

            stability_start = time.perf_counter()
            stability_results = tester.test_embedding_stability(
                texts=test_texts,
                n_runs=n_runs,
                use_process_isolation=False  # Set to True for maximum isolation
            )
            total_stability_time = time.perf_counter() - stability_start

            # Additional timing test: measure individual embedding generation
            print(f"   ⏱️  Measuring individual embedding generation times...")
            for run_idx in range(3):  # Do 3 timing runs
                embed_start = time.perf_counter()
                _ = tester.encode_texts(test_texts)
                embed_time = time.perf_counter() - embed_start
                run_times.append(embed_time)
                print(f"      Run {run_idx + 1}: {embed_time:.3f}s")

            # Calculate timing statistics
            avg_embed_time = statistics.mean(run_times)
            std_embed_time = statistics.stdev(run_times) if len(run_times) > 1 else 0.0
            min_embed_time = min(run_times)
            max_embed_time = max(run_times)

            # Store timing results
            timing_results[config_name] = {
                "initialization_time_s": init_time,
                "total_stability_test_time_s": total_stability_time,
                "individual_runs": run_times,
                "avg_embedding_time_s": avg_embed_time,
                "std_embedding_time_s": std_embed_time,
                "min_embedding_time_s": min_embed_time,
                "max_embedding_time_s": max_embed_time,
                "num_texts": len(test_texts),
                "time_per_text_ms": (avg_embed_time / len(test_texts)) * 1000,
                "texts_per_second": len(test_texts) / avg_embed_time
            }

            print(f"   ✅ Stability test completed (total time: {total_stability_time:.3f}s)")
            print(f"   ⚡ Avg embedding time: {avg_embed_time:.3f}s ± {std_embed_time:.3f}s")
            print(f"   📈 Processing rate: {len(test_texts) / avg_embed_time:.1f} texts/sec")
            print(f"   📏 Time per text: {(avg_embed_time / len(test_texts)) * 1000:.1f}ms")

            all_results[config_name] = stability_results

            # Print key findings
            metrics = stability_results.get('metrics', {})
            if 'exact_match_rate' in metrics:
                exact_match = metrics['exact_match_rate']
                print(f"   📊 Exact match rate: {exact_match:.3f} ({exact_match:.1%})")

            # Check for L2 distance in metrics
            l2_distance = metrics.get('l2_distance', {})
            if 'mean' in l2_distance:
                l2_mean = float(l2_distance['mean']) if isinstance(l2_distance['mean'], str) else l2_distance['mean']
                print(f"   📊 Mean L2 distance: {l2_mean:.2e}")

            # Check for max absolute difference
            max_abs_diff = metrics.get('max_abs_difference', {})
            if 'mean' in max_abs_diff:
                max_diff = float(max_abs_diff['mean']) if isinstance(max_abs_diff['mean'], str) else max_abs_diff['mean']
                print(f"   📊 Mean max absolute difference: {max_diff:.2e}")

            # Reproducibility verdict
            exact_match_rate = metrics.get('exact_match_rate', 0)
            if exact_match_rate >= 0.99:
                print(f"   🎯 VERDICT: Highly reproducible")
            elif exact_match_rate >= 0.95:
                print(f"   ⚠️  VERDICT: Mostly reproducible")
            else:
                print(f"   ❌ VERDICT: Poor reproducibility")

        except Exception as e:
            print(f"   ❌ Failed: {str(e)}")
            print(f"   📝 Error type: {type(e).__name__}")

            # More detailed error info for precision-related issues
            if "precision" in str(e).lower() or "bf16" in str(e).lower() or "tf32" in str(e).lower():
                print(f"   💡 This might be a precision-specific error")

            all_results[config_name] = {"error": str(e), "error_type": type(e).__name__}
            timing_results[config_name] = {"error": str(e), "error_type": type(e).__name__}

    # Timing Summary
    print("\n" + "="*60)
    print("⏱️  TIMING PERFORMANCE SUMMARY")
    print("="*60)

    # Sort configurations by average embedding time
    successful_timings = {k: v for k, v in timing_results.items() if "error" not in v}

    if successful_timings:
        sorted_configs = sorted(successful_timings.items(), key=lambda x: x[1]['avg_embedding_time_s'])

        print(f"{'Configuration':<25} {'Avg Time':<10} {'Rate':<12} {'Per Text':<10} {'Std Dev':<10}")
        print("-" * 75)

        for config_name, timing in sorted_configs:
            avg_time = timing['avg_embedding_time_s']
            rate = timing['texts_per_second']
            per_text = timing['time_per_text_ms']
            std_dev = timing['std_embedding_time_s']

            print(f"{config_name:<25} {avg_time:<10.3f}s {rate:<12.1f}/s {per_text:<10.1f}ms ±{std_dev:<9.3f}s")

        # Find fastest and slowest
        fastest = min(successful_timings.items(), key=lambda x: x[1]['avg_embedding_time_s'])
        slowest = max(successful_timings.items(), key=lambda x: x[1]['avg_embedding_time_s'])

        print(f"\n🚀 Fastest: {fastest[0]} ({fastest[1]['avg_embedding_time_s']:.3f}s)")
        print(f"🐌 Slowest: {slowest[0]} ({slowest[1]['avg_embedding_time_s']:.3f}s)")

        # Calculate speedup
        speedup = slowest[1]['avg_embedding_time_s'] / fastest[1]['avg_embedding_time_s']
        print(f"⚡ Speedup: {speedup:.2f}x faster")

    # Failed configurations timing
    failed_timings = {k: v for k, v in timing_results.items() if "error" in v}
    if failed_timings:
        print(f"\n❌ Failed configurations: {list(failed_timings.keys())}")

    # Summary
    print("\n" + "="*60)
    print("📊 REPRODUCIBILITY SUMMARY")
    print("="*60)

    reproducible_configs = []
    problematic_configs = []

    for config_name, results in all_results.items():
        if "error" in results:
            print(f"❌ {config_name}: Error occurred - {results.get('error_type', 'Unknown')}")
            problematic_configs.append(config_name)
        elif "metrics" in results:
            # Updated to access metrics correctly
            metrics = results["metrics"]
            rate = metrics.get("exact_match_rate", 0)
            if rate >= 0.99:
                print(f"✅ {config_name}: {rate:.1%} exact match (REPRODUCIBLE)")
                reproducible_configs.append(config_name)
            else:
                print(f"⚠️  {config_name}: {rate:.1%} exact match (VARIABLE)")
                problematic_configs.append(config_name)
        else:
            print(f"❓ {config_name}: Unexpected result structure")
            problematic_configs.append(config_name)

    print(f"\n🎯 {len(reproducible_configs)} highly reproducible configurations")
    print(f"⚠️  {len(problematic_configs)} configurations with variability")

    # Save detailed results to results directory
    import json
    from datetime import datetime

    # Save main results
    results_file = os.path.join(results_dir, "embedding_reproducibility_results.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Save summary analysis
    summary_results = {
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "num_texts": len(test_texts),
        "n_runs": n_runs,
        "timing_summary": {
            "successful_configs": len([k for k, v in timing_results.items() if "error" not in v]),
            "failed_configs": len([k for k, v in timing_results.items() if "error" in v]),
            "fastest_config": None,
            "slowest_config": None,
            "speedup_factor": None
        },
        "summary": {
            "total_configurations": len(all_results),
            "reproducible_configurations": len(reproducible_configs),
            "problematic_configurations": len(problematic_configs),
            "reproducible_config_names": reproducible_configs,
            "problematic_config_names": problematic_configs
        },
        "config_results": {},
        "timing_results": timing_results  # Add timing results to summary
    }

    # Add timing summary statistics
    successful_timings = {k: v for k, v in timing_results.items() if "error" not in v}
    if successful_timings:
        fastest = min(successful_timings.items(), key=lambda x: x[1]['avg_embedding_time_s'])
        slowest = max(successful_timings.items(), key=lambda x: x[1]['avg_embedding_time_s'])
        speedup = slowest[1]['avg_embedding_time_s'] / fastest[1]['avg_embedding_time_s']

        summary_results["timing_summary"].update({
            "fastest_config": fastest[0],
            "slowest_config": slowest[0],
            "speedup_factor": speedup,
            "fastest_time_s": fastest[1]['avg_embedding_time_s'],
            "slowest_time_s": slowest[1]['avg_embedding_time_s']
        })

    # Add simplified config results for analysis
    for config_name, results in all_results.items():
        if "error" not in results and "metrics" in results:
            metrics = results["metrics"]
            summary_results["config_results"][config_name] = {
                "exact_match_rate": metrics.get("exact_match_rate", 0),
                "l2_distance_mean": float(metrics.get("l2_distance", {}).get("mean", 0)) if isinstance(metrics.get("l2_distance", {}).get("mean"), str) else metrics.get("l2_distance", {}).get("mean", 0),
                "cosine_similarity_mean": float(metrics.get("cosine_similarity", {}).get("mean", 1)) if isinstance(metrics.get("cosine_similarity", {}).get("mean"), str) else metrics.get("cosine_similarity", {}).get("mean", 1),
                "reproducible": metrics.get("exact_match_rate", 0) >= 0.99
            }
        else:
            summary_results["config_results"][config_name] = {
                "error": results.get("error", "Unknown error"),
                "error_type": results.get("error_type", "Unknown"),
                "reproducible": False
            }

    summary_file = os.path.join(results_dir, "embedding_reproducibility_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary_results, f, indent=2, default=str)

    print(f"\n📁 Detailed results saved to: {results_file}")
    print(f"📁 Summary analysis saved to: {summary_file}")
    print(f"⏱️  Timing results included in summary for performance analysis")

    # Create timing visualization
    create_timing_plot(timing_results, results_dir)

    return all_results, timing_results

if __name__ == "__main__":
    run_embedding_reproducibility()
