"""
Test embedding reproducibility for the same configuration across multiple runs
This tests if the same embedding configuration produces identical results when run multiple times
"""

import sys
import os

# Add src directory to path
sys.path.append('src')

from embedding_reproducibility_tester import EmbeddingReproducibilityTester, EmbeddingConfig

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
    n_runs = 5  # Number of runs per configuration

    for config_name, config in configs_to_test:
        print(f"\n🧪 Testing {config_name} (precision: {config.precision}, deterministic: {config.deterministic})")
        print(f"   Running {n_runs} times to check reproducibility...")

        try:
            # Create tester for this configuration
            print(f"   🔧 Initializing {config_name}...")
            tester = EmbeddingReproducibilityTester(config)
            print(f"   ✅ Tester created successfully")

            # Test stability across multiple runs
            print(f"   🏃 Running stability test...")
            stability_results = tester.test_embedding_stability(
                texts=test_texts,
                n_runs=n_runs,
                use_process_isolation=False  # Set to True for maximum isolation
            )
            print(f"   ✅ Stability test completed")

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
    results_file = os.path.join(results_dir, "embedding_reproducibility_results_bge.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Save summary analysis
    summary_results = {
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "num_texts": len(test_texts),
        "n_runs": n_runs,
        "summary": {
            "total_configurations": len(all_results),
            "reproducible_configurations": len(reproducible_configs),
            "problematic_configurations": len(problematic_configs),
            "reproducible_config_names": reproducible_configs,
            "problematic_config_names": problematic_configs
        },
        "config_results": {}
    }

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

    summary_file = os.path.join(results_dir, "embedding_reproducibility_summary_bge.json")
    with open(summary_file, "w") as f:
        json.dump(summary_results, f, indent=2, default=str)

    print(f"\n📁 Detailed results saved to: {results_file}")
    print(f"📁 Summary analysis saved to: {summary_file}")

    return all_results

if __name__ == "__main__":
    run_embedding_reproducibility()
