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
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        configs_to_test.extend([
            ("BF16 Deterministic", EmbeddingConfig(precision="bf16", deterministic=True, model_name=model_path)),
            ("BF16 Non-Deterministic", EmbeddingConfig(precision="bf16", deterministic=False, model_name=model_path)),
            ("TF32 Deterministic", EmbeddingConfig(precision="tf32", deterministic=True, model_name=model_path)),
            ("TF32 Non-Deterministic", EmbeddingConfig(precision="tf32", deterministic=False, model_name=model_path)),
        ])

    all_results = {}
    n_runs = 5  # Number of runs per configuration

    for config_name, config in configs_to_test:
        print(f"\n🧪 Testing {config_name} (precision: {config.precision}, deterministic: {config.deterministic})")
        print(f"   Running {n_runs} times to check reproducibility...")

        try:
            # Create tester for this configuration
            tester = EmbeddingReproducibilityTester(config)

            # Test stability across multiple runs
            stability_results = tester.test_embedding_stability(
                texts=test_texts,
                n_runs=n_runs,
                use_process_isolation=False  # Set to True for maximum isolation
            )

            all_results[config_name] = stability_results

            # Print key findings
            if 'exact_match_rate' in stability_results:
                exact_match = stability_results['exact_match_rate']
                print(f"   ✅ Exact match rate: {exact_match:.3f} ({exact_match:.1%})")

            if 'mean_difference' in stability_results:
                mean_diff = stability_results['mean_difference']
                print(f"   📊 Mean difference across runs: {mean_diff:.2e}")

            if 'max_difference' in stability_results:
                max_diff = stability_results['max_difference']
                print(f"   📊 Max difference across runs: {max_diff:.2e}")

            # Reproducibility verdict
            if 'exact_match_rate' in stability_results:
                if stability_results['exact_match_rate'] >= 0.99:
                    print(f"   🎯 VERDICT: Highly reproducible")
                elif stability_results['exact_match_rate'] >= 0.95:
                    print(f"   ⚠️  VERDICT: Mostly reproducible")
                else:
                    print(f"   ❌ VERDICT: Poor reproducibility")

        except Exception as e:
            print(f"   ❌ Failed: {e}")
            all_results[config_name] = {"error": str(e)}

    # Summary
    print("\n" + "="*60)
    print("📊 REPRODUCIBILITY SUMMARY")
    print("="*60)

    reproducible_configs = []
    problematic_configs = []

    for config_name, results in all_results.items():
        if "error" in results:
            print(f"❌ {config_name}: Error occurred")
            problematic_configs.append(config_name)
        elif "exact_match_rate" in results:
            rate = results["exact_match_rate"]
            if rate >= 0.99:
                print(f"✅ {config_name}: {rate:.1%} exact match (REPRODUCIBLE)")
                reproducible_configs.append(config_name)
            else:
                print(f"⚠️  {config_name}: {rate:.1%} exact match (VARIABLE)")
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
        if "error" not in results:
            summary_results["config_results"][config_name] = {
                "exact_match_rate": results.get("exact_match_rate", 0),
                "mean_difference": results.get("mean_difference", 0),
                "max_difference": results.get("max_difference", 0),
                "reproducible": results.get("exact_match_rate", 0) >= 0.99
            }
        else:
            summary_results["config_results"][config_name] = {
                "error": results["error"],
                "reproducible": False
            }

    summary_file = os.path.join(results_dir, "embedding_reproducibility_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary_results, f, indent=2, default=str)

    print(f"\n📁 Detailed results saved to: {results_file}")
    print(f"📁 Summary analysis saved to: {summary_file}")

    return all_results

if __name__ == "__main__":
    run_embedding_reproducibility()
