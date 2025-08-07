#!/usr/bin/env python3
"""
Analyze embedding uncertainty results for all models
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append('.')

from analyze_embedding_uncertainty import EmbeddingUncertaintyAnalyzer

def analyze_all_models():
    """Analyze results for all tested models"""
    
    results_base = Path("results")
    models = ["bge", "e5", "qw"]
    
    print("🔬 Starting Multi-Model Embedding Uncertainty Analysis")
    print("=" * 60)
    
    successful_analyses = []
    failed_analyses = []
    
    for model in models:
        model_results_dir = results_base / model
        
        if not model_results_dir.exists():
            print(f"⚠️ Skipping {model.upper()}: Results directory not found - {model_results_dir}")
            failed_analyses.append(model)
            continue
            
        print(f"\n📊 Analyzing {model.upper()} Model Results")
        print("-" * 40)
        
        try:
            # Create analyzer for this model
            analyzer = EmbeddingUncertaintyAnalyzer(results_dir=str(model_results_dir))
            
            # Run the complete analysis pipeline
            analyzer.run_analysis()
            
            successful_analyses.append(model)
            print(f"✅ {model.upper()} analysis completed successfully")
            
        except Exception as e:
            print(f"❌ {model.upper()} analysis failed: {e}")
            failed_analyses.append(model)
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 MULTI-MODEL ANALYSIS SUMMARY")
    print("=" * 60)
    
    print(f"✅ Successfully analyzed: {len(successful_analyses)} models")
    for model in successful_analyses:
        print(f"   - {model.upper()}")
    
    if failed_analyses:
        print(f"\n❌ Failed to analyze: {len(failed_analyses)} models") 
        for model in failed_analyses:
            print(f"   - {model.upper()}")
    
    print(f"\n📁 Results saved in:")
    for model in successful_analyses:
        print(f"   - results/{model}/analyze/")
    
    return successful_analyses, failed_analyses

def analyze_single_model(model_name):
    """Analyze results for a single model"""
    
    model_results_dir = Path("results") / model_name
    
    if not model_results_dir.exists():
        print(f"❌ Results directory not found: {model_results_dir}")
        return False
        
    print(f"🔬 Analyzing {model_name.upper()} Model Results")
    print("=" * 60)
    
    try:
        analyzer = EmbeddingUncertaintyAnalyzer(results_dir=str(model_results_dir))
        analyzer.run_analysis()
        print(f"✅ {model_name.upper()} analysis completed successfully")
        print(f"📁 Results saved in: results/{model_name}/analyze/")
        return True
        
    except Exception as e:
        print(f"❌ {model_name.upper()} analysis failed: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Analyze specific model
        model_name = sys.argv[1].lower()
        if model_name in ["bge", "e5", "qw"]:
            analyze_single_model(model_name)
        else:
            print(f"❌ Unknown model: {model_name}")
            print("   Supported models: bge, e5, qw")
    else:
        # Analyze all models
        analyze_all_models()
