#!/usr/bin/env python3
"""
Test script to verify local models work correctly on cluster
"""

import os
import sys
from pathlib import Path

# Set offline mode
os.environ['TRANSFORMERS_OFFLINE'] = '1' 
os.environ['HF_DATASETS_OFFLINE'] = '1'

# Model paths on cluster
MODEL_PATHS = {
    "bge": "/scratch/user/u.bw269205/shared_models/bge_model",
    "e5": "/scratch/user/u.bw269205/shared_models/intfloat_e5-base-v2",
    "qw": "/scratch/user/u.bw269205/shared_models/Qwen_Qwen3-Embedding-0.6B"
}

def test_model_loading():
    """Test that each model can be loaded successfully"""
    
    print("Testing model loading...")
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✓ SentenceTransformers imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import SentenceTransformers: {e}")
        return False
    
    for model_name, model_path in MODEL_PATHS.items():
        print(f"\nTesting {model_name} at {model_path}...")
        
        # Check if path exists
        if not os.path.exists(model_path):
            print(f"✗ Model path does not exist: {model_path}")
            continue
            
        try:
            # Try to load the model
            model = SentenceTransformer(model_path)
            print(f"✓ {model_name} loaded successfully")
            
            # Test encoding a simple sentence
            test_text = "This is a test sentence for embedding generation."
            embedding = model.encode(test_text)
            print(f"✓ {model_name} embedding generated: shape={embedding.shape}")
            
            del model  # Free memory
            
        except Exception as e:
            print(f"✗ Failed to load {model_name}: {e}")
            import traceback
            traceback.print_exc()
            
    print("\nModel loading test completed.")

def test_embedding_config():
    """Test the EmbeddingConfig with local paths"""
    
    print("\nTesting EmbeddingConfig...")
    
    # Add project paths
    project_root = "/scratch/user/u.bw269205/rag_reproducibility/RAG_Reproducibility"
    sys.path.append(project_root)
    sys.path.append(os.path.join(project_root, "src"))
    
    try:
        from embedding_reproducibility_tester import EmbeddingConfig, EmbeddingReproducibilityTester
        print("✓ EmbeddingConfig imported successfully")
        
        # Test each model
        for model_name, model_path in MODEL_PATHS.items():
            print(f"\nTesting EmbeddingConfig for {model_name}...")
            
            try:
                config = EmbeddingConfig(
                    model_name=model_path,
                    precision="fp32",
                    deterministic=True,
                    device="cuda" if os.environ.get('CUDA_VISIBLE_DEVICES') else "cpu",
                    batch_size=16,
                    max_length=512,
                    normalize_embeddings=True
                )
                print(f"✓ {model_name} config created successfully")
                
                # Test tester initialization
                tester = EmbeddingReproducibilityTester(config)
                print(f"✓ {model_name} tester initialized successfully")
                
                # Test encoding
                test_texts = ["Test sentence one.", "Test sentence two."]
                embeddings = tester.encode_texts(test_texts)
                print(f"✓ {model_name} embeddings generated: shape={embeddings.shape}")
                
            except Exception as e:
                print(f"✗ Failed EmbeddingConfig test for {model_name}: {e}")
                import traceback
                traceback.print_exc()
                
    except ImportError as e:
        print(f"✗ Failed to import EmbeddingConfig: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test local models on cluster")
    parser.add_argument("--test-loading", action="store_true", 
                       help="Test model loading with SentenceTransformers")
    parser.add_argument("--test-config", action="store_true",
                       help="Test EmbeddingConfig and tester")
    parser.add_argument("--test-all", action="store_true",
                       help="Run all tests")
    
    args = parser.parse_args()
    
    if args.test_all or not any([args.test_loading, args.test_config]):
        # Run all tests by default
        test_model_loading()
        test_embedding_config()
    else:
        if args.test_loading:
            test_model_loading()
        if args.test_config:
            test_embedding_config()
    
    print("\nTest completed!")
