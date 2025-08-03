#!/usr/bin/env python3
"""
Example script demonstrating how to use the dataset loader
Run this script to test loading datasets from the data/ directory
"""

import sys
import os
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset_loader import (
    load_dataset_for_reproducibility,
    MSMARCOLoader,
    WikipediaLoader,
    CustomDatasetLoader
)

def main():
    """Demonstrate dataset loading capabilities"""

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    print("🚀 RAG Reproducibility Dataset Loader Demo")
    print("=" * 50)

    # Example 1: Auto-detection
    print("\n📊 Example 1: Auto-Detection")
    try:
        documents, queries = load_dataset_for_reproducibility(
            file_path="../data/ms_marco_passages.csv",
            dataset_type="auto",
            num_docs=100,
            num_queries=10
        )
        print(f"✅ Loaded {len(documents)} documents and {len(queries)} queries")
        print(f"📄 Sample document: {documents[0]['text'][:100]}...")
        print(f"❓ Sample query: {queries[0]}")

    except FileNotFoundError:
        print("⚠️  MS MARCO CSV not found. Place ms_marco_passages.csv in data/ directory")
    except Exception as e:
        print(f"❌ Error: {e}")

    # Example 2: Specialized MS MARCO loader
    print("\n📊 Example 2: MS MARCO Specialized Loader")
    try:
        loader = MSMARCOLoader(data_dir="../data")
        documents = loader.load_msmarco_csv("../data/ms_marco_passages.csv", max_documents=50)
        queries = loader.generate_queries_from_documents(num_queries=5, query_strategy="first_sentence")

        print(f"✅ MS MARCO loader: {len(documents)} documents, {len(queries)} queries")
        stats = loader.get_dataset_statistics()
        print(f"📈 Avg text length: {stats['text_length_stats']['mean']:.1f} characters")

    except FileNotFoundError:
        print("⚠️  MS MARCO CSV not found")
    except Exception as e:
        print(f"❌ Error: {e}")

    # Example 3: Custom dataset
    print("\n📊 Example 3: Custom Dataset Loader")
    try:
        custom_loader = CustomDatasetLoader("demo_dataset", data_dir="../data")
        print("✅ Custom loader created successfully")
        print("💡 Place your custom CSV in data/ directory and load with:")
        print("   documents = custom_loader.load_csv_data('path/to/your/file.csv')")

    except Exception as e:
        print(f"❌ Error: {e}")

    # Example 4: Query generation strategies
    print("\n📊 Example 4: Query Generation Strategies")
    strategies = ["first_sentence", "keyword_extraction", "first_words", "title_style"]

    try:
        loader = MSMARCOLoader(data_dir="../data")
        if loader.load_processed_data():  # Try to load cached data
            for strategy in strategies:
                queries = loader.generate_queries_from_documents(
                    num_queries=3,
                    query_strategy=strategy
                )
                print(f"🎯 {strategy}: {queries[0] if queries else 'No queries generated'}")
        else:
            print("⚠️  No cached data found. Load a dataset first.")

    except Exception as e:
        print(f"❌ Error: {e}")

    # Example 5: Document sampling strategies
    print("\n📊 Example 5: Document Sampling Strategies")
    sampling_strategies = ["random", "first", "diverse", "balanced"]

    try:
        loader = MSMARCOLoader(data_dir="../data")
        if loader.load_processed_data():
            for strategy in sampling_strategies:
                sample = loader.get_sample_documents(num_docs=5, strategy=strategy)
                avg_length = sum(doc['metadata']['text_length'] for doc in sample) / len(sample)
                print(f"📄 {strategy}: {len(sample)} docs, avg length: {avg_length:.1f}")
        else:
            print("⚠️  No cached data found. Load a dataset first.")

    except Exception as e:
        print(f"❌ Error: {e}")

    print("\n" + "=" * 50)
    print("🎉 Demo completed! Check scripts/README.md for detailed documentation")
    print("💡 Place your CSV files in data/ directory to get started")


if __name__ == "__main__":
    main()
