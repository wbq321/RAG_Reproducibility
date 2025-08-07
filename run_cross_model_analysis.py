#!/usr/bin/env python3
"""
Example script to run cross-model retrieval ranking correlation analysis

This script demonstrates how to use the CrossModelRetrievalAnalyzer to test
ranking consistency across different embedding models (BGE, E5, QW).
"""

import sys
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

try:
    from cross_model_retrieval_analysis import CrossModelRetrievalAnalyzer
    
    def main():
        """
        Run cross-model ranking correlation analysis
        """
        print("Starting Cross-Model Retrieval Ranking Correlation Analysis")
        print("=" * 60)
        
        # Configuration
        models_to_test = ["bge", "e5", "qw"]  # You specified these three models
        top_k = 50  # You specified top-50
        max_documents = 5000  # Reasonable size for testing
        max_queries = 100    # Manageable number of queries
        
        print(f"Models to test: {models_to_test}")
        print(f"Top-K retrieval: {top_k}")
        print(f"Documents: {max_documents}")
        print(f"Queries: {max_queries}")
        print(f"Precision: Fixed FP32")
        print()
        
        # Initialize analyzer
        analyzer = CrossModelRetrievalAnalyzer(
            models_to_test=models_to_test,
            top_k=top_k,
            data_dir="data",
            output_dir="results"
        )
        
        # Run the complete analysis
        try:
            analyzer.run_complete_analysis(
                max_documents=max_documents,
                max_queries=max_queries
            )
            
            print("\\nAnalysis completed successfully!")
            print("Check the 'results' directory for:")
            print("- cross_model_ranking_correlations.png")
            print("- correlation_matrix_heatmaps.png") 
            print("- ranking_correlations.json")
            print("- analysis_summary.json")
            
        except FileNotFoundError as e:
            print(f"\\nError: {e}")
            print("\\nMake sure you have MSMARCO data in one of these locations:")
            print("- data/ms_marco_passages.csv")
            print("- data/msmarco.csv")
            print("- data/passages.csv")
            print("\\nYou can download MSMARCO from: https://microsoft.github.io/msmarco/")
            
        except ImportError as e:
            print(f"\\nMissing dependency: {e}")
            print("\\nPlease install required packages:")
            print("pip install faiss-cpu scipy numpy pandas matplotlib seaborn")
            
        except Exception as e:
            print(f"\\nUnexpected error: {e}")
            print("Check the log file 'cross_model_retrieval_analysis.log' for details")
    
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print(f"Could not import CrossModelRetrievalAnalyzer: {e}")
    print("Make sure all dependencies are installed and the script is in the correct location.")
