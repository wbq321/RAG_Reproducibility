# Data Directory

📁 **Purpose**: Store your CSV dataset files here

## 🎯 Quick Start

1. **Place your CSV files here:**
   ```
   data/
   ├── ms_marco_passages.csv      # Your MS MARCO dataset
   ├── wikipedia_articles.csv     # Wikipedia dataset (optional)
   └── custom_dataset.csv         # Any custom dataset
   ```

2. **Use the dataset loader:**
   ```python
   from scripts.dataset_loader import load_dataset_for_reproducibility

   # Load your data
   docs, queries = load_dataset_for_reproducibility(
       file_path="data/ms_marco_passages.csv",
       num_docs=5000,
       num_queries=100
   )
   ```

## 📋 CSV Format Expected

Your CSV should have at least a text column:

```csv
id,text,category
1,"Document content here...","research"
2,"Another document...","academic"
```

## 📝 Auto-Generated Files

The scripts will create these files automatically:
- `processed_documents_*.json` - Cached processed data
- `dataset_metadata_*.json` - Dataset statistics
- `evaluation_dataset_*.json` - Ready-to-use evaluation sets

## 🚀 Running Tests

```bash
# Use dataset from data directory
python run_comprehensive_tests.py --dataset data/ms_marco_passages.csv

# Quick test mode
python run_comprehensive_tests.py --dataset data/my_dataset.csv --quick

# Use simulated data
python run_comprehensive_tests.py --use-simulated
```

## 📖 Documentation

For detailed usage instructions, see: `scripts/README.md`
