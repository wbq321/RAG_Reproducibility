# Dataset Loading Scripts

This directory contains scripts for loading and preprocessing various datasets for RAG reproducibility testing.

## 📁 Directory Structure

```
scripts/
├── dataset_loader.py     # Universal dataset loader with support for multiple formats
└── README.md            # This documentation file

data/                    # Place your CSV files here
├── ms_marco_passages.csv     # MS MARCO dataset (place your CSV here)
├── wikipedia_articles.csv   # Wikipedia dataset (optional)
└── custom_dataset.csv       # Any custom dataset (optional)
```

## 🚀 Quick Start

### 1. Basic Usage - Auto-Detection
```python
from scripts.dataset_loader import load_dataset_for_reproducibility

# Auto-detect dataset type and load
documents, queries = load_dataset_for_reproducibility(
    file_path="data/ms_marco_passages.csv",
    dataset_type="auto",  # Auto-detect from filename
    num_docs=5000,
    num_queries=100
)
```

### 2. MS MARCO Dataset
```python
from scripts.dataset_loader import MSMARCOLoader

# Using specialized MS MARCO loader
loader = MSMARCOLoader()
documents = loader.load_msmarco_csv("data/ms_marco_passages.csv")
queries = loader.generate_queries_from_documents(num_queries=50)
```

### 3. Custom Dataset
```python
from scripts.dataset_loader import load_dataset_for_reproducibility

# Load custom CSV with specific column mapping
documents, queries = load_dataset_for_reproducibility(
    file_path="data/research_papers.csv",
    dataset_type="custom",
    dataset_name="research_corpus",
    text_column="abstract",          # Specify text column
    id_column="paper_id",           # Specify ID column
    query_strategy="keyword_extraction",
    doc_sampling="balanced"
)
```

## 📊 Supported Dataset Types

### Automatic Detection
The loader automatically detects dataset types based on filename patterns:
- Files containing "marco" → MS MARCO loader
- Files containing "wiki" → Wikipedia loader
- Files containing "crawl" → Common Crawl loader
- Other files → Custom loader

### Specialized Loaders
- **MSMARCOLoader**: Optimized for MS MARCO passage datasets
- **WikipediaLoader**: Handles Wikipedia article dumps
- **CommonCrawlLoader**: Processes Common Crawl data
- **CustomDatasetLoader**: Flexible loader for any dataset format

## 📁 Supported File Formats

### CSV Files
- **Auto-detection** of text and ID columns
- **Flexible encoding** support (UTF-8, Latin-1, CP1252, UTF-16)
- **Custom column mapping** for any CSV structure

```python
# CSV with custom columns
documents = loader.load_csv_data(
    csv_path="data/papers.csv",
    text_column="abstract",
    id_column="doi",
    additional_columns=["title", "authors", "year"]
)
```

### JSON Files
- **Multiple JSON structures** supported
- **Nested object handling**
- **Flexible field mapping**

```python
# JSON dataset
documents = loader.load_json_data(
    json_path="data/articles.json",
    text_field="content",
    id_field="article_id"
)
```

### Text Files
- **Plain text** with customizable separators
- **Document splitting** by paragraphs or custom delimiters

```python
# Plain text file
documents = loader.load_text_data(
    text_path="data/corpus.txt",
    separator="\n\n"  # Split by double newlines
)
```

## 🎯 Query Generation Strategies

### Available Strategies
1. **first_sentence**: Extract the first sentence of each document
2. **random_sentence**: Extract a random sentence from each document
3. **first_words**: Use the first 15 words of each document
4. **keyword_extraction**: Extract meaningful keywords (skip stop words)
5. **title_style**: Extract title-like capitalized phrases

### Usage Example
```python
queries = loader.generate_queries_from_documents(
    num_queries=100,
    query_strategy="keyword_extraction",
    min_query_length=10,
    max_query_length=200
)
```

## 📈 Document Sampling Strategies

### Available Strategies
1. **random**: Random sampling of documents
2. **first**: Take the first N documents sequentially
3. **diverse**: Sample documents with diverse text lengths
4. **balanced**: Balance between short, medium, and long documents

### Usage Example
```python
sample_docs = loader.get_sample_documents(
    num_docs=1000,
    strategy="balanced"  # Mix of short/medium/long docs
)
```

## ⚙️ Advanced Configuration

### Loading with Custom Parameters
```python
documents, queries = load_dataset_for_reproducibility(
    file_path="data/custom_dataset.csv",
    dataset_type="custom",
    dataset_name="my_research_corpus",
    num_docs=3000,
    num_queries=150,

    # CSV-specific parameters
    text_column="document_text",
    id_column="doc_id",
    additional_columns=["category", "source"],
    encoding="utf-8",

    # Query generation
    query_strategy="first_sentence",
    min_query_length=15,
    max_query_length=150,

    # Document sampling
    doc_sampling="diverse",

    # Caching
    force_reload=False  # Use cached data if available
)
```

### Batch Processing Multiple Datasets
```python
datasets = [
    ("data/ms_marco.csv", "ms_marco"),
    ("data/wikipedia.json", "wikipedia"),
    ("data/research_papers.csv", "custom")
]

all_documents = []
all_queries = []

for file_path, dataset_type in datasets:
    docs, queries = load_dataset_for_reproducibility(
        file_path=file_path,
        dataset_type=dataset_type,
        num_docs=1000,
        num_queries=50
    )
    all_documents.extend(docs)
    all_queries.extend(queries)

print(f"Total documents: {len(all_documents)}")
print(f"Total queries: {len(all_queries)}")
```

## 💾 Data Caching & Performance

### Automatic Caching
- Processed datasets are automatically cached as JSON files
- Cache files stored in `data/processed_documents_{dataset_name}.json`
- Metadata saved in `data/dataset_metadata_{dataset_name}.json`

### Cache Management
```python
# Force reload from source (skip cache)
documents, queries = load_dataset_for_reproducibility(
    file_path="data/dataset.csv",
    force_reload=True
)

# Load from cache manually
loader = MSMARCOLoader()
if loader.load_processed_data():
    print("Loaded from cache")
else:
    print("Cache not found, loading from source")
    loader.load_csv_data("data/ms_marco.csv")
```

## 📊 Dataset Statistics

### Get Dataset Information
```python
loader = MSMARCOLoader()
loader.load_msmarco_csv("data/ms_marco.csv")

stats = loader.get_dataset_statistics()
print(f"Total documents: {stats['total_documents']}")
print(f"Average text length: {stats['text_length_stats']['mean']:.1f}")
print(f"Average word count: {stats['word_count_stats']['mean']:.1f}")
```

### Sample Output
```
Dataset Statistics:
  Dataset: ms_marco
  Total documents loaded: 10000
  Documents in evaluation set: 5000
  Queries generated: 100
  Avg text length: 156.7 chars
  Avg word count: 28.4 words
```

## 🔧 Integration with RAG Framework

### Use with Embedding Reproducibility Tester
```python
from src.embedding_reproducibility_tester import IntegratedRAGReproducibilityTester

tester = IntegratedRAGReproducibilityTester()

# Load dataset and run tests
documents, queries = load_dataset_for_reproducibility(
    file_path="data/ms_marco.csv",
    num_docs=5000,
    num_queries=100
)

# Run reproducibility tests
results = tester.test_integrated_reproducibility(
    documents=documents,
    queries=queries,
    output_dir="results/"
)
```

### Use with Main Test Runner
```python
# Run comprehensive tests with real dataset
python run_comprehensive_tests.py --msmarco-csv data/ms_marco.csv --n-docs 5000
```

## 🛠️ Troubleshooting

### Common Issues

1. **File Not Found Error**
   ```python
   # Check file paths
   import os
   print(os.path.exists("data/ms_marco.csv"))
   ```

2. **Encoding Issues**
   ```python
   # Specify encoding manually
   documents = loader.load_csv_data(
       csv_path="data/dataset.csv",
       encoding="latin-1"  # or "cp1252", "utf-16"
   )
   ```

3. **Column Detection Issues**
   ```python
   # Specify columns manually
   documents = loader.load_csv_data(
       csv_path="data/dataset.csv",
       text_column="content",
       id_column="id"
   )
   ```

4. **Memory Issues with Large Datasets**
   ```python
   # Limit document count
   documents = loader.load_csv_data(
       csv_path="data/large_dataset.csv",
       max_documents=5000  # Limit memory usage
   )
   ```

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now all operations will show detailed logs
documents, queries = load_dataset_for_reproducibility(...)
```

## 📋 Example CSV Format

Your CSV files should have at minimum a text column. Here's an example structure:

```csv
id,text,category,source
1,"This is the first document text content...","research","pubmed"
2,"This is the second document with more content...","news","reuters"
3,"Another document for testing purposes...","academic","arxiv"
```

The loader will auto-detect:
- `text` as the main content column
- `id` as the document identifier
- `category` and `source` as additional metadata

## 🚀 Getting Started Checklist

1. ✅ Place your CSV file in the `data/` directory
2. ✅ Import the dataset loader: `from scripts.dataset_loader import load_dataset_for_reproducibility`
3. ✅ Load your dataset: `docs, queries = load_dataset_for_reproducibility("data/your_file.csv")`
4. ✅ Run your reproducibility tests with the loaded data
5. ✅ Check the generated evaluation dataset in `data/evaluation_dataset_*.json`

## 📞 Support

For questions or issues:
- Check the debug logs with `logging.basicConfig(level=logging.DEBUG)`
- Review the dataset statistics with `loader.get_dataset_statistics()`
- Verify your CSV structure matches the expected format
- Try different query generation strategies if queries seem low quality

---

**Note**: This loader is designed to work with the RAG Reproducibility Testing Framework. All loaded datasets are automatically formatted for compatibility with the embedding and retrieval reproducibility tests.
