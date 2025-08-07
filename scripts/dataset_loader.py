"""
Generic Dataset Loader for RAG Reproducibility Testing
Handles loading and preprocessing of various datasets for embedding reproducibility experiments
Supports CSV, JSON, and text files with flexible column mapping
"""

import os
import csv
import json
import random
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
import logging
import re
from pathlib import Path
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseDatasetLoader(ABC):
    """Abstract base class for dataset loaders"""

    @abstractmethod
    def load_data(self, file_path: str, **kwargs) -> List[Dict[str, Any]]:
        """Load data from file"""
        pass

    @abstractmethod
    def get_dataset_name(self) -> str:
        """Get the name of the dataset"""
        pass


class GenericDatasetLoader(BaseDatasetLoader):
    """Generic loader for various dataset formats (CSV, JSON, text files)"""

    def __init__(self, data_dir: str = "data", dataset_name: str = "generic"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.dataset_name = dataset_name
        self.documents = []
        self.queries = []
        self.metadata = {}

    def get_dataset_name(self) -> str:
        return self.dataset_name

    def _convert_numpy_types(self, obj: Any) -> Any:
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj

    def load_data(self, file_path: str, **kwargs) -> List[Dict[str, Any]]:
        """Load data from various file formats"""
        file_path = Path(file_path)

        if file_path.suffix.lower() == '.csv':
            return self.load_csv_data(str(file_path), **kwargs)
        elif file_path.suffix.lower() == '.json':
            return self.load_json_data(str(file_path), **kwargs)
        elif file_path.suffix.lower() in ['.txt', '.tsv']:
            return self.load_text_data(str(file_path), **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def load_csv_data(self, csv_path: str,
                     text_column: Optional[str] = None,
                     id_column: Optional[str] = None,
                     additional_columns: Optional[List[str]] = None,
                     max_documents: int = 10000,
                     encoding: str = 'auto') -> List[Dict[str, Any]]:
        """Load dataset from CSV file with flexible column mapping"""

        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        logger.info(f"Loading {self.dataset_name} data from: {csv_path}")

        documents = []

        try:
            # Handle encoding
            if encoding == 'auto':
                encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'utf-16']
            else:
                encodings_to_try = [encoding]

            df = None
            for enc in encodings_to_try:
                try:
                    df = pd.read_csv(csv_path, encoding=enc, nrows=max_documents)
                    logger.info(f"Successfully loaded CSV with encoding: {enc}")
                    break
                except UnicodeDecodeError:
                    continue

            if df is None:
                raise ValueError("Could not decode CSV file with any encoding")

            logger.info(f"Loaded CSV with {len(df)} rows and columns: {list(df.columns)}")

            # Auto-detect text column if not specified
            if text_column is None or text_column not in df.columns:
                text_column = self._auto_detect_text_column(df)
                logger.info(f"Auto-detected text column: {text_column}")

            # Auto-detect ID column if not specified
            if id_column is None or id_column not in df.columns:
                id_column = self._auto_detect_id_column(df)
                if id_column:
                    logger.info(f"Auto-detected ID column: {id_column}")

            # Process documents
            for idx, row in df.iterrows():
                text = str(row[text_column]).strip()

                # Skip empty or very short texts
                if len(text) < 20:  # Reduced minimum length for flexibility
                    continue

                # Clean text
                text = self._clean_text(text)

                # Generate document ID
                if id_column and pd.notna(row[id_column]):
                    doc_id = str(row[id_column])
                else:
                    doc_id = f"{self.dataset_name}_{idx:06d}"

                # Create document entry
                doc = {
                    "id": doc_id,
                    "text": text,
                    "metadata": {
                        "original_index": self._convert_numpy_types(idx),
                        "text_length": len(text),
                        "word_count": len(text.split()),
                        "source": self.dataset_name
                    }
                }

                # Add specified additional columns
                if additional_columns:
                    for col in additional_columns:
                        if col in df.columns and pd.notna(row[col]):
                            doc["metadata"][col] = self._convert_numpy_types(row[col])
                else:
                    # Add all other columns as metadata
                    for col in df.columns:
                        if col not in [text_column, id_column] and pd.notna(row[col]):
                            doc["metadata"][col] = self._convert_numpy_types(row[col])

                documents.append(doc)

        except Exception as e:
            logger.error(f"Error loading CSV data: {e}")
            raise

        self.documents = documents
        logger.info(f"Successfully loaded {len(documents)} documents from {self.dataset_name}")

        # Save processed data
        self._save_processed_data()

        return documents

    def load_json_data(self, json_path: str,
                      text_field: str = "text",
                      id_field: str = "id",
                      max_documents: int = 10000) -> List[Dict[str, Any]]:
        """Load dataset from JSON file"""

        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        logger.info(f"Loading {self.dataset_name} data from JSON: {json_path}")

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Handle different JSON structures
            if isinstance(data, list):
                raw_documents = data[:max_documents]
            elif isinstance(data, dict):
                if 'documents' in data:
                    raw_documents = data['documents'][:max_documents]
                elif 'data' in data:
                    raw_documents = data['data'][:max_documents]
                else:
                    # Assume the dict values are the documents
                    raw_documents = list(data.values())[:max_documents]
            else:
                raise ValueError("Unsupported JSON structure")

            documents = []
            for idx, item in enumerate(raw_documents):
                if not isinstance(item, dict):
                    continue

                # Extract text
                text = ""
                if text_field in item:
                    text = str(item[text_field])
                else:
                    # Try common text field names
                    for field in ['text', 'content', 'passage', 'paragraph', 'body']:
                        if field in item:
                            text = str(item[field])
                            break

                if len(text.strip()) < 20:
                    continue

                text = self._clean_text(text)

                # Extract ID
                doc_id = item.get(id_field, f"{self.dataset_name}_{idx:06d}")

                # Create document
                doc = {
                    "id": str(doc_id),
                    "text": text,
                    "metadata": {
                        "original_index": self._convert_numpy_types(idx),
                        "text_length": len(text),
                        "word_count": len(text.split()),
                        "source": self.dataset_name
                    }
                }

                # Add other fields as metadata
                for key, value in item.items():
                    if key not in [text_field, id_field]:
                        doc["metadata"][key] = self._convert_numpy_types(value)

                documents.append(doc)

            self.documents = documents
            logger.info(f"Successfully loaded {len(documents)} documents from JSON")

            self._save_processed_data()
            return documents

        except Exception as e:
            logger.error(f"Error loading JSON data: {e}")
            raise

    def load_text_data(self, text_path: str,
                      separator: str = "\n\n",
                      max_documents: int = 10000) -> List[Dict[str, Any]]:
        """Load dataset from plain text file"""

        text_path = Path(text_path)
        if not text_path.exists():
            raise FileNotFoundError(f"Text file not found: {text_path}")

        logger.info(f"Loading {self.dataset_name} data from text file: {text_path}")

        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Split content into documents
            raw_texts = content.split(separator)

            documents = []
            for idx, text in enumerate(raw_texts[:max_documents]):
                text = text.strip()

                if len(text) < 20:
                    continue

                text = self._clean_text(text)

                doc = {
                    "id": f"{self.dataset_name}_{idx:06d}",
                    "text": text,
                    "metadata": {
                        "original_index": self._convert_numpy_types(idx),
                        "text_length": len(text),
                        "word_count": len(text.split()),
                        "source": self.dataset_name
                    }
                }

                documents.append(doc)

            self.documents = documents
            logger.info(f"Successfully loaded {len(documents)} documents from text file")

            self._save_processed_data()
            return documents

        except Exception as e:
            logger.error(f"Error loading text data: {e}")
            raise

    def _auto_detect_text_column(self, df: pd.DataFrame) -> str:
        """Auto-detect the main text column in a DataFrame"""

        # Try common text column names first
        text_candidates = ['text', 'passage', 'paragraph', 'content', 'body',
                          'document', 'article', 'description', 'summary']

        for candidate in text_candidates:
            if candidate in df.columns:
                return candidate

        # Find the column with the longest average text length
        string_columns = df.select_dtypes(include=['object']).columns
        if len(string_columns) == 0:
            raise ValueError("No text columns found in the dataset")

        best_column = string_columns[0]
        max_avg_length = 0

        for col in string_columns:
            try:
                avg_length = df[col].astype(str).str.len().mean()
                if avg_length > max_avg_length:
                    max_avg_length = avg_length
                    best_column = col
            except:
                continue

        return best_column

    def _auto_detect_id_column(self, df: pd.DataFrame) -> Optional[str]:
        """Auto-detect the ID column in a DataFrame"""

        id_candidates = ['id', 'doc_id', 'document_id', 'passage_id', 'idx', 'index']

        for candidate in id_candidates:
            if candidate in df.columns:
                return candidate

        return None

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', ' ', text)

        # Remove excessive punctuation
        text = re.sub(r'[\.]{2,}', '.', text)
        text = re.sub(r'[\!\?]{2,}', '!', text)

        # Trim
        text = text.strip()

        return text

    def generate_queries_from_documents(self, num_queries: int = 100,
                                      query_strategy: str = "first_sentence",
                                      min_query_length: int = 10,
                                      max_query_length: int = 200) -> List[str]:
        """Generate queries from the loaded documents using various strategies"""

        if not self.documents:
            raise ValueError("No documents loaded. Call load_data() first.")

        logger.info(f"Generating {num_queries} queries using strategy: {query_strategy}")

        queries = []
        used_doc_indices = set()

        # Ensure we have enough documents
        max_attempts = min(num_queries * 3, len(self.documents))

        for attempt in range(max_attempts):
            if len(queries) >= num_queries:
                break

            # Select random document that hasn't been used
            doc_idx = random.randint(0, len(self.documents) - 1)
            if doc_idx in used_doc_indices:
                continue

            used_doc_indices.add(doc_idx)
            doc = self.documents[doc_idx]
            text = doc["text"]

            query = self._generate_query_from_text(text, query_strategy,
                                                 min_query_length, max_query_length)

            if query:
                queries.append(query)

        if len(queries) < num_queries:
            logger.warning(f"Could only generate {len(queries)} queries out of {num_queries} requested")

        self.queries = queries
        logger.info(f"Successfully generated {len(queries)} queries")

        return queries

    def _generate_query_from_text(self, text: str, strategy: str,
                                min_length: int, max_length: int) -> Optional[str]:
        """Generate a single query from text using specified strategy"""

        query = None

        if strategy == "first_sentence":
            # Extract first sentence
            sentences = re.split(r'[.!?]+', text)
            if sentences:
                query = sentences[0].strip()

        elif strategy == "random_sentence":
            # Extract random sentence
            sentences = [s.strip() for s in re.split(r'[.!?]+', text)
                        if len(s.strip()) > min_length]
            if sentences:
                query = random.choice(sentences)

        elif strategy == "first_words":
            # Use first N words
            words = text.split()[:15]  # First 15 words
            query = ' '.join(words)

        elif strategy == "keyword_extraction":
            # Simple keyword extraction (first few meaningful words)
            words = text.split()
            # Skip common stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
                         'to', 'for', 'of', 'with', 'by', 'this', 'that', 'is',
                         'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had'}
            meaningful_words = [w for w in words
                              if w.lower() not in stop_words and len(w) > 2][:8]
            query = ' '.join(meaningful_words)

        elif strategy == "title_style":
            # Extract title-like phrases (capitalized words)
            words = text.split()[:20]  # First 20 words
            title_words = [w for w in words if w[0].isupper() and len(w) > 2][:6]
            query = ' '.join(title_words)

        else:
            raise ValueError(f"Unknown query generation strategy: {strategy}")

        # Validate and clean query
        if query and min_length <= len(query) <= max_length:
            query = self._clean_text(query)
            return query if query else None

        return None

    def get_sample_documents(self, num_docs: int = 1000,
                           strategy: str = "random") -> List[Dict[str, Any]]:
        """Get a sample of documents for testing with various sampling strategies"""

        if not self.documents:
            raise ValueError("No documents loaded. Call load_data() first.")

        num_docs = min(num_docs, len(self.documents))

        if strategy == "random":
            sampled_docs = random.sample(self.documents, num_docs)
        elif strategy == "first":
            sampled_docs = self.documents[:num_docs]
        elif strategy == "diverse":
            # Sample documents with diverse lengths
            sorted_docs = sorted(self.documents, key=lambda x: x["metadata"]["text_length"])
            indices = np.linspace(0, len(sorted_docs) - 1, num_docs, dtype=int)
            sampled_docs = [sorted_docs[i] for i in indices]
        elif strategy == "balanced":
            # Balance between short, medium, and long documents
            sorted_docs = sorted(self.documents, key=lambda x: x["metadata"]["text_length"])
            n_total = len(sorted_docs)
            short_docs = sorted_docs[:n_total//3]
            medium_docs = sorted_docs[n_total//3:2*n_total//3]
            long_docs = sorted_docs[2*n_total//3:]

            n_per_group = num_docs // 3
            sampled_docs = (
                random.sample(short_docs, min(n_per_group, len(short_docs))) +
                random.sample(medium_docs, min(n_per_group, len(medium_docs))) +
                random.sample(long_docs, min(num_docs - 2*n_per_group, len(long_docs)))
            )
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")

        logger.info(f"Sampled {len(sampled_docs)} documents using strategy: {strategy}")
        return sampled_docs

    def _save_processed_data(self):
        """Save processed data for future use"""

        # Save documents with dataset-specific filename
        docs_path = self.data_dir / f"processed_documents_{self.dataset_name}.json"
        with open(docs_path, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, indent=2, ensure_ascii=False)

        # Save metadata
        if self.documents:
            metadata = {
                "dataset_name": self.dataset_name,
                "num_documents": len(self.documents),
                "avg_text_length": np.mean([doc["metadata"]["text_length"] for doc in self.documents]),
                "avg_word_count": np.mean([doc["metadata"]["word_count"] for doc in self.documents]),
                "min_text_length": min([doc["metadata"]["text_length"] for doc in self.documents]),
                "max_text_length": max([doc["metadata"]["text_length"] for doc in self.documents]),
            }
        else:
            metadata = {"dataset_name": self.dataset_name, "num_documents": 0}

        metadata_path = self.data_dir / f"dataset_metadata_{self.dataset_name}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved processed data to {docs_path} and metadata to {metadata_path}")

    def load_processed_data(self) -> bool:
        """Load previously processed data if available"""

        docs_path = self.data_dir / f"processed_documents_{self.dataset_name}.json"

        if docs_path.exists():
            try:
                with open(docs_path, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)

                logger.info(f"Loaded {len(self.documents)} previously processed documents")
                return True

            except Exception as e:
                logger.error(f"Error loading processed data: {e}")
                return False

        return False

    def create_evaluation_dataset(self, num_docs: int = 5000, num_queries: int = 100,
                                doc_sampling: str = "diverse",
                                query_strategy: str = "first_sentence") -> Tuple[List[Dict[str, Any]], List[str]]:
        """Create a complete evaluation dataset from loaded data"""

        logger.info(f"Creating evaluation dataset from {self.dataset_name}")

        # Get sample documents
        documents = self.get_sample_documents(num_docs, doc_sampling)

        # Generate queries
        queries = self.generate_queries_from_documents(num_queries, query_strategy)

        # Save evaluation dataset
        eval_data = {
            "documents": documents,
            "queries": queries,
            "config": {
                "dataset_name": self.dataset_name,
                "num_docs": len(documents),
                "num_queries": len(queries),
                "doc_sampling": doc_sampling,
                "query_strategy": query_strategy
            },
            "metadata": self.get_dataset_statistics()
        }

        eval_path = self.data_dir / f"evaluation_dataset_{self.dataset_name}.json"
        with open(eval_path, 'w', encoding='utf-8') as f:
            json.dump(eval_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved evaluation dataset to {eval_path}")
        logger.info(f"Dataset: {len(documents)} documents, {len(queries)} queries")

        return documents, queries

    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get statistics about the loaded dataset"""

        if not self.documents:
            return {}

        text_lengths = [doc["metadata"]["text_length"] for doc in self.documents]
        word_counts = [doc["metadata"]["word_count"] for doc in self.documents]

        stats = {
            "total_documents": len(self.documents),
            "total_queries": len(self.queries),
            "text_length_stats": {
                "mean": np.mean(text_lengths),
                "std": np.std(text_lengths),
                "min": np.min(text_lengths),
                "max": np.max(text_lengths),
                "median": np.median(text_lengths)
            },
            "word_count_stats": {
                "mean": np.mean(word_counts),
                "std": np.std(word_counts),
                "min": np.min(word_counts),
                "max": np.max(word_counts),
                "median": np.median(word_counts)
            }
        }

        return self._convert_numpy_types(stats)


class MSMARCOLoader(GenericDatasetLoader):
    """Specialized loader for MS MARCO dataset"""

    def __init__(self, data_dir: str = "data"):
        super().__init__(data_dir, "ms_marco")

    def load_msmarco_csv(self, csv_path: str, max_documents: int = 10000) -> List[Dict[str, Any]]:
        """Load MS MARCO from CSV with default settings"""
        return self.load_csv_data(
            csv_path=csv_path,
            text_column=None,  # Auto-detect
            max_documents=max_documents
        )


class CommonCrawlLoader(GenericDatasetLoader):
    """Specialized loader for Common Crawl datasets"""

    def __init__(self, data_dir: str = "data"):
        super().__init__(data_dir, "common_crawl")


class WikipediaLoader(GenericDatasetLoader):
    """Specialized loader for Wikipedia datasets"""

    def __init__(self, data_dir: str = "data"):
        super().__init__(data_dir, "wikipedia")

    def load_wikipedia_json(self, json_path: str, max_documents: int = 10000) -> List[Dict[str, Any]]:
        """Load Wikipedia from JSON with typical field names"""
        return self.load_json_data(
            json_path=json_path,
            text_field="text",  # Common for Wikipedia dumps
            id_field="id",
            max_documents=max_documents
        )


class CustomDatasetLoader(GenericDatasetLoader):
    """Loader for custom datasets with user-defined parameters"""

    def __init__(self, dataset_name: str, data_dir: str = "data"):
        super().__init__(data_dir, dataset_name)


def load_dataset_for_reproducibility(file_path: str,
                                   dataset_type: str = "auto",
                                   num_docs: int = 5000,
                                   num_queries: int = 100,
                                   data_dir: str = "data",
                                   **loader_kwargs) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Universal function to load any dataset for reproducibility testing"""

    file_path = Path(file_path)

    # Auto-detect dataset type if not specified
    if dataset_type == "auto":
        if "marco" in file_path.name.lower():
            dataset_type = "ms_marco"
        elif "wiki" in file_path.name.lower():
            dataset_type = "wikipedia"
        elif "crawl" in file_path.name.lower():
            dataset_type = "common_crawl"
        else:
            dataset_type = "custom"

    # Create appropriate loader
    if dataset_type == "ms_marco":
        loader = MSMARCOLoader(data_dir)
    elif dataset_type == "wikipedia":
        loader = WikipediaLoader(data_dir)
    elif dataset_type == "common_crawl":
        loader = CommonCrawlLoader(data_dir)
    else:
        # Use custom or generic loader
        dataset_name = loader_kwargs.get('dataset_name', file_path.stem)
        loader = CustomDatasetLoader(dataset_name, data_dir)

    logger.info(f"Loading dataset: {loader.get_dataset_name()}")

    # Try to load processed data first
    processed_file = loader.data_dir / f"processed_documents_{loader.dataset_name}.json"
    if processed_file.exists() and not loader_kwargs.get('force_reload', False):
        logger.info("Loading previously processed data...")
        try:
            with open(processed_file, 'r', encoding='utf-8') as f:
                loader.documents = json.load(f)
            logger.info(f"Loaded {len(loader.documents)} previously processed documents")
        except Exception as e:
            logger.warning(f"Could not load processed data: {e}. Loading from source.")
            loader.load_data(str(file_path), **loader_kwargs)
    else:
        # Load from source file
        loader.load_data(str(file_path), **loader_kwargs)

    # Create evaluation dataset
    documents, queries = loader.create_evaluation_dataset(
        num_docs=num_docs,
        num_queries=num_queries,
        doc_sampling=loader_kwargs.get('doc_sampling', 'diverse'),
        query_strategy=loader_kwargs.get('query_strategy', 'first_sentence')
    )

    # Print statistics
    stats = loader.get_dataset_statistics()
    logger.info("Dataset Statistics:")
    logger.info(f"  Dataset: {loader.get_dataset_name()}")
    logger.info(f"  Total documents loaded: {stats['total_documents']}")
    logger.info(f"  Documents in evaluation set: {len(documents)}")
    logger.info(f"  Queries generated: {stats['total_queries']}")
    logger.info(f"  Avg text length: {stats['text_length_stats']['mean']:.1f} chars")
    logger.info(f"  Avg word count: {stats['word_count_stats']['mean']:.1f} words")

    # Final numpy type conversion to ensure JSON serialization compatibility
    clean_documents = []
    for doc in documents:
        clean_doc = {
            "id": doc["id"],
            "text": doc["text"],
            "metadata": loader._convert_numpy_types(doc["metadata"]) if hasattr(loader, '_convert_numpy_types') else doc["metadata"]
        }
        clean_documents.append(clean_doc)

    return clean_documents, queries


# Backward compatibility function
def load_msmarco_for_reproducibility(csv_path: str,
                                   num_docs: int = 5000,
                                   num_queries: int = 100,
                                   data_dir: str = "data") -> Tuple[List[Dict[str, Any]], List[str]]:
    """Backward compatibility function for MS MARCO loading"""

    return load_dataset_for_reproducibility(
        file_path=csv_path,
        dataset_type="ms_marco",
        num_docs=num_docs,
        num_queries=num_queries,
        data_dir=data_dir
    )


if __name__ == "__main__":
    # Example usage for different datasets
    logging.basicConfig(level=logging.INFO)

    # Example 1: Load MS MARCO data
    print("=== Example 1: MS MARCO Dataset ===")
    try:
        documents, queries = load_dataset_for_reproducibility(
            file_path="data/ms_marco_passages.csv",
            dataset_type="ms_marco",
            num_docs=2000,
            num_queries=50
        )

        print(f"\nSample MS MARCO document:")
        print(f"ID: {documents[0]['id']}")
        print(f"Text: {documents[0]['text'][:200]}...")
        print(f"Metadata: {documents[0]['metadata']}")

        print(f"\nSample MS MARCO queries:")
        for i, query in enumerate(queries[:3]):
            print(f"{i+1}: {query}")

    except FileNotFoundError:
        print("MS MARCO CSV file not found. Place your CSV file in the data/ directory.")

    # Example 2: Load custom JSON dataset
    print("\n=== Example 2: Custom JSON Dataset ===")
    try:
        # This would work with any JSON file containing documents
        custom_loader = CustomDatasetLoader("my_custom_dataset")
        # custom_loader.load_json_data("path/to/custom.json")
        print("Custom loader created successfully")
    except Exception as e:
        print(f"Custom loader example: {e}")

    # Example 3: Load Wikipedia dataset
    print("\n=== Example 3: Wikipedia Dataset ===")
    try:
        wiki_loader = WikipediaLoader()
        # wiki_loader.load_wikipedia_json("path/to/wikipedia.json")
        print("Wikipedia loader created successfully")
    except Exception as e:
        print(f"Wikipedia loader example: {e}")

    print("\n=== Available Query Generation Strategies ===")
    print("- first_sentence: Extract the first sentence of each document")
    print("- random_sentence: Extract a random sentence from each document")
    print("- first_words: Use the first 15 words of each document")
    print("- keyword_extraction: Extract meaningful keywords")
    print("- title_style: Extract title-like capitalized phrases")

    print("\n=== Available Document Sampling Strategies ===")
    print("- random: Random sampling of documents")
    print("- first: Take the first N documents")
    print("- diverse: Sample documents with diverse text lengths")
    print("- balanced: Balance between short, medium, and long documents")

    print("\n=== Supported File Formats ===")
    print("- CSV: Flexible column mapping with auto-detection")
    print("- JSON: Various JSON structures supported")
    print("- TXT: Plain text files with customizable separators")
    print("- TSV: Tab-separated values")

    print("\n=== Usage Examples ===")
    print("""
    # Load any dataset automatically
    docs, queries = load_dataset_for_reproducibility(
        file_path="path/to/dataset.csv",
        dataset_type="auto",  # Auto-detect
        num_docs=5000,
        num_queries=100
    )

    # Load with specific parameters
    docs, queries = load_dataset_for_reproducibility(
        file_path="path/to/custom.json",
        dataset_type="custom",
        dataset_name="my_dataset",
        num_docs=3000,
        query_strategy="keyword_extraction",
        doc_sampling="balanced"
    )

    # Use specialized loaders
    ms_marco_loader = MSMARCOLoader()
    docs = ms_marco_loader.load_msmarco_csv("data/passages.csv")

    wiki_loader = WikipediaLoader()
    docs = wiki_loader.load_wikipedia_json("data/wiki.json")
    """)
