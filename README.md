# RAG_Reproducibility

# The source of uncertainty in RAG system
## 1. Embdedding uncertainty
### 1. different embedding model ()
### 2. different floating point precision(FP 16 vs FP 32)

## 2. Retrieval uncertainty
### 1. Index uncertainty
### 2. Retrieval algorithm uncertainty (KNN)
Faiss library sets default seed to control the reproducibility of index building and retreival results(CPU version). How about parallel version(GPU version)

