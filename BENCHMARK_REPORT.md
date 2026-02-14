# BrowseComp-Plus Benchmark Report

## Overview

This report documents the results of running the BrowseComp-Plus benchmark using dspy-go's RLM (Recursive Language Model) module with an embedded Bleve full-text search index.

**Date:** 2026-02-14  
**Dataset:** BrowseComp-Plus (830 queries)  
**Model:** `gemini-2.5-flash-lite`  
**Max Iterations:** 10  
**Retriever:** Bleve full-text search on dataset JSONL

---

## Results Summary

| Metric | Value |
|---|---|
| **Total Queries** | 830 |
| **Exact Match** | 2/830 (0.2%) |
| **Partial Match** | 66/830 (8.0%) |
| **Wrong Answer** | 661/830 (79.6%) |
| **No Answer / Failed** | 101/830 (12.2%) |
| **Effective Accuracy (exact)** | 2/729 = 0.3% |
| **Effective Accuracy w/ partial** | 68/729 = 9.3% |

## Tool Usage

| Metric | Value |
|---|---|
| **Queries using Search** | 779/830 (94%) |
| **Total Search calls** | 3,348 (4.0/query avg) |
| **Total GetDocument calls** | 695 (0.8/query avg) |
| **Accuracy WITH tools** | 2/779 = 0.3% |
| **Accuracy WITHOUT tools** | 0/51 = 0.0% |

## Bugs Fixed During Benchmark

### 1. Scanner Buffer Too Small
**File:** `pkg/benchmark/browsecomp/types.go`  
The largest lines in the JSONL dataset are ~12.6MB, but the scanner buffer was only 10MB. Increased to 20MB.

### 2. Tools Not Visible to LLM (Critical)
**File:** `pkg/modules/rlm/env_yaegi.go`  
The `GetContextInfo()` method never mentioned that Search/GetDocument tools were available. The LLM saw `"Your context is a string with 0 total characters"` but had no idea tools existed. Fixed by appending `". Available tools: Search, GetDocument"` when custom tools are registered.

### 3. Yaegi "constant definition loop" (Critical)
**File:** `pkg/modules/rlm/env_yaegi.go`  
`RegisterFunction()` used `interp.Use()` with the same `"rlm/rlm"` package path as the builtins, causing a Yaegi interpreter conflict. Every `Search()` call failed with `"1:28: constant definition loop"`. Fixed by using a separate package path `"rlmtools/rlmtools"` for custom tools.

## Root Cause Analysis: Low Accuracy

The 0.2% accuracy is expected given three fundamental mismatches with the official BrowseComp-Plus setup:

### 1. Wrong Corpus (Critical)
The official benchmark searches over `Tevatron/browsecomp-plus-corpus`, a **~100K curated document collection** that contains the actual source material with answers. Our implementation only indexes the 830 query rows from the dataset JSONL file — the model is searching through questions, not through source documents with answers.

### 2. Weak Retriever
Even if we had the right corpus:
- **Official:** Pyserini/Lucene BM25 or Qwen3-Embedding (semantic search)
- **Ours:** Bleve full-text search with default settings

BrowseComp queries are deliberately written as oblique riddles ("A township established in the 1960s...") that don't share direct keywords with the answer documents. Semantic retrieval is essential.

### 3. Model Tier
- **Official leaderboard:** Uses `gemini-2.5-pro` (32.2%), `o3` (42.0%), `gpt-oss-120b` (32.2%)
- **Ours:** `gemini-2.5-flash-lite` (cheapest/smallest Gemini model)
- The official BM25 + `gemini-2.5-flash` achieves only ~15%, so even with the right corpus, Flash Lite would likely be single-digit.

### 4. Evaluation Method
- **Official:** Qwen3-32B as LLM judge for fuzzy answer matching
- **Ours:** Simple string containment (`truth in prediction.lower()`)
- This likely undercounts correct answers where the model paraphrases or provides context around the answer.

## Comparison with Paper's Results

| System | Retriever | Accuracy |
|---|---|---|
| o3 | BM25 | 42.0% |
| gpt-oss-120b | BM25 | 32.2% |
| gemini-2.5-pro | BM25 | 32.2% |
| gemini-2.5-flash | BM25 | ~15% |
| **dspy-go RLM** | **Bleve (dataset only)** | **0.2%** |

## What Would Improve Results

1. **Index the actual 100K corpus** — Download `Tevatron/browsecomp-plus-corpus` and build the search index from it
2. **Use the official BM25 index** — The BrowseComp-Plus repo provides pre-built Pyserini/Lucene BM25 indexes
3. **Use a stronger model** — `gemini-2.5-pro` or `gemini-2.5-flash` instead of Flash Lite
4. **Add semantic retrieval** — Use Qwen3-Embedding for dense retrieval instead of keyword-based Bleve
5. **Use LLM-as-judge evaluation** — Run the official evaluation script with Qwen3-32B

## Reproduction

```bash
# Run benchmark
export GEMINI_API_KEY=your_key
go run cmd/benchmark_browsecomp/main.go \
    --input datasets/BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl \
    --output runs/gemini_flash_lite \
    --model gemini-2.5-flash-lite \
    --max-iter 10

# Analyze results
python3 pkg/benchmark/browsecomp/scripts/analyze_results.py \
    --input datasets/BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl \
    --runs runs/gemini_flash_lite
```
