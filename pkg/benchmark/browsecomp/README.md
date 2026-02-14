# browsecomp — BrowseComp-Plus Benchmark Package

This package provides a pure-Go implementation for running the [BrowseComp-Plus](https://huggingface.co/datasets/Tevatron/browsecomp-plus) benchmark, which evaluates deep-research agents on reasoning-intensive queries with a fixed retrieval corpus.

## Architecture

```
pkg/benchmark/browsecomp/
├── types.go          # Core data types: BenchmarkRun, DatasetRow, OutputBlock
├── types_test.go     # Tests for dataset loading
├── searcher.go       # EmbeddedSearcher: Bleve-backed search + document retrieval
├── searcher_test.go  # Searcher tests
├── index.go          # Bleve index building from JSONL corpus
├── index_test.go     # Index tests
├── scripts/
│   └── analyze_results.py  # Results analysis tool
└── README.md         # This file
```

## Components

### `types.go`
Defines the core data structures:
- **`DatasetRow`** — A single entry from the BrowseComp dataset (query, answer, documents, etc.)
- **`BenchmarkRun`** — Output format for a single benchmark result (matches BrowseComp-Plus evaluation format)
- **`OutputBlock`** — A single output block within a benchmark result
- **`LoadDataset(path)`** — Loads the JSONL dataset file into `[]DatasetRow`
- **`LoadDatasetFromReader(r)`** — Loads from an `io.Reader` (useful for testing)

### `searcher.go`
Implements `EmbeddedSearcher`, which provides:
- **`Search(query string) string`** — Full-text search against the Bleve index, returns top-10 results with doc IDs and snippets
- **`GetDocument(docid string) string`** — Retrieves the full text of a document by its ID
- **`Tools() map[string]any`** — Returns a tool map for injection into RLM via `rlm.WithTools()`
- **`GetStats()`** — Returns tool call counts and retrieved document IDs
- **`Reset()`** — Resets per-query statistics

### `index.go`
Handles Bleve index creation:
- **`BuildIndex(indexPath, datasetPath)`** — Creates or opens a Bleve full-text index from the dataset JSONL

## Usage

```go
import "github.com/XiaoConstantine/dspy-go/pkg/benchmark/browsecomp"

// Load dataset
rows, err := browsecomp.LoadDataset("path/to/dataset.jsonl")

// Create searcher (builds/opens Bleve index)
searcher, err := browsecomp.NewEmbeddedSearcher("browsecomp.bleve", "path/to/dataset.jsonl")
defer searcher.Close()

// Inject into RLM
rlmModule := rlm.NewFromLLM(llm,
    rlm.WithTools(searcher.Tools()),
)

// Process queries
for _, row := range rows {
    searcher.Reset()
    result, err := rlmModule.Process(ctx, map[string]any{
        "context": "",
        "query":   row.Query,
    })

    counts, docIDs := searcher.GetStats()
    // Save BenchmarkRun to JSON...
}
```

## Analysis

```bash
python3 pkg/benchmark/browsecomp/scripts/analyze_results.py \
    --input datasets/BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl \
    --runs  runs/gemini_flash_lite
```

## Known Limitations

1. **Retrieval quality**: The current Bleve-based keyword search has low recall on BrowseComp queries, which are designed as oblique riddles. The official BrowseComp-Plus benchmark uses Pyserini/Lucene BM25 or Qwen3-Embedding (semantic search) over a ~100K document corpus.
2. **Corpus mismatch**: The official benchmark searches over `Tevatron/browsecomp-plus-corpus` (100K curated docs), not the dataset file itself.
3. **Evaluation**: The official evaluation uses Qwen3-32B as a judge for fuzzy answer matching. Our analysis uses simple string containment.
