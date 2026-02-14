# benchmark_browsecomp

CLI tool for running the BrowseComp-Plus benchmark with dspy-go's RLM module.

## Quick Start

```bash
# Gemini (default)
export GEMINI_API_KEY=your_key
go run cmd/benchmark_browsecomp/main.go \
    --input datasets/BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl \
    --output runs/gemini_flash_lite \
    --model gemini-2.5-flash-lite \
    --max-iter 10

# Ollama
go run cmd/benchmark_browsecomp/main.go \
    --model ollama:qwen3:8b \
    --max-iter 5
```

## Flags

| Flag | Default | Description |
|---|---|---|
| `--input` | `datasets/BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl` | Dataset JSONL path |
| `--output` | `runs/go_rlm` | Output directory for result JSONs |
| `--index` | `browsecomp.bleve` | Bleve index path |
| `--model` | `gemini-2.0-flash` | Model name |
| `--max-iter` | `10` | Max RLM iterations per query |

## Output Format

Each query produces a `run_{query_id}.json` file compatible with BrowseComp-Plus evaluation:

```json
{
    "query_id": "769",
    "status": "completed",
    "result": [{"type": "output_text", "output": "..."}],
    "tool_call_counts": {"Search": 3, "GetDocument": 1},
    "retrieved_docids": ["doc_123", "doc_456"],
    "metadata": {"model": "gemini-2.5-flash-lite", "duration": "12.5s"}
}
```

Existing result files are automatically skipped for resumability.

## Analysis

```bash
python3 pkg/benchmark/browsecomp/scripts/analyze_results.py \
    --input datasets/BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl \
    --runs runs/gemini_flash_lite
```

## Core Package

The reusable benchmark logic lives in [`pkg/benchmark/browsecomp`](../../pkg/benchmark/browsecomp/README.md).
