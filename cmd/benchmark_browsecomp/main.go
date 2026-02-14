package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/benchmark/browsecomp"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/llms"
	"github.com/XiaoConstantine/dspy-go/pkg/modules/rlm"
)

func main() {
	var (
		inputFile = flag.String("input", "datasets/BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl", "Path to dataset JSONL")
		outputDir = flag.String("output", "runs/go_rlm", "Output directory")
		indexPath = flag.String("index", "browsecomp.bleve", "Path to Bleve index")
		modelName = flag.String("model", "gemini-2.0-flash", "LLM model name")
		maxIter   = flag.Int("max-iter", 10, "Max iterations")
		_         = flag.String("mcp-url", "", "Ignored (using embedded searcher)")
	)
	flag.Parse()

	if err := os.MkdirAll(*outputDir, 0755); err != nil {
		log.Fatalf("Failed to create output dir: %v", err)
	}

	// Initialize LLM
	var llm core.LLM
	var err error

	if len(*modelName) > 6 && (*modelName)[:7] == "ollama:" || *modelName == "gpt-oss:latest" || *modelName == "gpt-oss" {
		// Ollama provider
		ollamaBaseURL := os.Getenv("OLLAMA_BASE_URL")
		if ollamaBaseURL == "" {
			ollamaBaseURL = "http://localhost:11434"
		}

		log.Printf("Initializing Ollama LLM with model: %s, URL: %s", *modelName, ollamaBaseURL)

		llm, err = llms.NewOllamaLLM(
			core.ModelID(*modelName),
			llms.WithBaseURL(ollamaBaseURL),
			llms.WithOpenAIAPI(), // Use OpenAI compatible endpoint by default as it's more stable
		)
	} else {
		// Gemini provider (Default)
		apiKey := os.Getenv("GEMINI_API_KEY")
		if apiKey == "" {
			log.Fatal("GEMINI_API_KEY environment variable not set")
		}
		llms.EnsureFactory()
		llm, err = llms.NewGeminiLLM(apiKey, core.ModelID(*modelName))
	}

	if err != nil {
		log.Fatalf("Failed to create LLM: %v", err)
	}

	searcher, err := browsecomp.NewEmbeddedSearcher(*indexPath, *inputFile)
	if err != nil {
		log.Fatalf("Failed to create searcher: %v", err)
	}
	defer searcher.Close()

	// Load dataset using the package function
	rows, err := browsecomp.LoadDataset(*inputFile)
	if err != nil {
		log.Fatalf("Failed to load dataset: %v", err)
	}

	logger := log.New(os.Stdout, "", log.LstdFlags)

	for _, row := range rows {
		queryID := fmt.Sprintf("%v", row.QueryID)
		outFile := filepath.Join(*outputDir, fmt.Sprintf("run_%s.json", queryID))
		if _, err := os.Stat(outFile); err == nil {
			logger.Printf("Skipping Query ID: %s (already exists)", queryID)
			continue
		}

		logger.Printf("Processing Query ID: %s", queryID)

		searcher.Reset()
		ctx := context.Background()

		rlmModule := rlm.NewFromLLM(
			llm,
			rlm.WithMaxIterations(*maxIter),
			rlm.WithVerbose(true),
			rlm.WithTools(searcher.Tools()),
			rlm.WithTraceDir(filepath.Join(*outputDir, "traces")),
		)

		modelInputs := map[string]any{
			"context": "",
			"query":   row.Query,
		}

		start := time.Now()
		result, err := rlmModule.Process(ctx, modelInputs)
		duration := time.Since(start)

		status := "completed"
		finalAnswer := ""
		if err != nil {
			status = "failed"
			logger.Printf("Check failed for %s: %v", queryID, err)
		} else {
			finalAnswer, _ = result["answer"].(string)
		}

		counts, retrievedDocs := searcher.GetStats()

		run := browsecomp.BenchmarkRun{
			QueryID: row.QueryID,
			Status:  status,
			Result: []browsecomp.OutputBlock{
				{Type: "output_text", Output: finalAnswer},
			},
			ToolCallCounts:  counts,
			RetrievedDocIDs: retrievedDocs,
			Metadata: map[string]any{
				"model":    *modelName,
				"duration": duration.String(),
			},
		}

		outBytes, _ := json.MarshalIndent(run, "", "  ")
		if err := os.WriteFile(outFile, outBytes, 0644); err != nil {
			logger.Printf("Failed to write output for %s: %v", queryID, err)
		}
	}
}
