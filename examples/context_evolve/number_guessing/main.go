package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/agents/context_evolve"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	_ "github.com/XiaoConstantine/dspy-go/pkg/llms"
)

// Target number to guess
const TargetNumber = 42

// NumberGuessingEvaluator evaluates how close the guess is to the target.
type NumberGuessingEvaluator struct{}

func (e *NumberGuessingEvaluator) Evaluate(ctx context.Context, program *context_evolve.Program) (map[string]interface{}, error) {
	// Parse content as JSON
	var data map[string]interface{}
	// Clean markdown if present
	content := strings.TrimPrefix(program.Content, "```json")
	content = strings.TrimSuffix(content, "```")
	content = strings.TrimSpace(content)
	
	if err := json.Unmarshal([]byte(content), &data); err != nil {
		// Penalize invalid JSON
		return map[string]interface{}{"score": -100.0, "error": "invalid json"}, nil
	}

	guessVal, ok := data["guess"]
	if !ok {
		return map[string]interface{}{"score": -50.0, "error": "missing guess field"}, nil
	}

	var guess float64
	switch v := guessVal.(type) {
	case float64:
		guess = v
	case string:
		if val, err := strconv.ParseFloat(v, 64); err == nil {
			guess = val
		}
	}

	diff := math.Abs(float64(TargetNumber) - guess)
	score := 100.0 - diff

	return map[string]interface{}{
		"score": score,
		"diff":  diff,
		"guess": guess,
	}, nil
}

func main() {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		fmt.Println("Please set OPENAI_API_KEY environment variable to run this example.")
        // For demonstration purposes, we'll exit, but in a real scenario you'd handle this gracefully.
		return
	}

	ctx := context.Background()

	// 1. Configure LLM
	if err := core.ConfigureDefaultLLM(apiKey, core.ModelOpenAIGPT4oMini); err != nil {
		log.Fatalf("Failed to configure LLM: %v", err)
	}

	// 2. Configure Evolution
	config := context_evolve.Config{
		Evolution: context_evolve.EvolutionConfig{
			MaxIterations:            5,   // Run for 5 epochs
			NumIslands:               2,   // 2 Parallel threads
			MigrationInterval:        2,   // Migrate every 2 epochs
			TargetScore:              100, // Perfect score
			RolloutWeightAllImproved: 0.6,
		},
        LLM: context_evolve.LLMConfig{
            MaxTokens: 1000,
        },
	}

	// 3. Create Controller
	controller := context_evolve.NewController(config)
	
	// 4. Override Evaluator with our custom logic
	controller.Evaluator = &NumberGuessingEvaluator{}

	// 5. Initial "Program" (Instructions/Policy)
	// We start with a bad guess.
	initialContent := `{
    "guess": 10,
    "strategy": "Start low and see what happens."
}`

	log.Printf("Starting evolution to guess target number: %d", TargetNumber)
	
	// 6. Run Evolution
	bestProgram, err := controller.Run(ctx, initialContent)
	if err != nil {
		log.Fatalf("Evolution failed: %v", err)
	}

	fmt.Println("\n--- Evolution Complete ---")
	fmt.Printf("Best Program ID: %s\n", bestProgram.ID)
	fmt.Printf("Content: %s\n", bestProgram.Content)
	fmt.Printf("Metrics: %v\n", bestProgram.Metrics)
}
