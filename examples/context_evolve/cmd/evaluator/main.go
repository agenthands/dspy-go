package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
)

// Policy represents the content we are evolving.
type Policy struct {
	Guess int `json:"guess"`
}

// Program input format matches what ContextEvolve writes.
type ProgramInput struct {
	Content string `json:"content"`
}

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: evaluator <program_file>\n")
		os.Exit(1)
	}

	inputFile := os.Args[1]
	data, err := os.ReadFile(inputFile)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to read input file: %v\n", err)
		os.Exit(1)
	}

	// Ideally, the content is the raw string from Program.Content.
    // However, CommandEvaluator currently handles it by writing Program.Content to a file.
    // If Program.Content is just "{"guess": 50}", then the file contains just that string.
    // Wait, let's check CommandEvaluator implementation.
    // Yes: tmpFile.WriteString(program.Content)
    
    // So if the program content is JSON, the file is JSON.
    var policy Policy
    if err := json.Unmarshal(data, &policy); err != nil {
        // If parsing fails, score 0.
        // Maybe the content is markdown wrapped?
        // For this example, we assume clean JSON or the evaluator handles cleanup.
        // Let's print strict failure for now.
        // Escape quotes in error message manually or just use a generic message
        fmt.Printf(`{"score": 0.0, "error": "failed to parse policy"}`+"\n")
        return
    }

	target := 77
	diff := math.Abs(float64(target - policy.Guess))
	
	// Normalize score: 1.0 if perfect, 0.0 if diff >= 100
	score := math.Max(0, 1.0 - (diff / 100.0))

	metrics := map[string]interface{}{
		"score": score,
		"diff":  diff,
        "target": target,
	}

	output, _ := json.Marshal(metrics)
	fmt.Println(string(output))
}
