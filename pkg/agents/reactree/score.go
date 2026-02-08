package reactree

import (
	"context"
	"strconv"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
)

// ScoreSignature defines the input/output for the Scorer module.
type ScoreSignature struct {
	Goal      string `dspy:"goal"`
	Memory    string `dspy:"memory"`
	Candidate string `dspy:"candidate"` // The thought/action to evaluate

	Score     string `dspy:"score"`     // 0.0 to 1.0
	Reasoning string `dspy:"reasoning"` // Why this score?
}

// Scorer defines the interface for evaluating candidates.
type Scorer interface {
	Score(ctx context.Context, goal, memory, candidate string) (float64, string, error)
}

// DefaultScorer uses an LLM to score candidates.
type DefaultScorer struct {
	Module core.Module
}

// NewDefaultScorer creates a new scorer with a Predict module.
func NewDefaultScorer() *DefaultScorer {
	// Define the signature programmatically to ensure correctness
	sig := core.NewSignature(
		[]core.InputField{
			{Field: core.NewTextField("goal")},
			{Field: core.NewTextField("memory")},
			{Field: core.NewTextField("candidate")},
		},
		[]core.OutputField{
			{Field: core.NewTextField("score")},
			{Field: core.NewTextField("reasoning")},
		},
	).WithInstruction("Evaluate the candidate step for achieving the goal. Output a score between 0.0 (useless/bad) and 1.0 (perfect/solution). Provide reasoning.")

	return &DefaultScorer{
		Module: modules.NewPredict(sig),
	}
}

// Score evaluates a candidate.
func (s *DefaultScorer) Score(ctx context.Context, goal, memory, candidate string) (float64, string, error) {
	inputs := map[string]interface{}{
		"goal":      goal,
		"memory":    memory,
		"candidate": candidate,
	}

	res, err := s.Module.Process(ctx, inputs)
	if err != nil {
		return 0, "", err
	}

	scoreStr, _ := res["score"].(string)
	reasoning, _ := res["reasoning"].(string)

	// Clean up score string (handle potential extra text like "Score: 0.8")
	scoreStr = strings.TrimSpace(scoreStr)
	// Simple heuristic to extract float
	// In a real robust system, we might use a typed output or regex
	
	score, err := strconv.ParseFloat(scoreStr, 64)
	if err != nil {
		// Try to find a float in the string if direct parse fails
		// For now, default to 0.5 if parsing fails but log it
		return 0.5, reasoning, nil 
	}

	return score, reasoning, nil
}
