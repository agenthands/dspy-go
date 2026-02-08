// Package context_evolve implements the ContextEvolve algorithm for optimizing content using LLM agents.
package context_evolve

import (
	"context"
	"fmt"
    "math/rand"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
    "github.com/XiaoConstantine/dspy-go/pkg/modules"
)

// --- Summarizer ---

// SummarizerAgent is responsible for summarizing the content of a program.
// It distills the core logic effectively for the Navigator to understand the history.
type SummarizerAgent struct {
	predictor *modules.Predict
}

// NewSummarizerAgent creates a new SummarizerAgent.
func NewSummarizerAgent() *SummarizerAgent {
	// Define signature: Content -> Abstract
	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.Field{Name: "content", Description: "The raw code or policy content to summarize"}},
            {Field: core.Field{Name: "parent_abstract", Description: "Abstract of the parent program (if available)"}},
		},
		[]core.OutputField{
			{Field: core.Field{Name: "abstract", Description: "A concise summary of the strategy or logic"}},
		},
	).WithInstruction(`You are an expert code analyst specializing in algorithmic optimization and code comprehension.
Your task is to summarize the core algorithmic idea from code implementations, focusing on the underlying strategy, data structures, and optimization techniques used in brief and concise manner.

# Requirements:
1. Concise but dense with technical details.
2. Highlight the new or modified aspects (2-4 phrases).
3. Keep core shared characteristics from the parent (if available) that are still present in the child (2-4 phrases).
4. Each phrase should be less than 8 words.

Output the summary of the child program directly:
- phrase 1
- phrase 2
- ...`)

	return &SummarizerAgent{
		predictor: modules.NewPredict(signature),
	}
}

// Summarize generates an abstract for the given program.
func (s *SummarizerAgent) Summarize(ctx context.Context, p *Program, parentAbstract string) (string, error) {
	result, err := s.predictor.Process(ctx, map[string]interface{}{
		"content":         p.Content,
        "parent_abstract": parentAbstract,
	})
	if err != nil {
		return "", err
	}
	// modules.Predict returns generic result, cast as string safely?
    // Actually Predict.Process returns map[string]interface{}. 
	return fmt.Sprintf("%v", result["abstract"]), nil
}

// --- Navigator ---

type NavigatorAgent struct {
	predictor *modules.Predict
}

// NewNavigatorAgent creates a new NavigatorAgent.
func NewNavigatorAgent() *NavigatorAgent {
	// Define signature: History -> Gradient
	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.Field{Name: "history", Description: "History of attempts (abstracts + metrics) to analyze"}},
		},
		[]core.OutputField{
			{Field: core.Field{Name: "gradient", Description: "A text description of the optimization direction (what to do next)"}},
		},
	).WithInstruction(`You are an expert algorithmic strategist (Meta-Critic).
Your task is to analyze the recent history of code evolution (both successes and failures) and synthesize 3-5 brief high-level **Optimization Directions** for the next batch of optimization attempts.

# Requirements:
1. **Identify Trends:** Identify which directions seem promising vs. dead ends.
2. **Potential Ideas:** Though some attempts may failed, the high-level ideas may still be worth exploring.
3. **Abstraction:** Do NOT suggest specific code lines. Suggest algorithmic paradigms or architectural patterns.
4. **Focus:** The goal is to maximize the evaluation metrics.

Output the directions directly.`)

	return &NavigatorAgent{
		predictor: modules.NewPredict(signature),
	}
}

func (n *NavigatorAgent) GetGradient(ctx context.Context, history []*Program) (string, error) {
	// 1. Calculate stats for the history batch
	stats := CalculatePopulationStats(history)
	
	// 2. Normalize metrics
	normalized := NormalizeProgramMetrics(history, stats)

	// Format history with normalized metrics
	var sb strings.Builder
	for i, p := range history {
		// Create a string representation of normalized metrics
		metricsStr := ""
		if norm, ok := normalized[p.ID]; ok {
			for k, v := range norm {
				metricsStr += fmt.Sprintf("%s: %.2f (raw: %v), ", k, v, p.Metrics[k])
			}
		}
		
		sb.WriteString(fmt.Sprintf("Attempt %d:\nAbstract: %s\nMetrics (Z-Scores): {%s}\n\n", i+1, p.Abstract, metricsStr))
	}

	result, err := n.predictor.Process(ctx, map[string]interface{}{
		"history": sb.String(),
	})
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("%v", result["gradient"]), nil
}

// --- Sampler ---

type SamplerAgent struct {
	predictor *modules.Predict
    config    Config
}

// NewSamplerAgent creates a new SamplerAgent.
func NewSamplerAgent(config Config) *SamplerAgent {
	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.Field{Name: "parent_abstract", Description: "Abstract of the program being evolved"}},
			{Field: core.Field{Name: "optimization_gradient", Description: "Directions for improvement"}},
			{Field: core.Field{Name: "candidates", Description: "List of candidate programs (ID: Abstract, Metrics)"}},
		},
		[]core.OutputField{
			{Field: core.Field{Name: "selected_indices", Description: "JSON list of indices of selected candidates (e.g., [1, 3, 5])"}},
            {Field: core.Field{Name: "reasoning", Description: "Brief reasoning for selection"}},
		},
	).WithInstruction(`You are an expert program selector specializing in evolutionary algorithm optimization.
Your task is to select the best reference programs from a population to guide the evolution of a parent program based on its current state and improvement needs.

# Selection Criteria:
1. The candidate should logically address or relate to the current optimization directions.
2. Prefer candidates with better evaluation metrics.
3. Avoid selecting multiple programs with identical or similar abstracts. Programs should complement each other.
4. Provide potential, attractive, and innovative ideas.

Output a JSON object with "reasoning" and "selected_indices".`)

	return &SamplerAgent{
		predictor: modules.NewPredict(signature),
        config: config,
	}
}

// SampleFromPopulation selects distinct programs from the population using weighted sampling based on normalized metrics.
func (s *SamplerAgent) SampleFromPopulation(database *ProgramDatabase, batchSize int) []*Program {
	allPrograms := database.GetAllPrograms()
	if len(allPrograms) == 0 {
		return nil
	}
	if batchSize >= len(allPrograms) {
		return allPrograms
	}

	// 1. Calculate Stats and Normalization
	stats := CalculatePopulationStats(allPrograms)
	normalizedMetrics := NormalizeProgramMetrics(allPrograms, stats)
	
	// 2. Classify Programs
	improved, mixed, degraded := ClassifyPrograms(allPrograms, normalizedMetrics)

	// Calculate target counts based on weights
    totalWeight := s.config.Evolution.RolloutWeightAllImproved + s.config.Evolution.RolloutWeightMixed + s.config.Evolution.RolloutWeightAllDegraded
    if totalWeight == 0 {
        totalWeight = 1
    }
    
	targetImproved := int(float64(batchSize) * s.config.Evolution.RolloutWeightAllImproved / totalWeight)
	targetMixed := int(float64(batchSize) * s.config.Evolution.RolloutWeightMixed / totalWeight)
	targetDegraded := int(float64(batchSize) * s.config.Evolution.RolloutWeightAllDegraded / totalWeight)

    // Adjust for rounding errors
    currentTotal := targetImproved + targetMixed + targetDegraded
    if currentTotal < batchSize {
        targetImproved += (batchSize - currentTotal)
    }

	sampled := make([]*Program, 0, batchSize)
    sampledIndices := make(map[string]bool)

	// Helper to sample from a slice
	sampleFromSlice := func(source []*Program, count int) {
        if len(source) == 0 {
            return
        }
        
        // Create a copy to shuffle
        shuffled := make([]*Program, len(source))
        copy(shuffled, source)
        rand.Shuffle(len(shuffled), func(i, j int) { shuffled[i], shuffled[j] = shuffled[j], shuffled[i] })
        
        added := 0
        for _, p := range shuffled {
            if added >= count {
                break
            }
            if !sampledIndices[p.ID] {
                sampled = append(sampled, p)
                sampledIndices[p.ID] = true
                added++
            }
        }
	}

	sampleFromSlice(improved, targetImproved)
	sampleFromSlice(mixed, targetMixed)
	sampleFromSlice(degraded, targetDegraded)

	// Fill remaining if needed
	if len(sampled) < batchSize {
        remaining := make([]*Program, 0)
        for _, p := range allPrograms {
            if !sampledIndices[p.ID] {
                remaining = append(remaining, p)
            }
        }
        sampleFromSlice(remaining, batchSize - len(sampled))
	}

    // Shuffle final result
    rand.Shuffle(len(sampled), func(i, j int) { sampled[i], sampled[j] = sampled[j], sampled[i] })
    
	return sampled
}

// SelectReferences uses LLM to select the best references from a pre-filtered candidate set.
func (s *SamplerAgent) SelectReferences(ctx context.Context, candidates []*Program, parent *Program, gradient string) ([]*Program, error) {
    // If few candidates, return all
    if len(candidates) <= 3 {
        return candidates, nil
    }

	var sb strings.Builder
	for i, p := range candidates {
		sb.WriteString(fmt.Sprintf("**Index %d**:\n- Abstract: %s\n- Metrics: %v\n\n", i, p.Abstract, p.Metrics))
	}

	result, err := s.predictor.Process(ctx, map[string]interface{}{
		"parent_abstract":       parent.Abstract,
		"optimization_gradient": gradient,
		"candidates":            sb.String(),
	})
	if err != nil {
		return nil, err
	}

	// Parse JSON output for indices
    indicesStr := fmt.Sprintf("%v", result["selected_indices"])
    // Handle simple JSON parsing (robustness improvement needed for complex JSON)
    indicesStr = strings.Trim(indicesStr, "[]")
    parts := strings.Split(indicesStr, ",")
    
    var selected []*Program
    for _, part := range parts {
        var idx int
        if _, err := fmt.Sscanf(strings.TrimSpace(part), "%d", &idx); err == nil && idx >= 0 && idx < len(candidates) {
            selected = append(selected, candidates[idx])
        }
    }
    
    // Fallback if parsing fails or returns nothing
    if len(selected) == 0 {
        // Return top 3 candidates as fallback
        return candidates[:3], nil
    }
    
	return selected, nil
}

// --- Evolution ---

type EvolutionAgent struct {
	predictor *modules.Predict
}

func NewEvolutionAgent() *EvolutionAgent {
	// Define signature: Parent + Gradient + References -> Child
	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.Field{Name: "parent_content", Description: "Current code or action sequence"}},
			{Field: core.Field{Name: "optimization_gradient", Description: "Directions for improvement"}},
			{Field: core.Field{Name: "references", Description: "Helpful reference programs (Abstracts and Content)"}},
		},
		[]core.OutputField{
			{Field: core.Field{Name: "improved_content", Description: "The improved code or action sequence"}},
		},
	).WithInstruction("Improve the given program based on the optimization directions and reference examples.")

	return &EvolutionAgent{
		predictor: modules.NewPredict(signature),
	}
}

func (e *EvolutionAgent) Evolve(ctx context.Context, parent *Program, references []*Program, gradient string) (string, error) {
	// Format references
	var sb strings.Builder
	for _, p := range references {
		sb.WriteString(fmt.Sprintf("Reference:\nAbstract: %s\nContent: %s\n\n", p.Abstract, p.Content))
	}

	result, err := e.predictor.Process(ctx, map[string]interface{}{
		"parent_content":        parent.Content,
		"optimization_gradient": gradient,
		"references":            sb.String(),
	})
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("%v", result["improved_content"]), nil
}
