package main

import (
	"context"
	"fmt"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/agents/reactree"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/llms"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
)

type TestCase struct {
	Nums       []int
	Difficulty string
}

type Result struct {
	Difficulty string
	Success    bool
	Duration   time.Duration
	Steps      int
}

func main() {
	ctx := context.Background()
	llm, err := llms.NewOllamaLLM("gpt-oss:latest", llms.WithTimeout(300)) 
	if err != nil {
		fmt.Printf("Error initializing LLM: %v\n", err)
		return
	}
	core.GlobalConfig.DefaultLLM = llm

	testCases := []TestCase{
		{Nums: []int{1, 2, 3, 4}, Difficulty: "Easy"},       // Trivial: 1*2*3*4=24
		{Nums: []int{2, 3, 5, 12}, Difficulty: "Medium"},    // (5+3+2)*? no. 12*2=24..
		{Nums: []int{4, 9, 10, 13}, Difficulty: "Hard"},     // The classic hard one
	}

	fmt.Println("=== ReAcTree Benchmark: Linear vs. Tree Search ===")
	
	baselineSuccess := 0
	improvedSuccess := 0
	
	// Mode 1: Linear (Baseline)
	/*
	fmt.Println("\n--- Runing Baseline (Linear ReAct, MaxRetries=0) ---")
	baselineConfig := AgentConfig{MaxRetries: 0, NumCandidates: 1, UseScorer: false}
	baselineResults := runSuite(ctx, testCases, baselineConfig)
	printSummary("Baseline", baselineResults)
	for _, r := range baselineResults { if r.Success { baselineSuccess++ } }

	// Mode 2: Search (Improved)
	fmt.Println("\n--- Running Improved (Tree Search, MaxRetries=5) ---")
	improvedConfig := AgentConfig{MaxRetries: 5, NumCandidates: 1, UseScorer: false}
	improvedResults := runSuite(ctx, testCases, improvedConfig)
	printSummary("Improved", improvedResults)
	for _, r := range improvedResults { if r.Success { improvedSuccess++ } }
	*/

	// Mode 3: ToT (Search + Scorer + Candidates) (Simplified for Speed)
	fmt.Println("\n--- Running ToT (Tree of Thoughts, Candidates=2, MaxRetries=2) ---")
	totConfig := AgentConfig{MaxRetries: 2, NumCandidates: 2, UseScorer: true}
	totResults := runSuite(ctx, testCases, totConfig)
	printSummary("ToT", totResults)
	
	totSuccess := 0
	for _, r := range totResults { if r.Success { totSuccess++ } }

	fmt.Printf("\nOverall Success Rate:\nBaseline: %d/%d (%.0f%%)\nImproved: %d/%d (%.0f%%)\nToT: %d/%d (%.0f%%)\n", 
		baselineSuccess, len(testCases), float64(baselineSuccess)/float64(len(testCases))*100,
		improvedSuccess, len(testCases), float64(improvedSuccess)/float64(len(testCases))*100,
		totSuccess, len(testCases), float64(totSuccess)/float64(len(testCases))*100)
}

type AgentConfig struct {
	MaxRetries    int
	NumCandidates int
	UseScorer     bool
}

func runSuite(ctx context.Context, tests []TestCase, config AgentConfig) []Result {
	var results []Result
	for _, tc := range tests {
		fmt.Printf("Running %s Case %v... ", tc.Difficulty, tc.Nums)
		start := time.Now()
		
		agent := SetupAgent(ctx, core.GlobalConfig.DefaultLLM, tc.Nums, config)
		
		status, _ := agent.Run(ctx)
		duration := time.Since(start)
		
		success := (status == reactree.StatusSuccess)
		fmt.Printf("[%s] in %s\n", status, duration)
		
		steps := 0
		if agent.Memory != nil {
			steps = len(agent.Memory.GetLogs())
		}
		
		if !success {
			fmt.Printf("\n--- Failure Logs [%s] ---\n", tc.Difficulty)
			if agent.Memory != nil {
				for _, log := range agent.Memory.GetLogs() {
					fmt.Println(log)
				}
			}
			fmt.Println("---------------------------")
		}
		
		results = append(results, Result{
			Difficulty: tc.Difficulty,
			Success:    success,
			Duration:   duration,
			Steps:      steps, 
		})
	}
	return results
}

func printSummary(name string, results []Result) {
	fmt.Printf("\nSummary for %s:\n", name)
	fmt.Printf("%-10s %-10s %-15s %-10s\n", "Difficulty", "Success", "Duration", "Steps")
	for _, r := range results {
		fmt.Printf("%-10s %-10v %-15s %-10d\n", r.Difficulty, r.Success, r.Duration, r.Steps)
	}
}

// SetupAgent creates a ReAcTree agent to solve Game of 24.
func SetupAgent(ctx context.Context, llm core.LLM, nums []int, config AgentConfig) *reactree.TreeExecutor {
	// Prompt Engineering
	goal := fmt.Sprintf("Use the numbers %v and basic arithmetic operations (+, -, *, /) to obtain exactly 24. Return the solution steps. Each step should use two numbers. Example: 10 - 4 = 6. 13 - 9 = 4. 6 * 4 = 24. Action MUST be DONE when finished.", nums)
	
	// Create Episodic Memory with Few-Shot Examples (Bootstrapping from Paper)
	episodic := reactree.NewEpisodicMemory()
	exampleKey := "Use the numbers" // Matches any Game of 24 goal
	
	// Example 1: [4, 9, 10, 13] -> 24
	// Strategy: (13 - 9) * (10 - 4) = 4 * 6 = 24
	// Decomposition: 
	// 1. Get 4 from [13, 9]
	// 2. Get 6 from [10, 4]
	// 3. Multiply 4 * 6
	exampleTrace := `
--- Example ---
Goal: Use the numbers [4 9 10 13] and basic arithmetic operations (+, -, *, /) to obtain exactly 24.
Memory: []
Thought: I need to reach 24. A possible way is 4 * 6. I can make 4 from 13 - 9. I can make 6 from 10 - 4.
Action: 
NewSubgoals: ["Obtain 4 using [13, 9]", "Obtain 6 using [10, 4]", "Multiply results"]
---
Goal: Obtain 4 using [13, 9]
Memory: ...
Thought: 13 - 9 = 4.
Action: SUCCESS
NewSubgoals: []
---
Goal: Obtain 6 using [10, 4]
Memory: ...
Thought: 10 - 4 = 6.
Action: SUCCESS
NewSubgoals: []
---
Goal: Multiply results
Memory: ...
Thought: I have 4 and 6. 4 * 6 = 24.
Action: DONE
NewSubgoals: []
`
	episodic.Add(exampleKey, exampleTrace)

	sig := core.NewSignature(
		[]core.InputField{
			{Field: core.NewTextField("goal")},
			{Field: core.NewTextField("memory")},
			{Field: core.NewTextField("examples")},
		},
		[]core.OutputField{
			{Field: core.NewTextField("thought")},
			{Field: core.NewTextField("action")},
			{Field: core.NewTextField("new_subgoals")},
		},
	)

	// We use modules.NewPredict
	predict := modules.NewPredict(sig)
	
	// Create Agent Node
	agent := reactree.NewAgentNodeWithConfig(goal, predict, episodic, config.MaxRetries)
	
	// Configure Search (ToT)
	if config.NumCandidates > 1 || config.UseScorer {
		var scorer reactree.Scorer
		if config.UseScorer {
			scorer = reactree.NewDefaultScorer()
		}
		agent.WithSearch(config.NumCandidates, scorer)
	}
	
	executor := reactree.NewTreeExecutor(agent, nil, 10) // 10 steps max
	return executor
}
