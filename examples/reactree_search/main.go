package main

import (
	"context"
	"fmt"

	"github.com/XiaoConstantine/dspy-go/pkg/agents/reactree"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/llms"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
)

// Game24Env represents the environment for the Game of 24.
type Game24Env struct {
	Numbers []int
}

// NewGame24Env creates a new environment.
func NewGame24Env(nums []int) *Game24Env {
	return &Game24Env{Numbers: nums}
}

// CheckSolution verifies if an expression evaluates to 24 using the given numbers.
// This is a simplified check; in a real agent, the agent steps through operations.
func (env *Game24Env) Validate(expression string) (bool, error) {
	// For this POC, we trust the agent's step-by-step logic, 
	// but we could use a math evaluator here.
	// For simplicity, we just check if the final result is 24.
	// Implementation of a full expression parser is out of scope for this snippet,
	// so we will rely on the agent verifying its own result or a simple robust check.
	return false, nil
}

func main() {
	ctx := context.Background()
	llm, err := llms.NewOllamaLLM("gpt-oss:latest") // Using available local model
	if err != nil {
		fmt.Printf("Error initializing LLM: %v\n", err)
		return
	}
	// Configure LLM
	core.GlobalConfig.DefaultLLM = llm

	// Test Case: 4, 9, 10, 13
	nums := []int{4, 9, 10, 13}
	fmt.Printf("Solving for: %v\n", nums)

	agentSearch := SetupAgent(ctx, core.GlobalConfig.DefaultLLM, nums, 5)
	status, err := agentSearch.Run(ctx)
	if err != nil {
		fmt.Printf("Search Execution Error: %v\n", err)
	}
	fmt.Printf("Search Status: %s\n", status)
	
	// Print Logs
	for _, log := range agentSearch.Memory.GetLogs() {
		fmt.Println(log)
	}
}

func SetupAgent(ctx context.Context, llm core.LLM, nums []int, maxRetries int) *reactree.TreeExecutor {
	// Prompt Engineering
	goal := fmt.Sprintf("Use the numbers %v and basic arithmetic operations (+, -, *, /) to obtain exactly 24. Return the solution steps. Each step should use two numbers. Example: 10 - 4 = 6. 13 - 9 = 4. 6 * 4 = 24. Action MUST be DONE when finished.", nums)
	
	// Create module with specific prompt for 24
	// ReActSignature is:
	// Goal, Memory, Examples -> Thought, Action, NewSubgoals
	
	// ReActSignature is:
	// Goal, Memory, Examples -> Thought, Action, NewSubgoals
	
	// We can use BootstrapFewShot here if we had more examples, but for now simple Predict is fine.
	// We can use BootstrapFewShot here if we had more examples, but for now simple Predict is fine.
	// But we need to ensure the LLM understands the ReAct format.
	// The `reactree` package uses `dspy:"field"` tags, so `Predict` should handle it.
	
	// We need to define the signature struct properly if we want `Predict` to work seamlessly.
	// `reactree.ReActSignature` is exported.
	
	// ReActSignature is:
	// Goal, Memory, Examples -> Thought, Action, NewSubgoals
	
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
	agent := reactree.NewAgentNodeWithConfig(goal, predict, nil, maxRetries)
	
	executor := reactree.NewTreeExecutor(agent, nil, 10) // 10 steps max
	return executor
}
