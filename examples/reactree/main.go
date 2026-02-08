package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/XiaoConstantine/dspy-go/pkg/agents/reactree"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	_ "github.com/XiaoConstantine/dspy-go/pkg/llms"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
)

func main() {
	ctx := context.Background()

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		fmt.Println("No OPENAI_API_KEY set. Skipping execution to avoid failure.")
		return
	}

	// 1. Configure LLM (Real one)
	if err := core.ConfigureDefaultLLM(apiKey, core.ModelOpenAIGPT4oMini); err != nil {
		log.Fatalf("Failed to configure LLM: %v", err)
	}

	fmt.Println("Starting ReAcTree Example...")

	// 2. Define the Agent Logic (using dspy.Predict)
	// We use the ReActSignature defined in the package
	// inputs: goal, memory, examples
	// outputs: thought, action, new_subgoals
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
	).WithInstruction("You are a ReAcTree agent. Decompose complex goals into subgoals or perform atomic actions.")

	// Create a Predict module
	predictModule := modules.NewPredict(sig)

	// 3. Initialize Memory
	workingMem := reactree.NewWorkingMemory()
	episodicMem := reactree.NewEpisodicMemory()
	// Seed episodic memory with an example
	episodicMem.Add("Complex Task", "Example: To solve X, I broke it down into A, B, C.")

	// 4. Create Root Agent Node
	rootGoal := "Research and summarize the history of AI."
	rootNode := reactree.NewAgentNode(rootGoal, predictModule, episodicMem)

	// 5. Create Executor
	executor := reactree.NewTreeExecutor(rootNode, workingMem, 10)

	// 6. Run

	status, err := executor.Run(ctx)
	if err != nil {
		log.Printf("Execution failed: %v", err)
	}

	fmt.Printf("Final Status: %s\n", status)
	
	// Print Logs
	fmt.Println("\n--- Execution Logs ---")
	for _, l := range workingMem.GetLogs() {
		fmt.Println(l)
	}
}
