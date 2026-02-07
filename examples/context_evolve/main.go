package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	ce "github.com/XiaoConstantine/dspy-go/pkg/agents/context_evolve"
    "github.com/XiaoConstantine/dspy-go/pkg/core"
    "github.com/XiaoConstantine/dspy-go/pkg/llms"
)

func main() {
    // 1. Build the evaluator binary
    evaluatorDir := "cmd/evaluator"
    evaluatorBin := "evaluator_bin"
    
    cmd := exec.Command("go", "build", "-o", evaluatorBin, ".")
    cmd.Dir = filepath.Join("examples", "context_evolve", evaluatorDir)
    // Adjust path if running from root
    if _, err := os.Stat(cmd.Dir); os.IsNotExist(err) {
        // Assume running from examples/context_evolve
         cmd.Dir = evaluatorDir
    }
    
    out, err := cmd.CombinedOutput()
    if err != nil {
        log.Fatalf("Failed to build evaluator: %v\n%s", err, out)
    }
    defer os.Remove(filepath.Join(cmd.Dir, evaluatorBin)) // Cleanup

    absEvaluatorPath, _ := filepath.Abs(filepath.Join(cmd.Dir, evaluatorBin))

    // 2. Setup LLM
    apiKey := os.Getenv("OPENAI_API_KEY")
    if apiKey == "" {
        // Just print warning and exit or continue with fake key if we just want to test compilation
        fmt.Println("Warning: OPENAI_API_KEY not set. Example will fail at LLM generation step.")
        apiKey = "fake-key"
    }
    
    // Create LLM instance
    llm, err := llms.NewOpenAI(core.ModelOpenAIGPT35Turbo, apiKey)
    if err != nil {
        log.Fatalf("Failed to configure LLM: %v", err)
    }
    core.SetDefaultLLM(llm)

    // 3. Setup Config
    config := ce.Config{
        Evolution: ce.EvolutionConfig{
            MaxIterations: 4,
            TargetScore:   0.99,
            NumIslands:    2,
            MigrationInterval: 2,
        },
        Prompt: ce.PromptConfig{
            NumReferences: 2,
        },
        Evaluator: ce.EvaluatorConfig{
            Command:        absEvaluatorPath,
            TimeoutSeconds: 2,
        },
        Database: ce.DatabaseConfig{
            FilePath: "population.json",
        },
    }

    // 4. Initialize Controller
    controller := ce.NewController(config)
    
    // Check for Mock Mode
    if os.Getenv("MOCK_MODE") == "true" || apiKey == "fake-key" {
        fmt.Println("Running in MOCK MODE (Simulated Agents)")
        controller.Agents.Summarizer = &MockSummarizer{}
        controller.Agents.Navigator = &MockNavigator{}
        controller.Agents.Sampler = &MockSampler{}
        controller.Agents.Evolution = &MockEvolution{}
    }

    // 5. Run Evolution
    initialPolicy := `{"guess": 0}` // Initial bad guess
    fmt.Printf("Starting evolution with initial policy: %s\n", initialPolicy)

    ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
    defer cancel()

    best, err := controller.Run(ctx, initialPolicy)
    if err != nil {
        log.Fatalf("Evolution failed: %v", err)
    }

    fmt.Printf("Evolution complete!\nBest Program ID: %s\nContent: %s\nMetrics: %v\n", 
        best.ID, best.Content, best.Metrics)
}

// --- Mock Agents for Automated Testing ---

type MockSummarizer struct{}
func (m *MockSummarizer) Summarize(ctx context.Context, p *ce.Program, parentAbstract string) (string, error) {
    return fmt.Sprintf("Summary of %s", p.Content), nil
}

type MockNavigator struct{}
func (m *MockNavigator) GetGradient(ctx context.Context, history []*ce.Program) (string, error) {
    return "Try increasing the guess number", nil
}

type MockSampler struct{}
func (m *MockSampler) SelectReferences(ctx context.Context, population []*ce.Program, parent *ce.Program, gradient string) ([]*ce.Program, error) {
    return []*ce.Program{}, nil
}
func (m *MockSampler) SampleFromPopulation(database *ce.ProgramDatabase, batchSize int) []*ce.Program {
    return []*ce.Program{}
}

type MockEvolution struct {
    Count int
}
func (m *MockEvolution) Evolve(ctx context.Context, parent *ce.Program, references []*ce.Program, gradient string) (string, error) {
    m.Count++
    // Simple heuristic evolution for the number guessing game
    return fmt.Sprintf(`{"guess": %d}`, (m.Count * 25)), nil
}
