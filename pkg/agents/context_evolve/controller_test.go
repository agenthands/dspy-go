package context_evolve

import (
	"context"
	"fmt"
	"testing"
	//"time"
)

// Mock agents for testing controller
type MockSummarizer struct{}
func (m *MockSummarizer) Summarize(ctx context.Context, p *Program, parentAbstract string) (string, error) {
    return "Mock Abstract", nil
}

type MockNavigator struct{}
func (m *MockNavigator) GetGradient(ctx context.Context, history []*Program) (string, error) {
    return "Mock Gradient", nil
}

type MockSampler struct{}
func (m *MockSampler) SelectReferences(ctx context.Context, population []*Program, parent *Program, gradient string) ([]*Program, error) {
    return []*Program{}, nil
}
func (m *MockSampler) SampleFromPopulation(database *ProgramDatabase, batchSize int) []*Program {
    return []*Program{}
}

type MockEvolution struct {
    Count int
}
func (m *MockEvolution) Evolve(ctx context.Context, parent *Program, references []*Program, gradient string) (string, error) {
    m.Count++
    return fmt.Sprintf("Evolved Content %d", m.Count), nil
}

// Mock Evaluator
type MockEvaluator struct{}
func (m *MockEvaluator) Evaluate(ctx context.Context, program *Program) (map[string]interface{}, error) {
    return map[string]interface{}{"score": 0.5}, nil
}

func TestController_Run_Integration(t *testing.T) {
    // Setup Controller with mocks
    config := Config{
        Evolution: EvolutionConfig{
            MaxIterations: 2,
            NumIslands: 1,
            TargetScore: 1.0, // Should not reach 1.0 with mock score 0.5
        },
        Evaluator: EvaluatorConfig{
            Command: "echo", 
        },
        LLM: LLMConfig{
            MaxTokens: 100,
        },
    }

    controller := NewController(config)
    
    // Inject mocks
    controller.Evaluator = &MockEvaluator{}
    controller.Agents.Summarizer = &MockSummarizer{}
    controller.Agents.Navigator = &MockNavigator{}
    controller.Agents.Sampler = &MockSampler{}
    controller.Agents.Evolution = &MockEvolution{}

    // Run
    initialContent := "Initial Content"
    best, err := controller.Run(context.Background(), initialContent)
    if err != nil {
        t.Fatalf("Run failed: %v", err)
    }

    if best == nil {
        t.Fatal("Expected best program, got nil")
    }

    // Since score is constant 0.5, best might be random or last one depending on logic.
    // We mainly want to ensure it ran through iterations without panic.
    
    // Check if ProgramDatabase has populated
    programs := controller.Database.GetAllPrograms()
    // Initial + 2 iterations = 3 programs
    if len(programs) != 3 {
        t.Errorf("Expected 3 programs in database, got %d", len(programs))
    }
}
