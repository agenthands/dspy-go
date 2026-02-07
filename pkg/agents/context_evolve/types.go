package context_evolve

import (
	"context"
	"time"
)

// Program represents an evolved program or action sequence.
// It contains the content (code/JSON), metadata, metrics, and lineage information.
type Program struct {
	ID             string                 `json:"id"`
	Content        string                 `json:"content"`          // Code or JSON/DSL action sequence
	Language       string                 `json:"language"`         // e.g., "python", "go", "json", "dsl"
	Metrics        map[string]interface{} `json:"metrics"`          // Evaluation metrics
	Abstract       string                 `json:"abstract"`         // High-level summary of logic/strategy
	Gradient       string                 `json:"gradient"`         // Optimization direction/feedback
	ParentID       string                 `json:"parent_id"`        // ID of the parent program
	Generation     int                    `json:"generation"`       // Generation number
	IterationFound int                    `json:"iteration_found"`  // Iteration when this program was created
	Timestamp      time.Time              `json:"timestamp"`
	Metadata       map[string]interface{} `json:"metadata"`
}

// Config holds configuration for the ContextEvolve process.
type Config struct {
	Evolution EvolutionConfig
	Evaluator EvaluatorConfig
	LLM       LLMConfig
	Database  DatabaseConfig
    Debug     DebugConfig
    Prompt    PromptConfig
}

type LLMConfig struct {
	Model       string  `json:"model"`
	Temperature float64 `json:"temperature"`
	MaxTokens   int     `json:"max_tokens"`
}

// EvolutionConfig defines parameters for the evolution process.
type EvolutionConfig struct {
	MaxIterations int
	NumIslands    int
	MigrationInterval int `json:"migration_interval"` // Iterations between migrations
	TargetScore   float64
    
    // Sampling weights for gradient synthesis (replay buffer)
    RolloutWeightAllImproved float64 `json:"rollout_weight_all_improved"` // Default: 0.6
    RolloutWeightMixed       float64 `json:"rollout_weight_mixed"`        // Default: 0.2
    RolloutWeightAllDegraded float64 `json:"rollout_weight_all_degraded"` // Default: 0.2
    
    // Limits
    MaxCodeLength int `json:"max_code_length"`
}

type DebugConfig struct {
    SimpleDebug   bool `json:"simple_debug"`
    PipelineDebug bool `json:"pipeline_debug"`
    PromptDebug   bool `json:"prompt_debug"`
    ResponseDebug bool `json:"response_debug"`
}

type PromptConfig struct {
	NumReferences     int `json:"num_references"`
	NumTopPrograms    int `json:"num_top_programs"`
	NumDiversePrograms int `json:"num_diverse_programs"`
}

type DatabaseConfig struct {
	FilePath string `json:"file_path"`
}

type EvaluatorConfig struct {
	Command       string `json:"command"`        // Command to run for evaluation
	TimeoutSeconds int    `json:"timeout_seconds"`
}

// Agent interfaces (potentially moved to agents.go if they get complex)
type Summarizer interface {
    Summarize(ctx context.Context, program *Program, parentAbstract string) (string, error)
}

type Navigator interface {
    GetGradient(ctx context.Context, history []*Program) (string, error)
}

type Sampler interface {
    SelectReferences(ctx context.Context, population []*Program, parent *Program, gradient string) ([]*Program, error)
    SampleFromPopulation(database *ProgramDatabase, batchSize int) []*Program
}

type Evolution interface {
    Evolve(ctx context.Context, parent *Program, references []*Program, gradient string) (string, error)
}
