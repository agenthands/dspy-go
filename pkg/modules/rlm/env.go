package rlm

import (
	"context"
	"time"
)

// ExecutionEnvironment defines the interface for RLM execution backends.
// This allows RLM to support different reasoning engines (e.g., Yaegi/Go, Mangle/Datalog).
type ExecutionEnvironment interface {
	// LoadContext loads the initial context data into the environment.
	LoadContext(contextPayload any) error

	// Execute runs a code snippet or query in the environment.
	Execute(ctx context.Context, code string) (ExecutionResult, error)

	// GetContextInfo returns a summary of the loaded context (e.g., size, schema).
	GetContextInfo() string

	// GetState returns a string representation of the current environment state (variables/facts).
	GetState() string

	// GetVariable retrieves the string value of a variable by name.
	// Returns error if variable not found or not supported.
	GetVariable(name string) (string, error)

	// GetLLMCalls returns a list of LLM calls made during execution (if any).
	GetLLMCalls() []LLMCall

	// SetVariable sets a variable in the environment.
	// Used for passing sub-RLM results back to the parent.
	SetVariable(name, value string) error

	// ClearFinal resets any "final answer" state.
	ClearFinal()

	// HasFinal returns true if a final answer has been determined by the environment logic.
	HasFinal() bool

	// RegisterFunction registers a custom function in the environment.
	RegisterFunction(name string, fn any) error

	// Final returns the final answer string, if available.
	Final() string
}

// ExecutionResult captures the output of an execution step.
type ExecutionResult struct {
	Stdout   string
	Stderr   string
	Duration time.Duration
}

// CodeContext allows the user to explicitly pass code content with language metadata.
type CodeContext struct {
	Language string
	Code     string
}
