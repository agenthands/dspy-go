package rlm

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestMangleEnvironmentIntegration tests the RLM with Mangle environment.
func TestMangleEnvironmentIntegration(t *testing.T) {
	// Mock LLM that acts like a Mangle-reasoning agent
	mockLLM := &mockLLM{
		responses: []string{
			// Iteration 1: Query
			`Reasoning: I will query the facts.
Action: query
Code: 
result(X) :- root(X).
Answer: `,
			// Iteration 2: Final
			`Reasoning: I found the answer.
Action: final
Code:
Answer: Mangle Stub Result`,
		},
	}

	// Create RLM with Mangle environment
	rlm := NewFromLLM(mockLLM, WithMangle())

	// Input context (will be walked into facts)
	ctxData := map[string]any{
		"users": []string{"alice", "bob"},
	}

	// Execute
	result, err := rlm.Complete(context.Background(), ctxData, "Who are the users?")
	require.NoError(t, err)

	// Verify
	assert.Equal(t, "Mangle Stub Result", result.Response)
	assert.Equal(t, 2, result.Iterations)
}

// TestMangleEnvironmentStructure tests the MangleEnvironment methods directly.
func TestMangleEnvironmentStructure(t *testing.T) {
	env := NewMangleEnvironment()

	// Load context
	err := env.LoadContext(map[string]int{"a": 1, "b": 2})
	require.NoError(t, err)

	// Check state
	state := env.GetState()
	assert.Contains(t, state, "Mangle Datalog State")
	assert.Contains(t, state, "facts loaded")

	// Execute (real)
	// We need facts for query to return results.
	// Add a fact via LoadContext first.
	err = env.LoadContext(map[string]string{"foo": "bar"})
	require.NoError(t, err)

	// Query foo("bar")?
	// The walker generates foo(root, "bar").
	res, err := env.Execute(context.Background(), `
match(X) :- foo(_, X).
result(X) :- match(X).
`)
	require.NoError(t, err)
	assert.Contains(t, res.Stdout, "bar")
}
