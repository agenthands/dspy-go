package reactree

import (
	"context"
	"fmt"

)

// TreeExecutor manages the execution of a ReAcTree.
type TreeExecutor struct {
	Root     Node
	Memory   *WorkingMemory
	MaxSteps int
}

// NewTreeExecutor creates a new executor.
func NewTreeExecutor(root Node, memory *WorkingMemory, maxSteps int) *TreeExecutor {
	if memory == nil {
		memory = NewWorkingMemory()
	}
	return &TreeExecutor{
		Root:     root,
		Memory:   memory,
		MaxSteps: maxSteps,
	}
}

// Run executes the tree until completion or max steps.
func (e *TreeExecutor) Run(ctx context.Context) (NodeStatus, error) {
	for i := 0; i < e.MaxSteps; i++ {
		// Check cancellation
		select {
		case <-ctx.Done():
			return StatusFailure, ctx.Err()
		default:
		}

		// Execute one tick
		status, err := e.Root.Execute(ctx, e.Memory)
		if err != nil {
			return StatusFailure, err
		}

		if status == StatusSuccess {
			return StatusSuccess, nil
		}
		if status == StatusFailure {
			return StatusFailure, nil
		}

		// If Running, continue loop
		// Sleep a bit to prevent tight loop if needed? 
		// For LLM agent, execution is slow, so tight loop is fine (it's driven by I/O).
		// But if we mock it, it might spin fast.
		// Let's add a minimal yield or sleep if desired, but 0 is fine for now.
	}
	
	return StatusFailure, fmt.Errorf("max steps %d reached", e.MaxSteps)
}
