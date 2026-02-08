package reactree

import (
	"context"
	"testing"

)

func TestTreeExecutor_Run(t *testing.T) {
	ctx := context.Background()
	mem := NewWorkingMemory()
	
	// Case 1: Instant Success
	root := &MockNode{Status: StatusSuccess}
	exec := NewTreeExecutor(root, mem, 10)
	status, err := exec.Run(ctx)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if status != StatusSuccess {
		t.Errorf("Expected Success, got %s", status)
	}
	
	// Case 2: Running then Success
	// We need a stateful mock that changes status
	statefulRoot := &MockStatefulNode{
		TicksBeforeSuccess: 3,
	}
	exec2 := NewTreeExecutor(statefulRoot, mem, 10)
	status, err = exec2.Run(ctx)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if status != StatusSuccess {
		t.Errorf("Expected Success, got %s", status)
	}
	if statefulRoot.Ticks != 4 { // 3 runnings + 1 success = 4 calls? 
		// Tick 1: Running (Ticks=1)
		// Tick 2: Running (Ticks=2)
		// Tick 3: Running (Ticks=3)
		// Tick 4: Success (Ticks=4)
		t.Errorf("Expected 4 ticks, got %d", statefulRoot.Ticks)
	}
	
	// Case 3: Max Steps Reached
	foreverRoot := &MockNode{Status: StatusRunning}
	exec3 := NewTreeExecutor(foreverRoot, mem, 5)
	status, err = exec3.Run(ctx)
	if err == nil {
		t.Fatal("Expected error for max steps")
	}
	if status != StatusFailure {
		t.Errorf("Expected Failure, got %s", status)
	}
}

type MockStatefulNode struct {
	MockNode
	Ticks              int
	TicksBeforeSuccess int
}

func (m *MockStatefulNode) Execute(ctx context.Context, memory *WorkingMemory) (NodeStatus, error) {
	m.Ticks++
	if m.Ticks > m.TicksBeforeSuccess {
		return StatusSuccess, nil
	}
	return StatusRunning, nil
}
