package reactree

import (
	"context"
	"errors"
	"testing"
)

// MockNode is a simple node for testing control flow.
type MockNode struct {
	Status      NodeStatus
	Err         error
	ExecuteFunc func(ctx context.Context, memory *WorkingMemory) (NodeStatus, error)
	Children    []Node
}

func (m *MockNode) Execute(ctx context.Context, memory *WorkingMemory) (NodeStatus, error) {
	if m.ExecuteFunc != nil {
		return m.ExecuteFunc(ctx, memory)
	}
	return m.Status, m.Err
}

func (m *MockNode) GetChildren() []Node {
	return m.Children
}

func (m *MockNode) AddChild(node Node) {
	m.Children = append(m.Children, node)
}

func TestSequenceNode(t *testing.T) {
	ctx := context.Background()
	mem := NewWorkingMemory()

	t.Run("AllSuccess", func(t *testing.T) {
		seq := NewSequenceNode(
			&MockNode{Status: StatusSuccess},
			&MockNode{Status: StatusSuccess},
		)
		status, err := seq.Execute(ctx, mem)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		if status != StatusSuccess {
			t.Errorf("Expected Success, got %s", status)
		}
	})

	t.Run("FailureStopsExecution", func(t *testing.T) {
		seq := NewSequenceNode(
			&MockNode{Status: StatusSuccess},
			&MockNode{Status: StatusFailure},
			&MockNode{Status: StatusSuccess}, // Should not be executed ideally, but mock just returns status
		)
		status, err := seq.Execute(ctx, mem)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		if status != StatusFailure {
			t.Errorf("Expected Failure, got %s", status)
		}
	})

	t.Run("RunningStopsExecution", func(t *testing.T) {
		seq := NewSequenceNode(
			&MockNode{Status: StatusSuccess},
			&MockNode{Status: StatusRunning},
			&MockNode{Status: StatusFailure},
		)
		status, err := seq.Execute(ctx, mem)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		if status != StatusRunning {
			t.Errorf("Expected Running, got %s", status)
		}
	})
	
	t.Run("ErrorPropagates", func(t *testing.T) {
		expectedErr := errors.New("child error")
		seq := NewSequenceNode(
			&MockNode{Status: StatusSuccess},
			&MockNode{Err: expectedErr},
		)
		_, err := seq.Execute(ctx, mem)
		if err != expectedErr {
			t.Errorf("Expected error %v, got %v", expectedErr, err)
		}
	})
}

func TestSelectorNode(t *testing.T) {
	ctx := context.Background()
	mem := NewWorkingMemory()

	t.Run("FirstSuccess", func(t *testing.T) {
		sel := NewSelectorNode(
			&MockNode{Status: StatusSuccess},
			&MockNode{Status: StatusFailure},
		)
		status, err := sel.Execute(ctx, mem)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		if status != StatusSuccess {
			t.Errorf("Expected Success, got %s", status)
		}
	})

	t.Run("LaterSuccess", func(t *testing.T) {
		sel := NewSelectorNode(
			&MockNode{Status: StatusFailure},
			&MockNode{Status: StatusSuccess},
		)
		status, err := sel.Execute(ctx, mem)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		if status != StatusSuccess {
			t.Errorf("Expected Success, got %s", status)
		}
	})

	t.Run("AllFailure", func(t *testing.T) {
		sel := NewSelectorNode(
			&MockNode{Status: StatusFailure},
			&MockNode{Status: StatusFailure},
		)
		status, err := sel.Execute(ctx, mem)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		if status != StatusFailure {
			t.Errorf("Expected Failure, got %s", status)
		}
	})
	
	t.Run("RunningStops", func(t *testing.T) {
		sel := NewSelectorNode(
			&MockNode{Status: StatusFailure},
			&MockNode{Status: StatusRunning},
			&MockNode{Status: StatusSuccess},
		)
		status, err := sel.Execute(ctx, mem)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		if status != StatusRunning {
			t.Errorf("Expected Running, got %s", status)
		}
	})
}
