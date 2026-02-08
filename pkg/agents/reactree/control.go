package reactree

import (
	"context"

)

// BaseNode provides common functionality for nodes.
type BaseNode struct {
	Children []Node
}

func (n *BaseNode) GetChildren() []Node {
	return n.Children
}

func (n *BaseNode) AddChild(child Node) {
	n.Children = append(n.Children, child)
}

// SequenceNode executes children sequentially.
// Fails if ANY child fails. Succeeds if ALL children succeed.
type SequenceNode struct {
	BaseNode
}

func NewSequenceNode(children ...Node) *SequenceNode {
	return &SequenceNode{
		BaseNode: BaseNode{Children: children},
	}
}

func (s *SequenceNode) Execute(ctx context.Context, memory *WorkingMemory) (NodeStatus, error) {
	for _, child := range s.Children {
		status, err := child.Execute(ctx, memory)
		if err != nil {
			return StatusFailure, err // Or propagate error behavior?
		}

		if status == StatusFailure {
			return StatusFailure, nil
		}
		
		// If child is running, the sequence is still running (wait for child completion)
		// For ReAcTree, execution might be recursive. If a child expands, returns Running?
		// Assuming synchronous execution for now, or returns Running to yield.
		if status == StatusRunning {
			return StatusRunning, nil
		}
	}
	return StatusSuccess, nil
}

func (s *SequenceNode) GetType() string {
	return "Sequence"
}

// SelectorNode executes children sequentially.
// Succeeds if ANY child succeeds. Fails if ALL children fail.
type SelectorNode struct {
	BaseNode
}

func NewSelectorNode(children ...Node) *SelectorNode {
	return &SelectorNode{
		BaseNode: BaseNode{Children: children},
	}
}

func (s *SelectorNode) Execute(ctx context.Context, memory *WorkingMemory) (NodeStatus, error) {
	for _, child := range s.Children {
		status, err := child.Execute(ctx, memory)
		if err != nil {
			return StatusFailure, err
		}

		if status == StatusSuccess {
			return StatusSuccess, nil
		}

		if status == StatusRunning {
			return StatusRunning, nil
		}
		// If failure, try next child
	}
	// All children failed
	return StatusFailure, nil
}

func (s *SelectorNode) GetType() string {
	return "Selector"
}
