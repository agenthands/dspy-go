package reactree

import "context"

// NodeStatus represents the status of a node execution.
type NodeStatus string

const (
	StatusRunning NodeStatus = "Running"
	StatusSuccess NodeStatus = "Success"
	StatusFailure NodeStatus = "Failure"
)

// Node is the base interface for all nodes in the ReAcTree.
type Node interface {
	// Execute runs the node's logic.
	// It returns the status of the execution and any error encountered.
	Execute(ctx context.Context, memory *WorkingMemory) (NodeStatus, error)
	
	// GetChildren returns the child nodes of this node.
	GetChildren() []Node
	
	// AddChild adds a child node.
	AddChild(node Node)
}

// AgentNode represents a node managed by an LLM agent.
// It can reason, act, and dynamically expand the tree by adding subgoals.
type AgentNode interface {
	Node
	// GetGoal returns the specific goal of this agent node.
	GetGoal() string
}

// ControlNode represents a node that manages the control flow of its children.
// Examples include Sequence (AND) and Selector (OR).
type ControlNode interface {
	Node
	// GetType returns the type of control flow (e.g., "Sequence", "Selector").
	GetType() string
}

// Memory represents a storage system for the agent.
type Memory interface {
	// Add stores a value associated with a key.
	Add(key string, value interface{})
	// Get retrieves a value by key.
	Get(key string) interface{}
}
