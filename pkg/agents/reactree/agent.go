package reactree

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

// ReActSignature defines the input/output for the ReAcTree agent.
type ReActSignature struct {
	Goal     string `dspy:"goal"`
	Memory   string `dspy:"memory"`
	Examples string `dspy:"examples"`
	
	Thought     string `dspy:"thought"`
	Action      string `dspy:"action"`       // e.g., "SEARCH[query]" or "DONE" or "FAIL"
	NewSubgoals string `dspy:"new_subgoals"` // e.g., JSON list ["subgoal1", "subgoal2"]
}

// Candidate represents a generated thought/action/subgoals tuple.
type Candidate struct {
	Thought     string
	Action      string
	NewSubgoals string
	Score       float64
	Reasoning   string
}

// AgentNodeImpl implements AgentNode.
type AgentNodeImpl struct {
	BaseNode
	Goal       string
	Module     core.Module
	Episodic   Memory // Optional
	MaxRetries int
	
	// Search Configuration
	NumCandidates int
	Scorer        Scorer
	
	// Runtime State (Persisted for Backtracking)
	Candidates            []Candidate
	CurrentCandidateIndex int
}

func NewAgentNode(goal string, module core.Module, episodic Memory) *AgentNodeImpl {
	return &AgentNodeImpl{
		Goal:          goal,
		Module:        module,
		Episodic:      episodic,
		MaxRetries:    3,
		NumCandidates: 1, // Default to linear
	}
}

func NewAgentNodeWithConfig(goal string, module core.Module, episodic Memory, maxRetries int) *AgentNodeImpl {
	return &AgentNodeImpl{
		Goal:          goal,
		Module:        module,
		Episodic:      episodic,
		MaxRetries:    maxRetries,
		NumCandidates: 1,
	}
}

// WithSearch configures the agent for Tree Search.
func (n *AgentNodeImpl) WithSearch(numCandidates int, scorer Scorer) *AgentNodeImpl {
	n.NumCandidates = numCandidates
	n.Scorer = scorer
	return n
}

func (n *AgentNodeImpl) GetGoal() string {
	return n.Goal
}

func (n *AgentNodeImpl) Execute(ctx context.Context, memory *WorkingMemory) (NodeStatus, error) {
	// 1. Prepare Inputs (Lazy load strings)
	logs := memory.GetLogs()
	memoryStr := strings.Join(logs, "\n")
	examplesStr := ""
	if n.Episodic != nil {
		if s, ok := n.Episodic.Get(n.Goal).(string); ok {
			examplesStr = s
		}
	}

	// 2. Generation Phase (If no candidates exist)
	if len(n.Candidates) == 0 {
		candidatesToGenerate := n.NumCandidates
		if candidatesToGenerate < 1 { candidatesToGenerate = 1 }
		// Optional: We could loop MaxRetries here for "rounds" of generation if needed.
		
		for i := 0; i < candidatesToGenerate; i++ {
			inputs := map[string]interface{}{
				"goal": n.Goal, "memory": memoryStr, "examples": examplesStr,
			}
			res, err := n.Module.Process(ctx, inputs)
			if err != nil { 
				memory.AddObservation(fmt.Sprintf("Agent[%s] Generation Error", n.Goal), err.Error())
				continue 
			}
			
			thought, _ := res["thought"].(string)
			action, _ := res["action"].(string)
			subgoals, _ := res["new_subgoals"].(string)
			
			c := Candidate{
				Thought:     thought,
				Action:      action,
				NewSubgoals: subgoals,
				Score:       0.5,
			}
			
			if n.Scorer != nil {
				candidateStr := fmt.Sprintf("Thought: %s\nAction: %s\nSubgoals: %s", c.Thought, c.Action, c.NewSubgoals)
				score, reasoning, err := n.Scorer.Score(ctx, n.Goal, memoryStr, candidateStr)
				if err == nil {
					c.Score = score
					c.Reasoning = reasoning
				}
			}
			n.Candidates = append(n.Candidates, c)
		}
		
		// Sort Candidates
		for i := 0; i < len(n.Candidates)-1; i++ {
			for j := 0; j < len(n.Candidates)-i-1; j++ {
				if n.Candidates[j].Score < n.Candidates[j+1].Score {
					n.Candidates[j], n.Candidates[j+1] = n.Candidates[j+1], n.Candidates[j]
				}
			}
		}
	}

	// 3. Execution Phase (Iterate through candidates)
	// We continue from CurrentCandidateIndex
	for n.CurrentCandidateIndex < len(n.Candidates) {
		cand := n.Candidates[n.CurrentCandidateIndex]
		
		// Check if we are resuming execution of this candidate's children
		if len(n.Children) > 0 {
			childrenFailed := false
			for _, child := range n.Children {
				status, err := child.Execute(ctx, memory)
				if err != nil { return StatusFailure, err }
				if status == StatusFailure {
					memory.AddObservation(fmt.Sprintf("Agent[%s] Candidate %d Subgoal failed", n.Goal, n.CurrentCandidateIndex+1), "Plan failed. Backtracking to next candidate...")
					n.Children = nil 
					childrenFailed = true
					break 
				}
				if status == StatusRunning { return StatusRunning, nil }
			}
			if !childrenFailed { return StatusSuccess, nil }
			
			// If children failed, we advance to next candidate
			n.CurrentCandidateIndex++
			continue 
		}

		// Use the candidate (First time execution)
		if n.Scorer != nil && n.NumCandidates > 1 {
			memory.AddObservation(fmt.Sprintf("Agent[%s] Trying Candidate %d/%d", n.Goal, n.CurrentCandidateIndex+1, len(n.Candidates)), 
				fmt.Sprintf("Score: %.2f | %s", cand.Score, cand.Reasoning))
		}
		memory.AddObservation(fmt.Sprintf("Agent[%s] Thought", n.Goal), cand.Thought)

		// Decomposition
		if cand.NewSubgoals != "" && cand.NewSubgoals != "[]" {
			var subgoals []string
			if err := json.Unmarshal([]byte(cand.NewSubgoals), &subgoals); err == nil && len(subgoals) > 0 {
				for _, sg := range subgoals {
					child := NewAgentNode(sg, n.Module, n.Episodic)
					child.WithSearch(n.NumCandidates, n.Scorer)
					n.AddChild(child)
				}
				memory.AddObservation(fmt.Sprintf("Agent[%s] Decomposed into %d subgoals", n.Goal, len(subgoals)), subgoals)
				return StatusRunning, nil
			}
		}

		// Action
		if cand.Action != "" {
			if cand.Action == "FAIL" {
				n.CurrentCandidateIndex++
				continue
			}
			if strings.Contains(cand.Action, "DONE") || strings.Contains(cand.Action, "SUCCESS") {
				return StatusSuccess, nil
			}
			memory.AddObservation(fmt.Sprintf("Agent[%s] Action", n.Goal), cand.Action)
			
			// Commit to this step: Clear candidates so we generate the NEXT step in the next Execute call.
			// This effectively makes the intra-node search "Greedy" per step, 
			// but allows inter-node (subgoal) backtracking.
			n.Candidates = nil
			n.CurrentCandidateIndex = 0
			
			return StatusRunning, nil
		}
		
		// If neither decomposition nor valid action, fail candidate
		n.CurrentCandidateIndex++
	}

	return StatusFailure, fmt.Errorf("all %d candidates failed", len(n.Candidates))
}
