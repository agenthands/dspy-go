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

// AgentNodeImpl implements AgentNode.
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
	// 1. Check if we have unexecuted children (already expanded)
	// In the backtrack logic, if children exist but failed previously, we might have cleared them.
	// If they exist now, we run them.
	if len(n.Children) > 0 {
		childrenFailed := false
		for _, child := range n.Children {
			status, err := child.Execute(ctx, memory)
			if err != nil { return StatusFailure, err }
			if status == StatusFailure {
				memory.AddObservation(fmt.Sprintf("Agent[%s] Subgoal failed", n.Goal), "Plan failed. Re-planning...")
				n.Children = nil 
				childrenFailed = true
				break 
			}
			if status == StatusRunning { return StatusRunning, nil }
		}
		if !childrenFailed { return StatusSuccess, nil }
		// Fall through to re-planning
	}

	// 2. Prepare Inputs
	logs := memory.GetLogs()
	memoryStr := strings.Join(logs, "\n")
	
	examplesStr := ""
	if n.Episodic != nil {
		if s, ok := n.Episodic.Get(n.Goal).(string); ok {
			examplesStr = s
		}
	}

	// 3. Generate Candidates (Tree Search / Sampling)
	// If NumCandidates > 1, we generate multiple. If Scorer exists, we rank them.
	// If MaxRetries > 0, strict "retries" are effectively subsumed by "trying the next candidate"
	// OR we can view MaxRetries as "Process Attempts" if we can't generate enough valid candidates.
	// For simplicity, we will generate TOTAL = NumCandidates + MaxRetries candidates?
	// Let's stick to the Plan: Generate k candidates, Sort, Try in order.
	
	candidatesToGenerate := n.NumCandidates
	if candidatesToGenerate < 1 { candidatesToGenerate = 1 }
	// Allow extra "retries" to essentially mean "more candidates" or "rounds"?
	// If MaxRetries=5, NumCandidates=1 -> 6 sequential attempts (Standard ReAct/Backtrack).
	// If MaxRetries=0, NumCandidates=3 -> 3 parallel samples, pick best.
	// We'll combine: Total Attempts = candidatesToGenerate + n.MaxRetries
	
	// We'll combine: Total Attempts = candidatesToGenerate + n.MaxRetries
	// totalAttempts := candidatesToGenerate + n.MaxRetries (Unused)
	
	var candidates []Candidate

	// Generation Loop
	// We generate one by one for now (limitation of current Module.Process which isn't batched easily without parallelism)
	// TODO: Parallelize this
	for i := 0; i < candidatesToGenerate; i++ {
		// Variable temperature could involve options... for now standard
		inputs := map[string]interface{}{
			"goal": n.Goal, "memory": memoryStr, "examples": examplesStr,
		}
		res, err := n.Module.Process(ctx, inputs)
		if err != nil { continue } // Skip failed generations
		
		thought, _ := res["thought"].(string)
		action, _ := res["action"].(string)
		subgoals, _ := res["new_subgoals"].(string)
		
		c := Candidate{
			Thought:     thought,
			Action:      action,
			NewSubgoals: subgoals,
			Score:       0.5, // Default
		}
		
		// Score
		if n.Scorer != nil {
			// Construct candidate representation for scorer
			candidateStr := fmt.Sprintf("Thought: %s\nAction: %s\nSubgoals: %s", c.Thought, c.Action, c.NewSubgoals)
			score, reasoning, err := n.Scorer.Score(ctx, n.Goal, memoryStr, candidateStr)
			if err == nil {
				c.Score = score
				c.Reasoning = reasoning
			}
		}
		candidates = append(candidates, c)
	}
	
	// If we have linear retries (MaxRetries) and didn't find enough candidates, or just treat retries as fallbacks?
	// Actually, the previous "Backtracking" relied on the loop to "try, fail, retry".
	// With Candidates, we "generate, sort, try all".
	// If we run out of candidates, we fail. 
	// To preserve `MaxRetries` behavior when NumCandidates=1 (default), we should just run the loop `MaxRetries` times.
	// BUT, we want to SORT them.
	// If NumCandidates=1, sorting is trivial.
	
	// Simple Hybrid:
	// We just collect `candidatesToGenerate` candidates.
	// If `candidatesToGenerate` == 1, we might effectively be doing linear retries if we wrap this whole thing in a loop?
	// NO, let's make `Execute` try the candidates in order.
	
	// Sort by Score Descending
	// Bubble sort for small N
	for i := 0; i < len(candidates)-1; i++ {
		for j := 0; j < len(candidates)-i-1; j++ {
			if candidates[j].Score < candidates[j+1].Score {
				candidates[j], candidates[j+1] = candidates[j+1], candidates[j]
			}
		}
	}
	
	// Execution Loop (Try candidates in order)
	for i, cand := range candidates {
		// Log selection
		if n.Scorer != nil && n.NumCandidates > 1 {
			memory.AddObservation(fmt.Sprintf("Agent[%s] Selected Candidate %d/%d", n.Goal, i+1, len(candidates)), 
				fmt.Sprintf("Score: %.2f | %s", cand.Score, cand.Reasoning))
		}
		
		// Log thought
		memory.AddObservation(fmt.Sprintf("Agent[%s] Thought", n.Goal), cand.Thought)

		// Decompose?
		if cand.NewSubgoals != "" && cand.NewSubgoals != "[]" {
			var subgoals []string
			if err := json.Unmarshal([]byte(cand.NewSubgoals), &subgoals); err == nil && len(subgoals) > 0 {
				for _, sg := range subgoals {
					child := NewAgentNode(sg, n.Module, n.Episodic)
					// Inherit Search Config? Maybe.
					child.WithSearch(n.NumCandidates, n.Scorer) // Recursive Search
					n.AddChild(child)
				}
				memory.AddObservation(fmt.Sprintf("Agent[%s] Decomposed into %d subgoals", n.Goal, len(subgoals)), subgoals)
				return StatusRunning, nil
			}
		}

		// Action?
		if cand.Action != "" {
			if cand.Action == "FAIL" {
				memory.AddObservation(fmt.Sprintf("Agent[%s] Failed explicitly", n.Goal), "Candidate failed. Backtracking...")
				continue // Try next candidate
			}
			if cand.Action == "DONE" || cand.Action == "SUCCESS" {
				return StatusSuccess, nil
			}
			memory.AddObservation(fmt.Sprintf("Agent[%s] Action", n.Goal), cand.Action)
			return StatusRunning, nil
		}
	}

	// Fallback to "Retries" if logic requires more linear attempts beyond initial batch?
	// For now, if all candidates fail, we fail. 
	// This respects `NumCandidates` as the breadth.
	
	return StatusFailure, fmt.Errorf("all %d candidates failed", len(candidates))
}
