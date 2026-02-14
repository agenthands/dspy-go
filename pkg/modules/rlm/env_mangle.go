package rlm

import (
	"context"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"time"

	"github.com/google/mangle/ast"
	"github.com/google/mangle/engine"
	"github.com/google/mangle/factstore"
	"github.com/google/mangle/parse"
)

// MangleEnvironment implements ExecutionEnvironment using Google's Mangle Datalog engine.
type MangleEnvironment struct {
	mu          sync.RWMutex
	facts       []ast.Clause // Loaded facts
	finalAnswer string
	hasFinal    bool
	contextInfo string
}

// Ensure MangleEnvironment implements ExecutionEnvironment.
var _ ExecutionEnvironment = (*MangleEnvironment)(nil)

// NewMangleEnvironment creates a new Mangle-based execution environment.
func NewMangleEnvironment() *MangleEnvironment {
	return &MangleEnvironment{
		facts: make([]ast.Clause, 0),
	}
}

// LoadContext converts the context payload into Mangle facts.
func (e *MangleEnvironment) LoadContext(contextPayload any) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	// Reset existing facts
	e.facts = make([]ast.Clause, 0)
	e.contextInfo = ""

	if contextPayload == nil {
		return nil
	}

	// Check explicit CodeContext
	if cc, ok := contextPayload.(CodeContext); ok {
		facts, info, err := ParseCode(context.Background(), cc.Code, cc.Language)
		if err != nil {
			return fmt.Errorf("failed to parse code: %w", err)
		}
		e.facts = facts
		e.contextInfo = info
		return nil
	}
	if cc, ok := contextPayload.(*CodeContext); ok {
		if cc == nil {
			return nil
		}
		facts, info, err := ParseCode(context.Background(), cc.Code, cc.Language)
		if err != nil {
			return fmt.Errorf("failed to parse code: %w", err)
		}
		e.facts = facts
		e.contextInfo = info
		return nil
	}

	// Use FactWalker for generic data
	walker := NewFactWalker()
	facts, info, err := walker.Walk(contextPayload)
	if err != nil {
		return fmt.Errorf("failed to walk context data: %w", err)
	}

	e.facts = facts
	e.contextInfo = info
	return nil
}

// Execute runs a Datalog query against the loaded facts.
func (e *MangleEnvironment) Execute(ctx context.Context, code string) (ExecutionResult, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	start := time.Now()

	// Parse the code (rules and query)
	source, err := parse.Unit(strings.NewReader(code))
	if err != nil {
		return ExecutionResult{
			Stderr:   fmt.Sprintf("Parse error: %v", err),
			Duration: time.Since(start),
		}, nil
	}

	// Create a store and load existing facts
	store := factstore.NewSimpleInMemoryStore()
	for _, clause := range e.facts {
		store.Add(clause.Head)
	}

	// Run inference
	err = engine.EvalProgramNaive(source.Clauses, store)
	if err != nil {
		return ExecutionResult{
			Stderr:   fmt.Sprintf("Execution error: %v", err),
			Duration: time.Since(start),
		}, nil
	}

	// Collect results from 'result(X)' predicate
	// We assume the user follows the prompt convention of defining result(X).
	// We construct a query atom result(X).
	queryFact, _ := parse.Unit(strings.NewReader("result(X)."))
	queryAtom := queryFact.Clauses[0].Head

	var results []string
	err = store.GetFacts(queryAtom, func(a ast.Atom) error {
		// Convert atom to string
		results = append(results, a.String())
		return nil
	})

	output := "No results found for predicate 'result(X)'."
	if len(results) > 0 {
		output = strings.Join(results, "\n")
	}

	if err != nil {
		return ExecutionResult{
			Stdout:   output,
			Stderr:   fmt.Sprintf("Result retrieval error: %v", err),
			Duration: time.Since(start),
		}, nil
	}

	return ExecutionResult{
		Stdout:   output,
		Stderr:   "",
		Duration: time.Since(start),
	}, nil
}

// GetContextInfo returns a summary of the loaded context.
func (e *MangleEnvironment) GetContextInfo() string {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.contextInfo
}

// GetState returns a string representation of the current environment state (facts).
func (e *MangleEnvironment) GetState() string {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return fmt.Sprintf("Mangle Datalog State: %d facts loaded.", len(e.facts))
}

// GetVariable retrieves the string value of a variable by name.
// For Mangle, this is not directly supported as it's logic-based, but we could support looking up 'result' relation.
func (e *MangleEnvironment) GetVariable(name string) (string, error) {
	return "", fmt.Errorf("GetVariable not supported in MangleEnvironment")
}

// SetVariable sets a variable in the environment.
func (e *MangleEnvironment) SetVariable(name, value string) error {
	return fmt.Errorf("SetVariable not supported in MangleEnvironment")
}

// GetLLMCalls returns a list of LLM calls made during execution (if any).
func (e *MangleEnvironment) GetLLMCalls() []LLMCall {
	return nil // Mangle currently does not support intrinsic LLM calls
}

// ClearFinal clears the final answer state.
func (e *MangleEnvironment) ClearFinal() {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.finalAnswer = ""
	e.hasFinal = false
}

// HasFinal checks if a final answer has been set.
func (e *MangleEnvironment) HasFinal() bool {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.hasFinal
}

// Final returns the final answer.
func (e *MangleEnvironment) Final() string {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.finalAnswer
}

// RegisterFunction registers a custom function in the environment.
// Mangle environment does not yet support custom functions.
func (e *MangleEnvironment) RegisterFunction(name string, fn any) error {
	return fmt.Errorf("RegisterFunction not supported in MangleEnvironment")
}

// --- FactWalker Implementation ---

// FactWalker converts arbitrary Go data structures into valid Mangle Datalog facts.
type FactWalker struct {
	nextID int
	facts  []ast.Clause
}

func NewFactWalker() *FactWalker {
	return &FactWalker{
		nextID: 1,
		facts:  make([]ast.Clause, 0),
	}
}

// Walk traverses the data and returns a list of Mangle clauses (facts).
func (w *FactWalker) Walk(data any) ([]ast.Clause, string, error) {
	if data == nil {
		return nil, "No data loaded.", nil
	}

	w.facts = make([]ast.Clause, 0)
	w.nextID = 1

	rootID := w.generateID()
	rootConst := ast.Constant(ast.String(rootID))

	// Create a root fact: root(id)
	rootAtom := ast.NewAtom("root", rootConst)
	rootClause := ast.NewClause(rootAtom, nil)
	w.facts = append(w.facts, rootClause)

	if err := w.walkRecursive(reflect.ValueOf(data), rootConst); err != nil {
		return nil, "", err
	}

	// Collect predicates for context info
	predicates := make(map[string]bool)
	for _, clause := range w.facts {
		predicates[clause.Head.Predicate.Symbol] = true
	}
	var predList []string
	for p := range predicates {
		predList = append(predList, p)
	}

	info := fmt.Sprintf("Loaded %d facts. Predicates: %s", len(w.facts), strings.Join(predList, ", "))
	return w.facts, info, nil
}

func (w *FactWalker) generateID() string {
	id := fmt.Sprintf("n%d", w.nextID)
	w.nextID++
	return id
}

func (w *FactWalker) walkRecursive(v reflect.Value, parentID ast.Constant) error {
	// Handle pointers and interfaces
	if v.Kind() == reflect.Ptr || v.Kind() == reflect.Interface {
		if v.IsNil() {
			return nil
		}
		return w.walkRecursive(v.Elem(), parentID)
	}

	switch v.Kind() {
	case reflect.Map:
		iter := v.MapRange()
		for iter.Next() {
			key := iter.Key()
			val := iter.Value()

			keyStr := fmt.Sprintf("%v", key.Interface())
			predicateName := sanitizePredicate(keyStr)

			if isPrimitive(val) {
				constVal := toMangleConstant(val)
				atom := ast.NewAtom(predicateName, parentID, constVal)
				clause := ast.NewClause(atom, nil)
				w.facts = append(w.facts, clause)
			} else {
				childID := w.generateID()
				childConst := ast.Constant(ast.String(childID))

				atom := ast.NewAtom(predicateName, parentID, childConst)
				clause := ast.NewClause(atom, nil)
				w.facts = append(w.facts, clause)

				if err := w.walkRecursive(val, childConst); err != nil {
					return err
				}
			}
		}

	case reflect.Struct:
		t := v.Type()
		for i := 0; i < v.NumField(); i++ {
			field := t.Field(i)
			val := v.Field(i)

			if !field.IsExported() {
				continue
			}

			predicateName := sanitizePredicate(field.Name)

			if isPrimitive(val) {
				constVal := toMangleConstant(val)
				atom := ast.NewAtom(predicateName, parentID, constVal)
				clause := ast.NewClause(atom, nil)
				w.facts = append(w.facts, clause)
			} else {
				childID := w.generateID()
				childConst := ast.Constant(ast.String(childID))

				atom := ast.NewAtom(predicateName, parentID, childConst)
				clause := ast.NewClause(atom, nil)
				w.facts = append(w.facts, clause)

				if err := w.walkRecursive(val, childConst); err != nil {
					return err
				}
			}
		}

	case reflect.Slice, reflect.Array:
		for i := 0; i < v.Len(); i++ {
			val := v.Index(i)

			indexConst := ast.Constant(ast.Number(int64(i)))

			if isPrimitive(val) {
				constVal := toMangleConstant(val)
				atom := ast.NewAtom("has_item", parentID, indexConst, constVal)
				clause := ast.NewClause(atom, nil)
				w.facts = append(w.facts, clause)
			} else {
				childID := w.generateID()
				childConst := ast.Constant(ast.String(childID))

				atom := ast.NewAtom("has_item", parentID, indexConst, childConst)
				clause := ast.NewClause(atom, nil)
				w.facts = append(w.facts, clause)

				if err := w.walkRecursive(val, childConst); err != nil {
					return err
				}
			}
		}

	default:
	}

	return nil
}

func isPrimitive(v reflect.Value) bool {
	v = reflect.Indirect(v)
	switch v.Kind() {
	case reflect.Bool, reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64,
		reflect.Float32, reflect.Float64, reflect.String:
		return true
	default:
		return false
	}
}

func toMangleConstant(v reflect.Value) ast.Constant {
	v = reflect.Indirect(v)
	switch v.Kind() {
	case reflect.String:
		return ast.Constant(ast.String(v.String()))
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return ast.Constant(ast.Number(v.Int()))
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return ast.Constant(ast.Number(int64(v.Uint())))
	case reflect.Float32, reflect.Float64:
		return ast.Constant(ast.String(fmt.Sprintf("%f", v.Float())))
	case reflect.Bool:
		if v.Bool() {
			return ast.Constant(ast.String("true"))
		}
		return ast.Constant(ast.String("false"))
	default:
		return ast.Constant(ast.String(fmt.Sprintf("%v", v)))
	}
}

func sanitizePredicate(s string) string {
	s = strings.ToLower(s)
	// Replace non-alphanumeric with underscore
	var sb strings.Builder
	for _, r := range s {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '_' {
			sb.WriteRune(r)
		} else {
			sb.WriteRune('_')
		}
	}
	res := sb.String()
	if res == "" || (res[0] >= '0' && res[0] <= '9') {
		res = "prop_" + res
	}
	return res
}
