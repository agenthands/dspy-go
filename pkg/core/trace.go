package core

import (
	"context"
	"sync"
)

// TraceEntry represents a recorded execution step of a module.
type TraceEntry struct {
	ModuleName string
	ModuleType string
	Inputs     map[string]interface{}
	Outputs    map[string]interface{}
	Err        error
}

// TraceInterceptor records module executions.
// It is thread-safe and can be shared across multiple executions if needed.
type TraceInterceptor struct {
	mu      sync.Mutex
	traces  []TraceEntry
	enabled bool
}

// NewTraceInterceptor creates a new TraceInterceptor.
func NewTraceInterceptor() *TraceInterceptor {
	return &TraceInterceptor{
		traces:  make([]TraceEntry, 0),
		enabled: true,
	}
}

// Intercept implements the ModuleInterceptor signature.
func (ti *TraceInterceptor) Intercept(ctx context.Context, inputs map[string]interface{}, info *ModuleInfo, handler ModuleHandler, opts ...Option) (map[string]interface{}, error) {
	if !ti.isEnabled() {
		return handler(ctx, inputs, opts...)
	}

	// Capture inputs (deep copy if necessary, but shallow for now for performance)
	// In a real system, we might want to deep copy to avoid mutation race conditions if inputs are modified.
	inputsCopy := make(map[string]interface{})
	for k, v := range inputs {
		inputsCopy[k] = v
	}

	res, err := handler(ctx, inputs, opts...)

	// Record the trace
	entry := TraceEntry{
		ModuleName: info.ModuleName,
		ModuleType: info.ModuleType,
		Inputs:     inputsCopy,
		// Deep copy outputs to be safe
		Outputs: nil,
		Err:     err,
	}

	if res != nil {
		outputsCopy := make(map[string]interface{})
		for k, v := range res {
			outputsCopy[k] = v
		}
		entry.Outputs = outputsCopy
	}

	ti.addTrace(entry)

	return res, err
}

func (ti *TraceInterceptor) addTrace(entry TraceEntry) {
	ti.mu.Lock()
	defer ti.mu.Unlock()
	ti.traces = append(ti.traces, entry)
}

func (ti *TraceInterceptor) isEnabled() bool {
	ti.mu.Lock()
	defer ti.mu.Unlock()
	return ti.enabled
}

// GetTraces returns a copy of the recorded traces.
func (ti *TraceInterceptor) GetTraces() []TraceEntry {
	ti.mu.Lock()
	defer ti.mu.Unlock()
	
	result := make([]TraceEntry, len(ti.traces))
	copy(result, ti.traces)
	return result
}

// Clear clears the recorded traces.
func (ti *TraceInterceptor) Clear() {
	ti.mu.Lock()
	defer ti.mu.Unlock()
	ti.traces = make([]TraceEntry, 0)
}
