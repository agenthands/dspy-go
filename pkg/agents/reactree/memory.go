package reactree

import (
	"fmt"
	"sync"
)

// WorkingMemory is a thread-safe key-value store for storing observations and shared state.
type WorkingMemory struct {
	mu           sync.RWMutex
	observations map[string]interface{}
	logs         []string // Chronological log of actions/observations
}

// NewWorkingMemory creates a new WorkingMemory instance.
func NewWorkingMemory() *WorkingMemory {
	return &WorkingMemory{
		observations: make(map[string]interface{}),
		logs:         make([]string, 0),
	}
}

// AddObservation stores an observation and adds a log entry.
func (wm *WorkingMemory) AddObservation(key string, value interface{}) {
	wm.mu.Lock()
	defer wm.mu.Unlock()
	wm.observations[key] = value
	wm.logs = append(wm.logs, fmt.Sprintf("Observed [%s]: %v", key, value))
}

// GetObservation retrieves an observation by key.
func (wm *WorkingMemory) GetObservation(key string) interface{} {
	wm.mu.RLock()
	defer wm.mu.RUnlock()
	return wm.observations[key]
}

// GetAllObservations returns a copy of all observations.
func (wm *WorkingMemory) GetAllObservations() map[string]interface{} {
	wm.mu.RLock()
	defer wm.mu.RUnlock()
	
	copy := make(map[string]interface{})
	for k, v := range wm.observations {
		copy[k] = v
	}
	return copy
}

// GetLogs returns a copy of the chronological logs.
func (wm *WorkingMemory) GetLogs() []string {
	wm.mu.RLock()
	defer wm.mu.RUnlock()
	
	logsCopy := make([]string, len(wm.logs))
	copy(logsCopy, wm.logs)
	return logsCopy
}
