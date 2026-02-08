package reactree

import (
	"strings"
	"sync"
)

// EpisodicMemoryImpl implements Memory for retrieving few-shot examples.
// For now, it's a simple exact match or substring match store.
// In production, this would use vector embeddings.
type EpisodicMemoryImpl struct {
	mu       sync.RWMutex
	examples map[string]string // Key: Goal, Value: Example text
}

func NewEpisodicMemory() *EpisodicMemoryImpl {
	return &EpisodicMemoryImpl{
		examples: make(map[string]string),
	}
}

func (em *EpisodicMemoryImpl) Add(key string, value interface{}) {
	em.mu.Lock()
	defer em.mu.Unlock()
	if str, ok := value.(string); ok {
		em.examples[key] = str
	}
}

func (em *EpisodicMemoryImpl) Get(key string) interface{} {
	em.mu.RLock()
	defer em.mu.RUnlock()
	
	// Exact match first
	if val, ok := em.examples[key]; ok {
		return val
	}
	
	// Fallback: search for substring?
	// Or maybe just return empty if not found.
	// For ReAcTree MVP, we can assume pre-loaded examples for specific goals.
	
	// Let's try to find ANY example that contains the key (naive retrieval)
	// Or where key contains example key?
	for k, v := range em.examples {
		if strings.Contains(key, k) {
			return v
		}
	}
	
	return ""
}
