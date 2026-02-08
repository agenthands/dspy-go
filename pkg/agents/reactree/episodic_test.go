package reactree

import "testing"

func TestEpisodicMemory(t *testing.T) {
	em := NewEpisodicMemory()
	
	em.Add("search", "Example: search query -> result")
	
	// Exact match
	val := em.Get("search")
	if val != "Example: search query -> result" {
		t.Errorf("Expected exact match, got %v", val)
	}
	
	// Partial match (key contains stored key)
	val2 := em.Get("I want to search something")
	if val2 != "Example: search query -> result" {
		t.Errorf("Expected partial match, got %v", val2)
	}
	
	// No match
	val3 := em.Get("random task")
	if val3 != "" {
		t.Errorf("Expected empty for no match, got %v", val3)
	}
}
