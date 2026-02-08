package reactree

import (
	"sync"
	"testing"
)

func TestWorkingMemory_AddGet(t *testing.T) {
	wm := NewWorkingMemory()
	
	wm.AddObservation("key1", "value1")
	val := wm.GetObservation("key1")
	
	if val != "value1" {
		t.Errorf("Expected 'value1', got %v", val)
	}
	
	valMissing := wm.GetObservation("missing")
	if valMissing != nil {
		t.Errorf("Expected nil for missing key, got %v", valMissing)
	}
}

func TestWorkingMemory_Concurrency(t *testing.T) {
	wm := NewWorkingMemory()
	var wg sync.WaitGroup
	
	// Concurrent writes
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(val int) {
			defer wg.Done()
			wm.AddObservation("key", val)
		}(i)
	}
	
	// Concurrent reads
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = wm.GetObservation("key")
		}()
	}
	
	wg.Wait()
	
	// Just verify no panic and eventual consistency (last write wins, but we don't know which one)
	// Key should exist
	if wm.GetObservation("key") == nil {
		t.Error("Key should exist after concurrent writes")
	}
}

func TestWorkingMemory_GetAllObservations(t *testing.T) {
	wm := NewWorkingMemory()
	wm.AddObservation("a", 1)
	wm.AddObservation("b", 2)
	
	all := wm.GetAllObservations()
	if len(all) != 2 {
		t.Errorf("Expected 2 observations, got %d", len(all))
	}
	
	// Verify copy safety
	all["c"] = 3
	if wm.GetObservation("c") != nil {
		t.Error("GetAllObservations should return a copy")
	}
}

func TestWorkingMemory_Logs(t *testing.T) {
	wm := NewWorkingMemory()
	wm.AddObservation("a", 1)
	wm.AddObservation("b", 2)
	
	logs := wm.GetLogs()
	if len(logs) != 2 {
		t.Errorf("Expected 2 logs, got %d", len(logs))
	}
}
