package context_evolve

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestCommandEvaluator_Evaluate(t *testing.T) {
	// Create a temporary script that acts as the evaluator command
	// It reads the input file and prints a fixed JSON metric
	tempDir := t.TempDir()
	scriptPath := filepath.Join(tempDir, "mock_evaluator.sh")
	scriptContent := `#!/bin/sh
cat $1 > /dev/null # Read input to simulate usage
echo '{"score": 0.95, "success": true}'
`
	if err := os.WriteFile(scriptPath, []byte(scriptContent), 0755); err != nil {
		t.Fatalf("Failed to create mock script: %v", err)
	}

	evaluator := NewCommandEvaluator(scriptPath, 5)

	program := &Program{
		Content: "some content",
	}

	metrics, err := evaluator.Evaluate(context.Background(), program)
	if err != nil {
		t.Fatalf("Evaluate failed: %v", err)
	}

	if metrics["score"] != 0.95 {
		t.Errorf("Expected score 0.95, got %v", metrics["score"])
	}
	if metrics["success"] != true {
		t.Errorf("Expected success true, got %v", metrics["success"])
	}
}

func TestCommandEvaluator_Timeout(t *testing.T) {
	// Create a script that sleeps longer than timeout
	scriptContent := `#!/bin/sh
sleep 2
echo '{"score": 0.5}'
`
	tmpfile, err := os.CreateTemp("", "sleepy_evaluator_*.sh")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpfile.Name())

	if _, err := tmpfile.Write([]byte(scriptContent)); err != nil {
		t.Fatal(err)
	}
	if err := tmpfile.Close(); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(tmpfile.Name(), 0755); err != nil {
		t.Fatal(err)
	}

	evaluator := NewCommandEvaluator(tmpfile.Name(), 1) // 1 second timeout

	ctx := context.Background()
	_, err = evaluator.Evaluate(ctx, &Program{Content: "test"})
	if err == nil {
		t.Error("Expected timeout error, got nil")
	}
}

func TestProgramDatabase_Persistence(t *testing.T) {
	// Create temporary file for database
	tmpfile, err := os.CreateTemp("", "db_test_*.json")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpfile.Name())
	dbPath := tmpfile.Name()

	// 1. Create and populate database
	db := NewProgramDatabase(1)
	p1 := &Program{ID: "p1", Content: "content1", Metrics: map[string]interface{}{"score": 0.8}}
	p2 := &Program{ID: "p2", Content: "content2", Metrics: map[string]interface{}{"score": 0.9}}
	
	db.Add(p1, 0)
	db.Add(p2, 0)
	
	// Save
	if err := db.Save(dbPath); err != nil {
		t.Fatalf("Failed to save database: %v", err)
	}

	// 2. Load into new database instance
	loadedDB, err := LoadProgramDatabase(dbPath)
	if err != nil {
		t.Fatalf("Failed to load database: %v", err)
	}

	// Verify contents
	if len(loadedDB.Programs) != 2 {
		t.Errorf("Expected 2 programs, got %d", len(loadedDB.Programs))
	}

	loadedP1, exists := loadedDB.Get("p1")
	if !exists {
		t.Error("Program p1 not found in loaded DB")
	}
	if loadedP1.Content != "content1" {
		t.Errorf("Expected content1, got %s", loadedP1.Content)
	}
    
    // Check if best program was tracked (loadedDB should re-calculate or serialize best?)
    // Our implementation of LoadProgramDatabase serializes the map and islands, but maybe not 'best'.
    // Let's check LoadProgramDatabase implementation.
}
