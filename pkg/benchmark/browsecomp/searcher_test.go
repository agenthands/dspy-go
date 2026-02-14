package browsecomp

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestNewEmbeddedSearcher_BuildsIndex(t *testing.T) {
	tmpDir := t.TempDir()
	datasetPath := filepath.Join(tmpDir, "test_dataset.jsonl")
	indexPath := filepath.Join(tmpDir, "test.bleve")

	data := `{"query_id":"1","query":"capital of France","answer":"Paris","gold_docs":[{"docid":"doc_1","text":"Paris is the capital of France and its largest city."},{"docid":"doc_2","text":"The Eiffel Tower is located in Paris, France."}],"negative_docs":[{"docid":"doc_neg_1","text":"Berlin is the capital of Germany."}]}
{"query_id":"2","query":"author of Hamlet","answer":"Shakespeare","gold_docs":[{"docid":"doc_3","text":"Hamlet was written by William Shakespeare in around 1600."}],"negative_docs":[{"docid":"doc_neg_2","text":"Cervantes wrote Don Quixote."}]}
`
	if err := os.WriteFile(datasetPath, []byte(data), 0644); err != nil {
		t.Fatalf("failed to write dataset: %v", err)
	}

	searcher, err := NewEmbeddedSearcher(indexPath, datasetPath)
	if err != nil {
		t.Fatalf("NewEmbeddedSearcher failed: %v", err)
	}
	defer searcher.Close()

	if _, err := os.Stat(indexPath); os.IsNotExist(err) {
		t.Error("expected index directory to be created")
	}
}

func TestNewEmbeddedSearcher_OpensExistingIndex(t *testing.T) {
	tmpDir := t.TempDir()
	datasetPath := filepath.Join(tmpDir, "test_dataset.jsonl")
	indexPath := filepath.Join(tmpDir, "test.bleve")

	data := `{"query_id":"1","query":"test","answer":"test","gold_docs":[{"docid":"doc_x","text":"test doc"}],"negative_docs":[]}
`
	if err := os.WriteFile(datasetPath, []byte(data), 0644); err != nil {
		t.Fatalf("failed to write dataset: %v", err)
	}

	s1, err := NewEmbeddedSearcher(indexPath, datasetPath)
	if err != nil {
		t.Fatalf("first NewEmbeddedSearcher failed: %v", err)
	}
	s1.Close()

	s2, err := NewEmbeddedSearcher(indexPath, "")
	if err != nil {
		t.Fatalf("second NewEmbeddedSearcher failed: %v", err)
	}
	defer s2.Close()
}

func TestNewEmbeddedSearcher_ErrorNoIndexNoData(t *testing.T) {
	tmpDir := t.TempDir()
	indexPath := filepath.Join(tmpDir, "nonexistent.bleve")

	_, err := NewEmbeddedSearcher(indexPath, "")
	if err == nil {
		t.Fatal("expected error when no index and no dataset, got nil")
	}
}

func TestNewEmbeddedSearcher_ErrorBadDataset(t *testing.T) {
	tmpDir := t.TempDir()
	indexPath := filepath.Join(tmpDir, "test.bleve")
	// Dataset path that doesn't exist
	_, err := NewEmbeddedSearcher(indexPath, "/nonexistent/path/to/data.jsonl")
	if err == nil {
		t.Fatal("expected error for non-existent dataset path, got nil")
	}
	if !strings.Contains(err.Error(), "failed to build index") {
		t.Errorf("expected 'failed to build index' in error, got: %v", err)
	}
}

func TestNewEmbeddedSearcher_ErrorCorruptIndex(t *testing.T) {
	tmpDir := t.TempDir()
	indexPath := filepath.Join(tmpDir, "corrupt.bleve")

	// Create a directory that looks like an index but isn't valid
	if err := os.MkdirAll(indexPath, 0755); err != nil {
		t.Fatalf("failed to create fake index dir: %v", err)
	}

	_, err := NewEmbeddedSearcher(indexPath, "")
	if err == nil {
		t.Fatal("expected error for corrupt index, got nil")
	}
	if !strings.Contains(err.Error(), "failed to open index") {
		t.Errorf("expected 'failed to open index' in error, got: %v", err)
	}
}

func TestSearch(t *testing.T) {
	tmpDir := t.TempDir()
	datasetPath := filepath.Join(tmpDir, "test_dataset.jsonl")
	indexPath := filepath.Join(tmpDir, "test.bleve")

	data := `{"query_id":"1","query":"capital","answer":"Paris","gold_docs":[{"docid":"doc_paris","text":"Paris is the capital of France and is known for the Eiffel Tower."},{"docid":"doc_london","text":"London is the capital of the United Kingdom."}],"negative_docs":[{"docid":"doc_berlin","text":"Berlin is the capital of Germany and a major European city."}]}
`
	if err := os.WriteFile(datasetPath, []byte(data), 0644); err != nil {
		t.Fatalf("failed to write dataset: %v", err)
	}

	searcher, err := NewEmbeddedSearcher(indexPath, datasetPath)
	if err != nil {
		t.Fatalf("NewEmbeddedSearcher failed: %v", err)
	}
	defer searcher.Close()

	result := searcher.Search("Paris capital France")
	if result == "No results found." {
		t.Error("expected search results, got 'No results found.'")
	}
	if result == "" {
		t.Error("expected non-empty search result")
	}

	if searcher.ToolCallCounts["Search"] != 1 {
		t.Errorf("ToolCallCounts[Search] = %d, want 1", searcher.ToolCallCounts["Search"])
	}

	searcher.Search("London")
	if searcher.ToolCallCounts["Search"] != 2 {
		t.Errorf("ToolCallCounts[Search] = %d, want 2", searcher.ToolCallCounts["Search"])
	}
}

func TestSearch_QueryStringFallback(t *testing.T) {
	tmpDir := t.TempDir()
	datasetPath := filepath.Join(tmpDir, "test_dataset.jsonl")
	indexPath := filepath.Join(tmpDir, "test.bleve")

	data := `{"query_id":"1","query":"test","answer":"test","gold_docs":[{"docid":"doc_1","text":"The quick brown fox jumps over the lazy dog."}],"negative_docs":[]}
`
	if err := os.WriteFile(datasetPath, []byte(data), 0644); err != nil {
		t.Fatalf("failed to write dataset: %v", err)
	}

	searcher, err := NewEmbeddedSearcher(indexPath, datasetPath)
	if err != nil {
		t.Fatalf("NewEmbeddedSearcher failed: %v", err)
	}
	defer searcher.Close()

	// Use bleve query string syntax that causes a parse error to trigger fallback
	// An unclosed quote in QueryStringQuery should trigger the fallback to MatchQuery
	result := searcher.Search("fox \"unclosed quote")
	// Should still return results via fallback MatchQuery
	if searcher.ToolCallCounts["Search"] != 1 {
		t.Errorf("ToolCallCounts[Search] = %d, want 1", searcher.ToolCallCounts["Search"])
	}
	// The result should not be an error (the fallback should work)
	if strings.HasPrefix(result, "Error calling search:") {
		t.Errorf("fallback should have succeeded, got error: %s", result)
	}
}

func TestSearch_TextTruncation(t *testing.T) {
	tmpDir := t.TempDir()
	datasetPath := filepath.Join(tmpDir, "test_dataset.jsonl")
	indexPath := filepath.Join(tmpDir, "test.bleve")

	// Create a document with text > 200 chars to trigger truncation
	longText := strings.Repeat("The quick brown fox jumps over the lazy dog. ", 10) // ~450 chars
	data := `{"query_id":"1","query":"test","answer":"test","gold_docs":[{"docid":"long_doc","text":"` + longText + `"}],"negative_docs":[]}
`
	if err := os.WriteFile(datasetPath, []byte(data), 0644); err != nil {
		t.Fatalf("failed to write dataset: %v", err)
	}

	searcher, err := NewEmbeddedSearcher(indexPath, datasetPath)
	if err != nil {
		t.Fatalf("NewEmbeddedSearcher failed: %v", err)
	}
	defer searcher.Close()

	result := searcher.Search("fox")
	if result == "No results found." {
		t.Fatal("expected search results for long doc")
	}
	// The search result snippet should contain "..." indicating truncation
	if !strings.Contains(result, "...") {
		t.Error("expected truncated text with '...' for document with >200 chars")
	}
}

func TestSearch_NoResults(t *testing.T) {
	tmpDir := t.TempDir()
	datasetPath := filepath.Join(tmpDir, "test_dataset.jsonl")
	indexPath := filepath.Join(tmpDir, "test.bleve")

	data := `{"query_id":"1","query":"test","answer":"test","gold_docs":[{"docid":"doc_1","text":"Some text about cats and dogs."}],"negative_docs":[]}
`
	if err := os.WriteFile(datasetPath, []byte(data), 0644); err != nil {
		t.Fatalf("failed to write dataset: %v", err)
	}

	searcher, err := NewEmbeddedSearcher(indexPath, datasetPath)
	if err != nil {
		t.Fatalf("NewEmbeddedSearcher failed: %v", err)
	}
	defer searcher.Close()

	result := searcher.Search("xyzzyplughtwisty")
	if result != "No results found." {
		t.Errorf("expected 'No results found.', got: %s", result)
	}
}

func TestGetDocument(t *testing.T) {
	tmpDir := t.TempDir()
	datasetPath := filepath.Join(tmpDir, "test_dataset.jsonl")
	indexPath := filepath.Join(tmpDir, "test.bleve")

	data := `{"query_id":"1","query":"test","answer":"test","gold_docs":[{"docid":"doc_42","text":"Document forty-two content here."}],"negative_docs":[]}
`
	if err := os.WriteFile(datasetPath, []byte(data), 0644); err != nil {
		t.Fatalf("failed to write dataset: %v", err)
	}

	searcher, err := NewEmbeddedSearcher(indexPath, datasetPath)
	if err != nil {
		t.Fatalf("NewEmbeddedSearcher failed: %v", err)
	}
	defer searcher.Close()

	result := searcher.GetDocument("doc_42")
	if result == "Document not found." {
		t.Error("expected document content, got 'Document not found.'")
	}
	if !strings.Contains(result, "DocID: doc_42") {
		t.Errorf("expected 'DocID: doc_42' in result, got: %s", result)
	}

	if searcher.ToolCallCounts["GetDocument"] != 1 {
		t.Errorf("ToolCallCounts[GetDocument] = %d, want 1", searcher.ToolCallCounts["GetDocument"])
	}

	// Verify docID was tracked
	if !searcher.RetrievedDocIDs["doc_42"] {
		t.Error("expected doc_42 in RetrievedDocIDs")
	}
}

func TestGetDocument_NotFound(t *testing.T) {
	tmpDir := t.TempDir()
	datasetPath := filepath.Join(tmpDir, "test_dataset.jsonl")
	indexPath := filepath.Join(tmpDir, "test.bleve")

	data := `{"query_id":"1","query":"test","answer":"test","gold_docs":[{"docid":"doc_1","text":"Some content."}],"negative_docs":[]}
`
	if err := os.WriteFile(datasetPath, []byte(data), 0644); err != nil {
		t.Fatalf("failed to write dataset: %v", err)
	}

	searcher, err := NewEmbeddedSearcher(indexPath, datasetPath)
	if err != nil {
		t.Fatalf("NewEmbeddedSearcher failed: %v", err)
	}
	defer searcher.Close()

	result := searcher.GetDocument("nonexistent_doc_id")
	if result != "Document not found." {
		t.Errorf("expected 'Document not found.', got: %s", result)
	}
}

func TestReset(t *testing.T) {
	tmpDir := t.TempDir()
	datasetPath := filepath.Join(tmpDir, "test_dataset.jsonl")
	indexPath := filepath.Join(tmpDir, "test.bleve")

	data := `{"query_id":"1","query":"test","answer":"test","gold_docs":[{"docid":"doc_1","text":"Hello world content for testing."}],"negative_docs":[]}
`
	if err := os.WriteFile(datasetPath, []byte(data), 0644); err != nil {
		t.Fatalf("failed to write dataset: %v", err)
	}

	searcher, err := NewEmbeddedSearcher(indexPath, datasetPath)
	if err != nil {
		t.Fatalf("NewEmbeddedSearcher failed: %v", err)
	}
	defer searcher.Close()

	searcher.Search("hello")
	searcher.GetDocument("doc_1")

	if searcher.ToolCallCounts["Search"] != 1 {
		t.Errorf("pre-reset Search count = %d, want 1", searcher.ToolCallCounts["Search"])
	}

	searcher.Reset()

	if len(searcher.ToolCallCounts) != 0 {
		t.Errorf("post-reset ToolCallCounts should be empty, got: %v", searcher.ToolCallCounts)
	}
	if len(searcher.RetrievedDocIDs) != 0 {
		t.Errorf("post-reset RetrievedDocIDs should be empty, got: %v", searcher.RetrievedDocIDs)
	}
}

func TestClose_NilIndex(t *testing.T) {
	// Test that Close doesn't panic when Index is nil
	searcher := &EmbeddedSearcher{
		Index:           nil,
		ToolCallCounts:  make(map[string]int),
		RetrievedDocIDs: make(map[string]bool),
	}
	// Should NOT panic
	searcher.Close()
}

func TestTools_BothToolsCalled(t *testing.T) {
	tmpDir := t.TempDir()
	datasetPath := filepath.Join(tmpDir, "test_dataset.jsonl")
	indexPath := filepath.Join(tmpDir, "test.bleve")

	data := `{"query_id":"1","query":"test","answer":"test","gold_docs":[{"docid":"doc_1","text":"Content for tools test."}],"negative_docs":[]}
`
	if err := os.WriteFile(datasetPath, []byte(data), 0644); err != nil {
		t.Fatalf("failed to write dataset: %v", err)
	}

	searcher, err := NewEmbeddedSearcher(indexPath, datasetPath)
	if err != nil {
		t.Fatalf("NewEmbeddedSearcher failed: %v", err)
	}
	defer searcher.Close()

	tools := searcher.Tools()

	// Verify both tools exist
	if _, ok := tools["Search"]; !ok {
		t.Error("missing 'Search' tool")
	}
	if _, ok := tools["GetDocument"]; !ok {
		t.Error("missing 'GetDocument' tool")
	}

	// Call Search via tools map
	searchFn := tools["Search"].(func(string) string)
	result := searchFn("content")
	if result == "" {
		t.Error("tools Search returned empty result")
	}

	if searcher.ToolCallCounts["Search"] != 1 {
		t.Errorf("ToolCallCounts[Search] after tools call = %d, want 1", searcher.ToolCallCounts["Search"])
	}

	// Call GetDocument via tools map (covers the GetDocument wrapper)
	getDocFn := tools["GetDocument"].(func(string) string)
	docResult := getDocFn("doc_1")
	if docResult == "" {
		t.Error("tools GetDocument returned empty result")
	}

	if searcher.ToolCallCounts["GetDocument"] != 1 {
		t.Errorf("ToolCallCounts[GetDocument] after tools call = %d, want 1", searcher.ToolCallCounts["GetDocument"])
	}
}

func TestGetStats(t *testing.T) {
	tmpDir := t.TempDir()
	datasetPath := filepath.Join(tmpDir, "test_dataset.jsonl")
	indexPath := filepath.Join(tmpDir, "test.bleve")

	data := `{"query_id":"1","query":"test","answer":"test","gold_docs":[{"docid":"doc_1","text":"Stats test content."},{"docid":"doc_2","text":"More stats test content."}],"negative_docs":[]}
`
	if err := os.WriteFile(datasetPath, []byte(data), 0644); err != nil {
		t.Fatalf("failed to write dataset: %v", err)
	}

	searcher, err := NewEmbeddedSearcher(indexPath, datasetPath)
	if err != nil {
		t.Fatalf("NewEmbeddedSearcher failed: %v", err)
	}
	defer searcher.Close()

	searcher.Search("stats test")
	searcher.Search("content")
	searcher.GetDocument("doc_1")

	counts, docIDs := searcher.GetStats()

	if counts["Search"] != 2 {
		t.Errorf("stats Search count = %d, want 2", counts["Search"])
	}
	if counts["GetDocument"] != 1 {
		t.Errorf("stats GetDocument count = %d, want 1", counts["GetDocument"])
	}

	// Verify that returned counts are a copy (not a reference)
	counts["Search"] = 999
	if searcher.ToolCallCounts["Search"] != 2 {
		t.Error("GetStats should return a copy, not a reference")
	}

	// docIDs should contain at least doc_1 (from GetDocument)
	found := false
	for _, id := range docIDs {
		if id == "doc_1" {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected doc_1 in retrieved doc IDs, got: %v", docIDs)
	}
}

func TestGetDocument_ClosedIndex(t *testing.T) {
	tmpDir := t.TempDir()
	datasetPath := filepath.Join(tmpDir, "test_dataset.jsonl")
	indexPath := filepath.Join(tmpDir, "test.bleve")

	data := `{"query_id":"1","query":"test","answer":"test","gold_docs":[{"docid":"doc_1","text":"Test content."}],"negative_docs":[]}
`
	if err := os.WriteFile(datasetPath, []byte(data), 0644); err != nil {
		t.Fatalf("failed to write dataset: %v", err)
	}

	searcher, err := NewEmbeddedSearcher(indexPath, datasetPath)
	if err != nil {
		t.Fatalf("NewEmbeddedSearcher failed: %v", err)
	}

	// Close the index to trigger error paths
	searcher.Index.Close()

	result := searcher.GetDocument("doc_1")
	if !strings.HasPrefix(result, "Error calling get_document:") {
		t.Errorf("expected error message for closed index, got: %s", result)
	}
}

func TestSearch_ClosedIndex(t *testing.T) {
	tmpDir := t.TempDir()
	datasetPath := filepath.Join(tmpDir, "test_dataset.jsonl")
	indexPath := filepath.Join(tmpDir, "test.bleve")

	data := `{"query_id":"1","query":"test","answer":"test","gold_docs":[{"docid":"doc_1","text":"Test content."}],"negative_docs":[]}
`
	if err := os.WriteFile(datasetPath, []byte(data), 0644); err != nil {
		t.Fatalf("failed to write dataset: %v", err)
	}

	searcher, err := NewEmbeddedSearcher(indexPath, datasetPath)
	if err != nil {
		t.Fatalf("NewEmbeddedSearcher failed: %v", err)
	}

	// Close the index to trigger both fallback error paths in Search
	searcher.Index.Close()

	result := searcher.Search("test query")
	// Both QueryStringQuery and MatchQuery should fail on closed index
	if !strings.HasPrefix(result, "Error calling search:") {
		t.Errorf("expected error message for closed index, got: %s", result)
	}
}
