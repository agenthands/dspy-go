package browsecomp

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/blevesearch/bleve/v2"
)

func TestBuildIndex_FromDatasetRowWithDocs(t *testing.T) {
	tmpDir := t.TempDir()
	datasetPath := filepath.Join(tmpDir, "test_dataset.jsonl")
	indexPath := filepath.Join(tmpDir, "test.bleve")

	data := `{"query_id":"1","query":"test query","answer":"test answer","gold_docs":[{"docid":"gold_1","text":"Gold document one."},{"docid":"gold_2","text":"Gold document two."}],"negative_docs":[{"docid":"neg_1","text":"Negative document one."}]}
{"query_id":"2","query":"another query","answer":"another answer","gold_docs":[{"docid":"gold_3","text":"Gold document three."}],"negative_docs":[{"docid":"neg_2","text":"Negative document two."}]}
`
	if err := os.WriteFile(datasetPath, []byte(data), 0644); err != nil {
		t.Fatalf("failed to write dataset: %v", err)
	}

	index, err := buildIndex(datasetPath, indexPath)
	if err != nil {
		t.Fatalf("buildIndex failed: %v", err)
	}
	defer index.Close()

	count, err := index.DocCount()
	if err != nil {
		t.Fatalf("DocCount failed: %v", err)
	}
	// 2 gold + 1 neg from first row, 1 gold + 1 neg from second row = 5
	if count != 5 {
		t.Errorf("DocCount = %d, want 5", count)
	}
}

func TestBuildIndex_Deduplication(t *testing.T) {
	tmpDir := t.TempDir()
	datasetPath := filepath.Join(tmpDir, "test_dataset.jsonl")
	indexPath := filepath.Join(tmpDir, "test.bleve")

	// Same doc_id appears in both rows — should only be indexed once
	data := `{"query_id":"1","query":"q1","answer":"a1","gold_docs":[{"docid":"shared_doc","text":"Shared document."},{"docid":"unique_1","text":"Unique one."}],"negative_docs":[]}
{"query_id":"2","query":"q2","answer":"a2","gold_docs":[{"docid":"shared_doc","text":"Shared document."},{"docid":"unique_2","text":"Unique two."}],"negative_docs":[]}
`
	if err := os.WriteFile(datasetPath, []byte(data), 0644); err != nil {
		t.Fatalf("failed to write dataset: %v", err)
	}

	index, err := buildIndex(datasetPath, indexPath)
	if err != nil {
		t.Fatalf("buildIndex failed: %v", err)
	}
	defer index.Close()

	count, err := index.DocCount()
	if err != nil {
		t.Fatalf("DocCount failed: %v", err)
	}
	if count != 3 {
		t.Errorf("DocCount = %d, want 3 (deduplication failed)", count)
	}
}

func TestBuildIndex_FlatDocumentParsesBehavior(t *testing.T) {
	tmpDir := t.TempDir()
	datasetPath := filepath.Join(tmpDir, "test_dataset.jsonl")
	indexPath := filepath.Join(tmpDir, "test.bleve")

	// Flat document format parses as DatasetRowWithDocs with empty arrays
	data := `{"docid":"flat_1","text":"Flat document one."}
{"docid":"flat_2","text":"Flat document two."}
`
	if err := os.WriteFile(datasetPath, []byte(data), 0644); err != nil {
		t.Fatalf("failed to write dataset: %v", err)
	}

	index, err := buildIndex(datasetPath, indexPath)
	if err != nil {
		t.Fatalf("buildIndex failed: %v", err)
	}
	defer index.Close()

	count, err := index.DocCount()
	if err != nil {
		t.Fatalf("DocCount failed: %v", err)
	}
	// These parse as valid DatasetRowWithDocs (empty gold/neg), so 0 docs indexed
	if count != 0 {
		t.Errorf("DocCount = %d, want 0 (flat docs parse as empty DatasetRowWithDocs)", count)
	}
}

func TestBuildIndex_EmptyDataset(t *testing.T) {
	tmpDir := t.TempDir()
	datasetPath := filepath.Join(tmpDir, "empty.jsonl")
	indexPath := filepath.Join(tmpDir, "test.bleve")

	if err := os.WriteFile(datasetPath, []byte(""), 0644); err != nil {
		t.Fatalf("failed to write dataset: %v", err)
	}

	index, err := buildIndex(datasetPath, indexPath)
	if err != nil {
		t.Fatalf("buildIndex failed on empty dataset: %v", err)
	}
	defer index.Close()

	count, err := index.DocCount()
	if err != nil {
		t.Fatalf("DocCount failed: %v", err)
	}
	if count != 0 {
		t.Errorf("DocCount = %d, want 0 for empty dataset", count)
	}
}

func TestBuildIndex_MixedFormats(t *testing.T) {
	tmpDir := t.TempDir()
	datasetPath := filepath.Join(tmpDir, "test_dataset.jsonl")
	indexPath := filepath.Join(tmpDir, "test.bleve")

	data := `{"query_id":"1","query":"q","answer":"a","gold_docs":[{"docid":"structured_1","text":"Structured doc."}],"negative_docs":[]}
{"docid":"flat_1","text":"Flat doc."}
not-valid-json-line
{"query_id":"2","query":"q2","answer":"a2","gold_docs":[{"docid":"structured_2","text":"Another structured doc."}],"negative_docs":[]}
`
	if err := os.WriteFile(datasetPath, []byte(data), 0644); err != nil {
		t.Fatalf("failed to write dataset: %v", err)
	}

	index, err := buildIndex(datasetPath, indexPath)
	if err != nil {
		t.Fatalf("buildIndex failed: %v", err)
	}
	defer index.Close()

	count, err := index.DocCount()
	if err != nil {
		t.Fatalf("DocCount failed: %v", err)
	}
	// structured_1 + structured_2 = 2
	if count != 2 {
		t.Errorf("DocCount = %d, want 2", count)
	}
}

func TestBuildIndex_DocumentFallbackPath(t *testing.T) {
	tmpDir := t.TempDir()
	datasetPath := filepath.Join(tmpDir, "test_dataset.jsonl")
	indexPath := filepath.Join(tmpDir, "test.bleve")

	// Lines that are valid JSON but NOT valid as DatasetRowWithDocs should
	// hit the Document fallback. We need lines that fail json.Unmarshal
	// for DatasetRowWithDocs. Since Go's JSON unmarshaler is lenient with
	// missing fields, we need truly invalid JSON for that path, but with
	// valid Document JSON. Use a JSON array which will fail struct unmarshal.
	//
	// Actually the fallback path is for when json.Unmarshal into DatasetRowWithDocs
	// fails. A JSON array `[...]` will fail to unmarshal into a struct.
	data := `[{"docid":"array_doc","text":"This is a JSON array line"}]
`
	if err := os.WriteFile(datasetPath, []byte(data), 0644); err != nil {
		t.Fatalf("failed to write dataset: %v", err)
	}

	index, err := buildIndex(datasetPath, indexPath)
	if err != nil {
		t.Fatalf("buildIndex failed: %v", err)
	}
	defer index.Close()

	count, err := index.DocCount()
	if err != nil {
		t.Fatalf("DocCount failed: %v", err)
	}
	// JSON array fails DatasetRowWithDocs parse, but also fails Document parse -> skip
	if count != 0 {
		t.Errorf("DocCount = %d, want 0", count)
	}
}

func TestBuildIndex_DocumentFallbackWithValidDoc(t *testing.T) {
	tmpDir := t.TempDir()
	datasetPath := filepath.Join(tmpDir, "test_dataset.jsonl")
	indexPath := filepath.Join(tmpDir, "test.bleve")

	// Create a line that:
	// 1. Fails json.Unmarshal into DatasetRowWithDocs (has gold_docs as string instead of array)
	// 2. Succeeds json.Unmarshal into Document (has docid and text)
	// This will trigger the fallback Document path.
	line1 := `{"docid":"fallback_doc_1","text":"Fallback document text","gold_docs":"invalid_not_array"}`

	if err := os.WriteFile(datasetPath, []byte(line1+"\n"), 0644); err != nil {
		t.Fatalf("failed to write dataset: %v", err)
	}

	index, err := buildIndex(datasetPath, indexPath)
	if err != nil {
		t.Fatalf("buildIndex failed: %v", err)
	}
	defer index.Close()

	count, err := index.DocCount()
	if err != nil {
		t.Fatalf("DocCount failed: %v", err)
	}
	if count != 1 {
		t.Errorf("DocCount = %d, want 1 (fallback Document path)", count)
	}
}

func TestBuildIndex_DocumentFallbackDedup(t *testing.T) {
	tmpDir := t.TempDir()
	datasetPath := filepath.Join(tmpDir, "test_dataset.jsonl")
	indexPath := filepath.Join(tmpDir, "test.bleve")

	// Two lines with same docid via fallback: should be deduped
	line := `{"docid":"dup_doc","text":"Some text","gold_docs":"invalid"}`
	data := line + "\n" + line + "\n"

	if err := os.WriteFile(datasetPath, []byte(data), 0644); err != nil {
		t.Fatalf("failed to write dataset: %v", err)
	}

	index, err := buildIndex(datasetPath, indexPath)
	if err != nil {
		t.Fatalf("buildIndex failed: %v", err)
	}
	defer index.Close()

	count, err := index.DocCount()
	if err != nil {
		t.Fatalf("DocCount failed: %v", err)
	}
	if count != 1 {
		t.Errorf("DocCount = %d, want 1 (dedup in fallback path)", count)
	}
}

func TestBuildIndex_DocumentFallbackEmptyDocID(t *testing.T) {
	tmpDir := t.TempDir()
	datasetPath := filepath.Join(tmpDir, "test_dataset.jsonl")
	indexPath := filepath.Join(tmpDir, "test.bleve")

	// Fallback Document with empty DocID should be skipped
	line := `{"docid":"","text":"No ID document","gold_docs":"invalid"}`
	if err := os.WriteFile(datasetPath, []byte(line+"\n"), 0644); err != nil {
		t.Fatalf("failed to write dataset: %v", err)
	}

	index, err := buildIndex(datasetPath, indexPath)
	if err != nil {
		t.Fatalf("buildIndex failed: %v", err)
	}
	defer index.Close()

	count, err := index.DocCount()
	if err != nil {
		t.Fatalf("DocCount failed: %v", err)
	}
	if count != 0 {
		t.Errorf("DocCount = %d, want 0 (empty docid skipped)", count)
	}
}

func TestBuildIndex_EmptyDocIDInGoldDocs(t *testing.T) {
	tmpDir := t.TempDir()
	datasetPath := filepath.Join(tmpDir, "test_dataset.jsonl")
	indexPath := filepath.Join(tmpDir, "test.bleve")

	// Gold docs and negative docs with empty docid should be skipped
	data := `{"query_id":"1","query":"q","answer":"a","gold_docs":[{"docid":"","text":"No ID gold doc"},{"docid":"valid_1","text":"Valid doc"}],"negative_docs":[{"docid":"","text":"No ID neg doc"}]}
`
	if err := os.WriteFile(datasetPath, []byte(data), 0644); err != nil {
		t.Fatalf("failed to write dataset: %v", err)
	}

	index, err := buildIndex(datasetPath, indexPath)
	if err != nil {
		t.Fatalf("buildIndex failed: %v", err)
	}
	defer index.Close()

	count, err := index.DocCount()
	if err != nil {
		t.Fatalf("DocCount failed: %v", err)
	}
	// Only valid_1 should be indexed
	if count != 1 {
		t.Errorf("DocCount = %d, want 1 (empty docid skipped)", count)
	}
}

func TestBuildIndex_BatchFlushAt1000(t *testing.T) {
	tmpDir := t.TempDir()
	datasetPath := filepath.Join(tmpDir, "test_dataset.jsonl")
	indexPath := filepath.Join(tmpDir, "test.bleve")

	// Generate a dataset with >1000 documents to trigger the batch flush at 1000
	f, err := os.Create(datasetPath)
	if err != nil {
		t.Fatalf("failed to create dataset: %v", err)
	}

	// Create rows with enough docs to exceed 1000
	// Each row has 50 gold docs, so 21 rows = 1050 docs
	for i := 0; i < 21; i++ {
		goldDocs := make([]Document, 50)
		for j := 0; j < 50; j++ {
			goldDocs[j] = Document{
				DocID: fmt.Sprintf("doc_%d_%d", i, j),
				Text:  fmt.Sprintf("Document %d-%d content for batch test", i, j),
			}
		}
		row := DatasetRowWithDocs{
			QueryID:      fmt.Sprintf("q_%d", i),
			Query:        fmt.Sprintf("query %d", i),
			Answer:       fmt.Sprintf("answer %d", i),
			GoldDocs:     goldDocs,
			NegativeDocs: nil,
		}
		data, _ := json.Marshal(row)
		f.Write(data)
		f.WriteString("\n")
	}
	f.Close()

	index, err := buildIndex(datasetPath, indexPath)
	if err != nil {
		t.Fatalf("buildIndex failed: %v", err)
	}
	defer index.Close()

	count, err := index.DocCount()
	if err != nil {
		t.Fatalf("DocCount failed: %v", err)
	}
	// 21 rows * 50 docs = 1050 unique docs
	if count != 1050 {
		t.Errorf("DocCount = %d, want 1050", count)
	}
}

func TestBuildIndex_FileOpenError(t *testing.T) {
	tmpDir := t.TempDir()
	indexPath := filepath.Join(tmpDir, "test.bleve")

	// Use a non-existent file path
	_, err := buildIndex("/nonexistent/path/data.jsonl", indexPath)
	if err == nil {
		t.Fatal("expected error for non-existent dataset file, got nil")
	}
}

func TestBuildIndex_IndexAlreadyExists(t *testing.T) {
	tmpDir := t.TempDir()
	datasetPath := filepath.Join(tmpDir, "test_dataset.jsonl")
	indexPath := filepath.Join(tmpDir, "test.bleve")

	data := `{"query_id":"1","query":"q","answer":"a","gold_docs":[{"docid":"d1","text":"text1"}],"negative_docs":[]}
`
	if err := os.WriteFile(datasetPath, []byte(data), 0644); err != nil {
		t.Fatalf("failed to write dataset: %v", err)
	}

	// Build index first time
	idx, err := buildIndex(datasetPath, indexPath)
	if err != nil {
		t.Fatalf("first buildIndex failed: %v", err)
	}
	idx.Close()

	// Try to build again at same path -> bleve.New should fail (index already exists)
	_, err = buildIndex(datasetPath, indexPath)
	if err == nil {
		t.Fatal("expected error when index already exists, got nil")
	}
}

func TestPopulateIndex_FileOpenError(t *testing.T) {
	tmpDir := t.TempDir()
	indexPath := filepath.Join(tmpDir, "test.bleve")

	// Create a real index to pass in
	mapping := bleve.NewIndexMapping()
	index, err := bleve.New(indexPath, mapping)
	if err != nil {
		t.Fatalf("bleve.New failed: %v", err)
	}
	defer index.Close()

	// Non-existent dataset file should trigger os.Open error
	err = populateIndex(index, "/nonexistent/path/data.jsonl")
	if err == nil {
		t.Fatal("expected error for non-existent dataset file, got nil")
	}
}

func TestPopulateIndex_ClosedIndex_FinalFlush(t *testing.T) {
	tmpDir := t.TempDir()
	datasetPath := filepath.Join(tmpDir, "test_dataset.jsonl")
	indexPath := filepath.Join(tmpDir, "test.bleve")

	// Write dataset with a few docs
	data := `{"query_id":"1","query":"q","answer":"a","gold_docs":[{"docid":"d1","text":"Some text"}],"negative_docs":[]}
`
	if err := os.WriteFile(datasetPath, []byte(data), 0644); err != nil {
		t.Fatalf("failed to write dataset: %v", err)
	}

	// Create and then close the index — the final flushBatch will fail
	mapping := bleve.NewIndexMapping()
	index, err := bleve.New(indexPath, mapping)
	if err != nil {
		t.Fatalf("bleve.New failed: %v", err)
	}
	index.Close()

	err = populateIndex(index, datasetPath)
	if err == nil {
		t.Fatal("expected error from populateIndex on closed index, got nil")
	}
}

func TestPopulateIndex_ClosedIndex_MidFlush(t *testing.T) {
	tmpDir := t.TempDir()
	datasetPath := filepath.Join(tmpDir, "test_dataset.jsonl")
	indexPath := filepath.Join(tmpDir, "test.bleve")

	// Generate a dataset with exactly 1000 docs in a single row to trigger mid-loop flush
	goldDocs := make([]Document, 1000)
	for i := 0; i < 1000; i++ {
		goldDocs[i] = Document{
			DocID: fmt.Sprintf("doc_%d", i),
			Text:  fmt.Sprintf("Document %d content", i),
		}
	}
	row := DatasetRowWithDocs{
		QueryID:      "batch_test",
		Query:        "batch query",
		Answer:       "batch answer",
		GoldDocs:     goldDocs,
		NegativeDocs: nil,
	}
	rowData, _ := json.Marshal(row)
	if err := os.WriteFile(datasetPath, append(rowData, '\n'), 0644); err != nil {
		t.Fatalf("failed to write dataset: %v", err)
	}

	// Create and close the index — the mid-loop flushBatch at 1000 docs will fail
	mapping := bleve.NewIndexMapping()
	index, err := bleve.New(indexPath, mapping)
	if err != nil {
		t.Fatalf("bleve.New failed: %v", err)
	}
	index.Close()

	err = populateIndex(index, datasetPath)
	if err == nil {
		t.Fatal("expected error from populateIndex mid-flush on closed index, got nil")
	}
}

func TestBuildIndex_PopulateError(t *testing.T) {
	tmpDir := t.TempDir()
	indexPath := filepath.Join(tmpDir, "test.bleve")

	// buildIndex with non-existent dataset should fail at populateIndex and clean up
	_, err := buildIndex("/nonexistent/path/data.jsonl", indexPath)
	if err == nil {
		t.Fatal("expected error for non-existent dataset file, got nil")
	}
}
