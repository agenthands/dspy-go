package browsecomp

import (
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoadDataset_Valid(t *testing.T) {
	tmpDir := t.TempDir()
	datasetPath := filepath.Join(tmpDir, "test.jsonl")

	data := `{"query_id":"1","query":"What is Go?","answer":"A programming language"}
{"query_id":"2","query":"Who made Linux?","answer":"Linus Torvalds"}
{"query_id":"3","query":"What is Rust?","answer":"A systems language"}
`
	if err := os.WriteFile(datasetPath, []byte(data), 0644); err != nil {
		t.Fatalf("failed to write dataset: %v", err)
	}

	rows, err := LoadDataset(datasetPath)
	if err != nil {
		t.Fatalf("LoadDataset failed: %v", err)
	}

	if len(rows) != 3 {
		t.Fatalf("expected 3 rows, got %d", len(rows))
	}
	if rows[0].QueryID != "1" {
		t.Errorf("rows[0].QueryID = %v, want 1", rows[0].QueryID)
	}
	if rows[1].Query != "Who made Linux?" {
		t.Errorf("rows[1].Query = %q, want 'Who made Linux?'", rows[1].Query)
	}
	if rows[2].Answer != "A systems language" {
		t.Errorf("rows[2].Answer = %q, want 'A systems language'", rows[2].Answer)
	}
}

func TestLoadDataset_SkipsMalformedLines(t *testing.T) {
	tmpDir := t.TempDir()
	datasetPath := filepath.Join(tmpDir, "test.jsonl")

	data := `{"query_id":"1","query":"Valid","answer":"Yes"}
this is not json
{"query_id":"2","query":"Also valid","answer":"Yes"}

{"query_id":"3","query":"Third","answer":"Yes"}
`
	if err := os.WriteFile(datasetPath, []byte(data), 0644); err != nil {
		t.Fatalf("failed to write dataset: %v", err)
	}

	rows, err := LoadDataset(datasetPath)
	if err != nil {
		t.Fatalf("LoadDataset failed: %v", err)
	}

	// 3 valid + 1 bad JSON + 1 empty line = 3 parsed
	if len(rows) != 3 {
		t.Errorf("expected 3 rows (skipping malformed), got %d", len(rows))
	}
}

func TestLoadDataset_EmptyFile(t *testing.T) {
	tmpDir := t.TempDir()
	datasetPath := filepath.Join(tmpDir, "empty.jsonl")

	if err := os.WriteFile(datasetPath, []byte(""), 0644); err != nil {
		t.Fatalf("failed to write dataset: %v", err)
	}

	rows, err := LoadDataset(datasetPath)
	if err != nil {
		t.Fatalf("LoadDataset failed: %v", err)
	}
	if len(rows) != 0 {
		t.Errorf("expected 0 rows for empty file, got %d", len(rows))
	}
}

func TestLoadDataset_FileNotFound(t *testing.T) {
	_, err := LoadDataset("/nonexistent/path/data.jsonl")
	if err == nil {
		t.Fatal("expected error for non-existent file, got nil")
	}
}

// errReader returns some data then an error to trigger scanner.Err()
type errReader struct {
	data    string
	pos     int
	errOnce bool
}

func (r *errReader) Read(p []byte) (n int, err error) {
	if r.pos >= len(r.data) {
		if !r.errOnce {
			r.errOnce = true
			return 0, errors.New("simulated read error")
		}
		return 0, io.EOF
	}
	n = copy(p, r.data[r.pos:])
	r.pos += n
	return n, nil
}

func TestLoadDatasetFromReader_Valid(t *testing.T) {
	data := `{"query_id":"1","query":"Test","answer":"Yes"}
{"query_id":"2","query":"Test2","answer":"No"}
`
	rows, err := LoadDatasetFromReader(strings.NewReader(data))
	if err != nil {
		t.Fatalf("LoadDatasetFromReader failed: %v", err)
	}
	if len(rows) != 2 {
		t.Errorf("expected 2 rows, got %d", len(rows))
	}
}

func TestLoadDatasetFromReader_ScannerError(t *testing.T) {
	// Provide some valid data then trigger an error mid-stream
	r := &errReader{
		data: `{"query_id":"1","query":"Before error","answer":"Yes"}
`,
	}

	rows, err := LoadDatasetFromReader(r)
	if err == nil {
		t.Fatal("expected scanner error, got nil")
	}
	// The rows parsed before the error should still be returned
	if len(rows) != 1 {
		t.Errorf("expected 1 row before error, got %d", len(rows))
	}
}

func TestParseDatasetRow(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		wantID  any
		wantQ   string
		wantA   string
		wantErr bool
	}{
		{
			name:   "valid string query_id",
			input:  `{"query_id":"769","query":"What is the capital?","answer":"Paris"}`,
			wantID: "769",
			wantQ:  "What is the capital?",
			wantA:  "Paris",
		},
		{
			name:   "valid numeric query_id",
			input:  `{"query_id":770,"query":"Who wrote Hamlet?","answer":"Shakespeare"}`,
			wantID: float64(770),
			wantQ:  "Who wrote Hamlet?",
			wantA:  "Shakespeare",
		},
		{
			name:    "invalid json",
			input:   `{broken json`,
			wantErr: true,
		},
		{
			name:   "empty fields",
			input:  `{"query_id":"","query":"","answer":""}`,
			wantID: "",
			wantQ:  "",
			wantA:  "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			row, err := ParseDatasetRow([]byte(tt.input))
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if row.QueryID != tt.wantID {
				t.Errorf("QueryID = %v, want %v", row.QueryID, tt.wantID)
			}
			if row.Query != tt.wantQ {
				t.Errorf("Query = %q, want %q", row.Query, tt.wantQ)
			}
			if row.Answer != tt.wantA {
				t.Errorf("Answer = %q, want %q", row.Answer, tt.wantA)
			}
		})
	}
}

func TestBenchmarkRunSerialization(t *testing.T) {
	run := BenchmarkRun{
		QueryID: "769",
		Status:  "completed",
		Result: []OutputBlock{
			{Type: "output_text", Output: "Paris"},
		},
		ToolCallCounts:  map[string]int{"Search": 3, "GetDocument": 1},
		RetrievedDocIDs: []string{"doc_1", "doc_2"},
		Metadata: map[string]any{
			"model":    "qwen3:8b",
			"duration": "1m30s",
		},
	}

	data, err := json.Marshal(run)
	if err != nil {
		t.Fatalf("failed to marshal: %v", err)
	}

	var decoded BenchmarkRun
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}

	if decoded.QueryID != "769" {
		t.Errorf("QueryID = %v, want 769", decoded.QueryID)
	}
	if decoded.Status != "completed" {
		t.Errorf("Status = %q, want completed", decoded.Status)
	}
	if len(decoded.Result) != 1 || decoded.Result[0].Output != "Paris" {
		t.Errorf("Result mismatch: %+v", decoded.Result)
	}
	if decoded.ToolCallCounts["Search"] != 3 {
		t.Errorf("ToolCallCounts[Search] = %d, want 3", decoded.ToolCallCounts["Search"])
	}
	if len(decoded.RetrievedDocIDs) != 2 {
		t.Errorf("RetrievedDocIDs length = %d, want 2", len(decoded.RetrievedDocIDs))
	}
}

func TestDatasetRowWithDocsSerialization(t *testing.T) {
	input := `{
		"query_id": "773",
		"query": "What color shirt?",
		"answer": "Red",
		"gold_docs": [
			{"docid": "38119", "text": "Some gold doc text"},
			{"docid": "80922", "text": "Another gold doc"}
		],
		"negative_docs": [
			{"docid": "99001", "text": "Negative doc text"}
		]
	}`

	var row DatasetRowWithDocs
	if err := json.Unmarshal([]byte(input), &row); err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}

	if row.QueryID != "773" {
		t.Errorf("QueryID = %v, want 773", row.QueryID)
	}
	if len(row.GoldDocs) != 2 {
		t.Errorf("GoldDocs length = %d, want 2", len(row.GoldDocs))
	}
	if row.GoldDocs[0].DocID != "38119" {
		t.Errorf("GoldDocs[0].DocID = %q, want 38119", row.GoldDocs[0].DocID)
	}
	if len(row.NegativeDocs) != 1 {
		t.Errorf("NegativeDocs length = %d, want 1", len(row.NegativeDocs))
	}
}

func TestDocumentSerialization(t *testing.T) {
	doc := Document{DocID: "12345", Text: "Hello world"}
	data, err := json.Marshal(doc)
	if err != nil {
		t.Fatalf("failed to marshal: %v", err)
	}

	var decoded Document
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}

	if decoded.DocID != "12345" {
		t.Errorf("DocID = %q, want 12345", decoded.DocID)
	}
	if decoded.Text != "Hello world" {
		t.Errorf("Text = %q, want 'Hello world'", decoded.Text)
	}
}

func TestOutputBlockSerialization(t *testing.T) {
	block := OutputBlock{Type: "output_text", Output: "test output"}
	data, err := json.Marshal(block)
	if err != nil {
		t.Fatalf("failed to marshal: %v", err)
	}

	var decoded OutputBlock
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}

	if decoded.Type != "output_text" {
		t.Errorf("Type = %q, want output_text", decoded.Type)
	}
	if decoded.Output != "test output" {
		t.Errorf("Output = %q, want 'test output'", decoded.Output)
	}
}

func TestDatasetRowSerialization(t *testing.T) {
	row := DatasetRow{QueryID: "42", Query: "test query", Answer: "test answer"}
	data, err := json.Marshal(row)
	if err != nil {
		t.Fatalf("failed to marshal: %v", err)
	}

	var decoded DatasetRow
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}

	if decoded.QueryID != "42" {
		t.Errorf("QueryID = %v, want 42", decoded.QueryID)
	}
}
