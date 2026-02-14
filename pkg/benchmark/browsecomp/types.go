package browsecomp

import (
	"bufio"
	"encoding/json"
	"io"
	"os"
)

// BenchmarkRun represents the result of a single query execution.
type BenchmarkRun struct {
	QueryID         any            `json:"query_id"`
	Status          string         `json:"status"`
	Result          []OutputBlock  `json:"result"`
	ToolCallCounts  map[string]int `json:"tool_call_counts"`
	RetrievedDocIDs []string       `json:"retrieved_docids"`
	Metadata        map[string]any `json:"metadata"`
}

// OutputBlock represents a generic block of output from the model.
type OutputBlock struct {
	Type   string `json:"type"`
	Output string `json:"output"`
}

// DatasetRow represents a row in the benchmark dataset.
type DatasetRow struct {
	QueryID any    `json:"query_id"`
	Query   string `json:"query"`
	Answer  string `json:"answer"`
}

// Document represents a searchable document.
type Document struct {
	DocID string `json:"docid"`
	Text  string `json:"text"`
}

// DatasetRowWithDocs represents a row in the dataset that includes associated documents.
// This is used during index building.
type DatasetRowWithDocs struct {
	QueryID      any        `json:"query_id"`
	Query        string     `json:"query"`
	Answer       string     `json:"answer"`
	GoldDocs     []Document `json:"gold_docs"`
	NegativeDocs []Document `json:"negative_docs"`
}

// LoadDataset reads a JSONL file and returns all parsed DatasetRow entries.
// Each line in the file should be a valid JSON object with query_id, query, and answer fields.
// Lines that fail to parse are silently skipped.
func LoadDataset(path string) ([]DatasetRow, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	return LoadDatasetFromReader(file)
}

// LoadDatasetFromReader reads DatasetRow entries from an io.Reader containing JSONL data.
// Lines that fail to parse are silently skipped.
func LoadDatasetFromReader(r io.Reader) ([]DatasetRow, error) {
	scanner := bufio.NewScanner(r)
	buf := make([]byte, 0, 1024*1024)
	scanner.Buffer(buf, 20*1024*1024) // 20MB buffer for large lines

	var rows []DatasetRow
	for scanner.Scan() {
		var row DatasetRow
		if err := json.Unmarshal(scanner.Bytes(), &row); err != nil {
			continue // skip malformed lines
		}
		rows = append(rows, row)
	}
	if err := scanner.Err(); err != nil {
		return rows, err
	}
	return rows, nil
}

// ParseDatasetRow parses a raw JSON line into a DatasetRow.
func ParseDatasetRow(line []byte) (*DatasetRow, error) {
	var row DatasetRow
	if err := json.Unmarshal(line, &row); err != nil {
		return nil, err
	}
	return &row, nil
}
