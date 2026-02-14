package browsecomp

import (
	"fmt"
	"log"
	"os"

	"github.com/blevesearch/bleve/v2"
	index_api "github.com/blevesearch/bleve_index_api"
)

// EmbeddedSearcher replaces MCPClient and provides embedded Bleve search logic.
type EmbeddedSearcher struct {
	Index           bleve.Index
	ToolCallCounts  map[string]int
	RetrievedDocIDs map[string]bool
}

// NewEmbeddedSearcher creates or opens a Bleve index at the specified path.
// If the index does not exist, it builds it from the provided datasetPath.
func NewEmbeddedSearcher(indexPath, datasetPath string) (*EmbeddedSearcher, error) {
	// Initialize Index
	var index bleve.Index
	if _, err := os.Stat(indexPath); os.IsNotExist(err) {
		if datasetPath == "" {
			return nil, fmt.Errorf("index not found and dataset path not provided")
		}
		log.Printf("Building index at %s from %s...", indexPath, datasetPath)
		index, err = buildIndex(datasetPath, indexPath)
		if err != nil {
			return nil, fmt.Errorf("failed to build index: %v", err)
		}
	} else {
		log.Printf("Opening existing index at %s...", indexPath)
		index, err = bleve.Open(indexPath)
		if err != nil {
			return nil, fmt.Errorf("failed to open index: %v", err)
		}
	}

	return &EmbeddedSearcher{
		Index:           index,
		ToolCallCounts:  make(map[string]int),
		RetrievedDocIDs: make(map[string]bool),
	}, nil
}

// Reset clears the internal counters and retrieved document history for a new query.
func (s *EmbeddedSearcher) Reset() {
	s.ToolCallCounts = make(map[string]int)
	s.RetrievedDocIDs = make(map[string]bool)
}

// Close closes the underlying Bleve index.
func (s *EmbeddedSearcher) Close() {
	if s.Index != nil {
		s.Index.Close()
	}
}

// Search executes a query against the index and returns the top 10 results.
// It tracks usage statistics.
func (s *EmbeddedSearcher) Search(query string) string {
	s.ToolCallCounts["Search"]++
	log.Printf("DEBUG: Executing Search with query: '%s'. Count so far: %d", query, s.ToolCallCounts["Search"])

	// Use QueryStringQuery for flexible operator support (OR default often better for long queries)
	// But it can fail on syntax errors. Fallback to MatchQuery.
	q := bleve.NewQueryStringQuery(query)
	// q.SetOperator(bleve.QueryStringQueryOperatorOr) // If supported, but might not be exposed easily without import.
	// Standard bleve QueryStringQuery defaults to OR for space-separated terms usually.

	req := bleve.NewSearchRequest(q)
	req.Size = 10
	req.Fields = []string{"text"}

	res, err := s.Index.Search(req)
	if err != nil {
		// Fallback to simple MatchQuery if syntax error
		log.Printf("DEBUG: QueryStringQuery failed (%v), falling back to MatchQuery", err)
		mq := bleve.NewMatchQuery(query)
		req = bleve.NewSearchRequest(mq)
		req.Size = 10
		req.Fields = []string{"text"}
		res, err = s.Index.Search(req)
		if err != nil {
			return fmt.Sprintf("Error calling search: %v", err)
		}
	}

	log.Printf("DEBUG: Search found %d hits", len(res.Hits))

	if len(res.Hits) == 0 {
		return "No results found."
	}

	var sb string
	for i, hit := range res.Hits {
		s.RetrievedDocIDs[hit.ID] = true
		text := ""
		if val, ok := hit.Fields["text"].(string); ok {
			if len(val) > 200 {
				text = val[:200] + "..."
			} else {
				text = val
			}
		}
		sb += fmt.Sprintf("[%d] DocID: %s\n%s\n\n", i+1, hit.ID, text)
	}
	return sb
}

// GetDocument retrieves the full text of a document by its ID.
// It tracks usage statistics.
func (s *EmbeddedSearcher) GetDocument(docid string) string {
	s.ToolCallCounts["GetDocument"]++

	doc, err := s.Index.Document(docid)
	if err != nil {
		return fmt.Sprintf("Error calling get_document: %v", err)
	}
	if doc == nil {
		return "Document not found."
	}
	s.RetrievedDocIDs[docid] = true

	// Reconstruct
	// Assuming text field name is "text"
	text := ""
	doc.VisitFields(func(field index_api.Field) {
		if field.Name() == "text" {
			text = string(field.Value())
		}
	})

	return fmt.Sprintf("DocID: %s\n%s", docid, text)
}

// Tools returns a map of tool functions suitable for injection into an RLM agent.
func (s *EmbeddedSearcher) Tools() map[string]any {
	return map[string]any{
		"Search": func(query string) string {
			return s.Search(query)
		},
		"GetDocument": func(docid string) string {
			return s.GetDocument(docid)
		},
	}
}

// GetStats returns the current tool usage counts and list of retrieved document IDs.
func (s *EmbeddedSearcher) GetStats() (map[string]int, []string) {
	counts := make(map[string]int)
	for k, v := range s.ToolCallCounts {
		counts[k] = v
	}
	var docIDs []string
	for k := range s.RetrievedDocIDs {
		docIDs = append(docIDs, k)
	}
	return counts, docIDs
}
