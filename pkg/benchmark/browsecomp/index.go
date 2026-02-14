package browsecomp

import (
	"bufio"
	"encoding/json"
	"log"
	"os"
	"time"

	"github.com/blevesearch/bleve/v2"
)

// buildIndex creates a new Bleve index from the specified dataset file.
func buildIndex(datasetPath, indexPath string) (bleve.Index, error) {
	mapping := bleve.NewIndexMapping()
	index, err := bleve.New(indexPath, mapping)
	if err != nil {
		return nil, err
	}

	if err := populateIndex(index, datasetPath); err != nil {
		index.Close()
		return nil, err
	}

	return index, nil
}

// populateIndex reads the dataset and indexes all documents into the given index.
// Separated from buildIndex for testability.
func populateIndex(index bleve.Index, datasetPath string) error {
	file, err := os.Open(datasetPath)
	if err != nil {
		return err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	buf := make([]byte, 0, 1024*1024)
	scanner.Buffer(buf, 10*1024*1024) // 10MB buffer

	batch := index.NewBatch()
	count := 0
	docCount := 0
	startTime := time.Now()

	// Keep track of indexed docIDs to avoid duplicates if they appear in multiple queries
	indexedDocs := make(map[string]bool)

	// flushBatch commits the current batch and resets it.
	flushBatch := func() error {
		if batch.Size() > 0 {
			if err := index.Batch(batch); err != nil {
				return err
			}
			batch = index.NewBatch()
		}
		return nil
	}

	for scanner.Scan() {
		var row DatasetRowWithDocs
		line := scanner.Bytes()
		// Try unmarshaling as a row with docs
		if err := json.Unmarshal(line, &row); err != nil {
			// Fallback: try to unmarshal as Document if Row fails or has no docs
			var doc Document
			if err2 := json.Unmarshal(line, &doc); err2 == nil && doc.DocID != "" {
				if !indexedDocs[doc.DocID] {
					batch.Index(doc.DocID, doc)
					indexedDocs[doc.DocID] = true
					docCount++
				}
				continue
			}
			continue
		}

		// Index Gold Docs
		for _, doc := range row.GoldDocs {
			if doc.DocID != "" && !indexedDocs[doc.DocID] {
				batch.Index(doc.DocID, doc)
				indexedDocs[doc.DocID] = true
				docCount++
			}
		}

		// Index Negative Docs
		for _, doc := range row.NegativeDocs {
			if doc.DocID != "" && !indexedDocs[doc.DocID] {
				batch.Index(doc.DocID, doc)
				indexedDocs[doc.DocID] = true
				docCount++
			}
		}

		if docCount > 0 && docCount%1000 == 0 {
			if err := flushBatch(); err != nil {
				return err
			}
			log.Printf("Indexed %d documents...", docCount)
		}
		count++
	}
	if err := flushBatch(); err != nil {
		return err
	}
	log.Printf("Finished indexing %d documents from %d rows in %v", docCount, count, time.Since(startTime))
	return scanner.Err()
}
