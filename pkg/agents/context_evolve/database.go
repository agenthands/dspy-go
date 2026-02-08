package context_evolve

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"sync"
)

// ProgramDatabase manages the population of programs.
// It maintains the population of programs and organizes them into "islands" for distributed evolution.
type ProgramDatabase struct {
	Programs      map[string]*Program `json:"programs"`
	Islands       map[int][]string    `json:"islands"`
	BestProgramID string              `json:"best_program_id"`
	mu            sync.RWMutex
}

// NewProgramDatabase creates a new ProgramDatabase.
func NewProgramDatabase(numIslands int) *ProgramDatabase {
	db := &ProgramDatabase{
		Programs: make(map[string]*Program),
		Islands:  make(map[int][]string),
	}
	for i := 0; i < numIslands; i++ {
		db.Islands[i] = []string{}
	}
	return db
}

// Add adds a program to the database and assigns it to an island.
func (db *ProgramDatabase) Add(program *Program, islandID int) {
	db.mu.Lock()
	defer db.mu.Unlock()

	db.Programs[program.ID] = program
	db.Islands[islandID] = append(db.Islands[islandID], program.ID)

	// Simple logic to declare best program based on "score" metric if available
	if db.BestProgramID == "" {
		db.BestProgramID = program.ID
	} else {
		currentBest := db.Programs[db.BestProgramID]
		if getScore(program) > getScore(currentBest) {
			db.BestProgramID = program.ID
		}
	}
}

// Get retrieves a program by ID.
func (db *ProgramDatabase) Get(id string) (*Program, bool) {
	db.mu.RLock()
	defer db.mu.RUnlock()
	p, ok := db.Programs[id]
	return p, ok
}

// GetIslandPrograms returns all programs in a specific island.
func (db *ProgramDatabase) GetIslandPrograms(islandID int) []*Program {
	db.mu.RLock()
	defer db.mu.RUnlock()

	ids, ok := db.Islands[islandID]
	if !ok {
		return nil
	}

	programs := make([]*Program, 0, len(ids))
	for _, id := range ids {
		if p, ok := db.Programs[id]; ok {
			programs = append(programs, p)
		}
	}
	return programs
}

// GetAllPrograms returns all programs in the database.
func (db *ProgramDatabase) GetAllPrograms() []*Program {
	db.mu.RLock()
	defer db.mu.RUnlock()

	programs := make([]*Program, 0, len(db.Programs))
	for _, p := range db.Programs {
		programs = append(programs, p)
	}
	return programs
}

// Save saves the database to a file.
func (db *ProgramDatabase) Save(filePath string) error {
	db.mu.RLock()
	defer db.mu.RUnlock()

	data, err := json.MarshalIndent(db, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal database: %w", err)
	}

	return os.WriteFile(filePath, data, 0644)
}

// Load loads the database from a file.
func LoadProgramDatabase(filePath string) (*ProgramDatabase, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read database file: %w", err)
	}

	var db ProgramDatabase
	if err := json.Unmarshal(data, &db); err != nil {
		return nil, fmt.Errorf("failed to unmarshal database: %w", err)
	}
    // Initialize mutex as it's not serialized
    db.mu = sync.RWMutex{}
	return &db, nil
}

// Helper to extract score safely
func getScore(p *Program) float64 {
	return p.GetScore()
}

// GetScore extracts the score from the program's metrics.
func (p *Program) GetScore() float64 {
	if val, ok := p.Metrics["score"]; ok {
		switch v := val.(type) {
		case float64:
			return v
		case int:
			return float64(v)
		}
	}
	return 0.0
}

// Migrate copies the top 'count' programs from sourceIsland to targetIsland.
func (db *ProgramDatabase) Migrate(sourceIsland, targetIsland int, count int) int {
	db.mu.Lock()
	defer db.mu.Unlock()

	// Validate islands exist
	if _, ok := db.Islands[sourceIsland]; !ok {
		return 0
	}
	if _, ok := db.Islands[targetIsland]; !ok {
		// Initialize if not exists (though NewProgramDatabase should have done it)
		db.Islands[targetIsland] = []string{}
	}

	// Get source program IDs
	sourceIDs := db.Islands[sourceIsland]
	if len(sourceIDs) == 0 {
		return 0
	}

	// Retrieve actual programs for sorting
	type progScore struct {
		ID    string
		Score float64
	}
	candidates := make([]progScore, 0, len(sourceIDs))
	for _, id := range sourceIDs {
		if p, ok := db.Programs[id]; ok {
			candidates = append(candidates, progScore{ID: id, Score: p.GetScore()})
		}
	}

	// Sort descending by score
    sort.Slice(candidates, func(i, j int) bool {
        return candidates[i].Score > candidates[j].Score
    })
    
    migrated := 0
    for i := 0; i < count && i < len(candidates); i++ {
        targetID := candidates[i].ID
        
        // Check if already in target island (to avoid duplicates)
        found := false
        for _, existingID := range db.Islands[targetIsland] {
            if existingID == targetID {
                found = true
                break
            }
        }
        
        if !found {
            db.Islands[targetIsland] = append(db.Islands[targetIsland], targetID)
            migrated++
        }
    }
    
    return migrated
}
