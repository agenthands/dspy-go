package context_evolve

import (
	"context"
	"fmt"
	"log"
	"math/rand"
    "sync"
	"time"
)

// Controller orchestrates the ContextEvolve process.
// It manages the interactions between agents (Summary, Navigator, Sampler, Evolution)
// and the Evaluator to evolve the population of programs.
type Controller struct {
	Config    Config
	Database  *ProgramDatabase
	Evaluator Evaluator
	Agents    struct {
		Summarizer Summarizer
		Navigator  Navigator
		Sampler    Sampler
		Evolution  Evolution
	}
}

// Evaluator interface allows for different evaluation strategies (e.g., local command, remote service).
type Evaluator interface {
	Evaluate(ctx context.Context, program *Program) (map[string]interface{}, error)
}

// NewController creates a new Controller instance with the given configuration.
// It initializes default agents and database.
//
// To use custom agents/evaluators, create the Controller manually or add Setters.
func NewController(config Config) *Controller {
	c := &Controller{
		Config:    config,
		Database:  NewProgramDatabase(config.Evolution.NumIslands),
		Evaluator: NewCommandEvaluator(config.Evaluator.Command, config.Evaluator.TimeoutSeconds),
	}
	c.Agents.Summarizer = NewSummarizerAgent()
	c.Agents.Navigator = NewNavigatorAgent()
	c.Agents.Sampler = NewSamplerAgent(config)
	c.Agents.Evolution = NewEvolutionAgent()
	return c
}

// Run starts the evolution process.
func (c *Controller) Run(ctx context.Context, initialContent string) (*Program, error) {
	// Initialize with the first program
	initialProgram := &Program{
		ID:             fmt.Sprintf("init-%d", time.Now().UnixNano()),
		Content:        initialContent,
		Language:       "unknown", // Detect if possible, or config
		Generation:     0,
		IterationFound: 0,
		Timestamp:      time.Now(),
	}

	// Elevate initial program
	metrics, err := c.Evaluator.Evaluate(ctx, initialProgram)
	if err != nil {
		return nil, fmt.Errorf("initial evaluation failed: %w", err)
	}
	initialProgram.Metrics = metrics

	// Summarize initial program
	abstract, err := c.Agents.Summarizer.Summarize(ctx, initialProgram, "")
	if err != nil {
		log.Printf("Warning: failed to summarize initial program: %v", err)
		initialProgram.Abstract = "No abstract available"
	} else {
		initialProgram.Abstract = abstract
	}

	// Add to all islands initially to seed them
	for i := 0; i < c.Config.Evolution.NumIslands; i++ {
		c.Database.Add(initialProgram, i)
	}

	// Main Evolution Loop (Parallel across islands)
	// We run MaxIterations 'epochs', where each epoch runs one evolution step per island concurrently.
	// Total evaluations = MaxIterations * NumIslands (approx)
	// Or should MaxIterations be total evals? Let's assume MaxIterations is total steps per island for simplicity.
    maxEpochs := c.Config.Evolution.MaxIterations
    if c.Config.Evolution.NumIslands > 0 {
         // Adjust if user meant total evals. Usually iterations implies loops.
         // Let's keep it simple: Loop N times, each time parallelize.
    }

	for epoch := 1; epoch <= maxEpochs; epoch++ {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		log.Printf("Epoch %d/%d (Parallel Islands: %d)...", epoch, maxEpochs, c.Config.Evolution.NumIslands)

		// 1. Parallel Evolution Step
		var wg sync.WaitGroup
		errChan := make(chan error, c.Config.Evolution.NumIslands)
        
        // We'll collect children to add to DB after to avoid lock contention during heavy compute?
        // Actually DB has lock, so concurrent Add is fine.
        
		for islandID := 0; islandID < c.Config.Evolution.NumIslands; islandID++ {
			wg.Add(1)
			go func(id int) {
				defer wg.Done()
				
				// A. Select Parent
				parent := c.selectParent(id)
				if parent == nil {
					// Should not happen if seeded
					return
				}

				// B. Get Gradient (Navigator)
                // Use weighted sampling from population history
                batchSize := 20
                if c.Config.LLM.MaxTokens > 0 {
                    batchSize = c.Config.LLM.MaxTokens / 500
                }
                history := c.Agents.Sampler.SampleFromPopulation(c.Database, batchSize)
                if len(history) == 0 {
                     history = c.Database.GetIslandPrograms(id)
                }
                
                // Use history from specific island or global?
                // Ideally per-island history for diversity, but Sampler samples globally currently.
                // Let's stick to global sampling for now as per previous implementation, 
                // but maybe we should scope it later.
                
				gradient, err := c.Agents.Navigator.GetGradient(ctx, history)
				if err != nil {
					errChan <- fmt.Errorf("island %d navigator failed: %w", id, err)
					return
				}

				// C. Get References (Sampler)
				allPrograms := c.Database.GetAllPrograms() // or just Island programs?
				references, err := c.Agents.Sampler.SelectReferences(ctx, allPrograms, parent, gradient)
				if err != nil {
                    // Log but continue
					references = []*Program{}
				}

				// D. Generate Child (Evolution)
				newContent, err := c.Agents.Evolution.Evolve(ctx, parent, references, gradient)
				if err != nil {
					errChan <- fmt.Errorf("island %d evolution failed: %w", id, err)
					return
				}

				child := &Program{
					ID:             fmt.Sprintf("gen-%d-is%d-%d", epoch, id, time.Now().UnixNano()),
					Content:        newContent,
					ParentID:       parent.ID,
					Generation:     parent.Generation + 1,
					IterationFound: epoch,
					Timestamp:      time.Now(),
					Gradient:       gradient,
				}

				// E. Evaluate Child
				childMetrics, err := c.Evaluator.Evaluate(ctx, child)
				if err != nil {
					child.Metrics = map[string]interface{}{"error": err.Error()}
				} else {
					child.Metrics = childMetrics
				}

				// F. Summarize Child
				childAbstract, err := c.Agents.Summarizer.Summarize(ctx, child, parent.Abstract)
				if err != nil {
					child.Abstract = "No abstract available"
				} else {
					child.Abstract = childAbstract
				}

				// G. Add to Database (Same Island)
				c.Database.Add(child, id)
                
			}(islandID)
		}
		wg.Wait()
		close(errChan)

        // Check errors
        for err := range errChan {
            log.Printf("Error in epoch %d: %v", epoch, err)
        }

		// 2. Migration Step
        if c.Config.Evolution.MigrationInterval > 0 && epoch%c.Config.Evolution.MigrationInterval == 0 {
            log.Printf("Migrating programs between islands...")
            // Simple ring migration: 0->1, 1->2, ... N->0
            migratedTotal := 0
            for i := 0; i < c.Config.Evolution.NumIslands; i++ {
                target := (i + 1) % c.Config.Evolution.NumIslands
                count := c.Database.Migrate(i, target, 2) // Migrate top 2
                migratedTotal += count
            }
            log.Printf("Migrated %d programs.", migratedTotal)
        }

        // 3. Early Exit Check (Global Best)
        best, ok := c.Database.Get(c.Database.BestProgramID)
        if ok {
            score := getScore(best)
            if c.Config.Evolution.TargetScore > 0 && score >= c.Config.Evolution.TargetScore {
                log.Printf("Target score reached: %f by %s", score, best.ID)
                break
            }
        }
	}

	best, ok := c.Database.Get(c.Database.BestProgramID)
    if !ok {
        return nil, fmt.Errorf("no best program found")
    }
	return best, nil
}

// selectParent selects a parent program from the population.
// This is a placeholder for a more sophisticated selection strategy (e.g., UCB, Tournament).
func (c *Controller) selectParent(islandID int) *Program {
    // Simple: Pick the best program 50% of the time, random 50%
    // Best program logic here is global, might need per-island best.
    // For now, let's use global best if in same island, or random from island.
    
    islandPrograms := c.Database.GetIslandPrograms(islandID)
    if len(islandPrograms) == 0 {
        // Fallback to global if island empty (should not happen after init)
        all := c.Database.GetAllPrograms()
        if len(all) == 0 {
            return nil
        }
        return all[rand.Intn(len(all))]
    }

    // Attempt to pick top performer
    // Since we don't track per-island best explicitly yet, just pick random deeply.
    // Or scan for best in island (O(N) for island size).
    bestInIsland := islandPrograms[0]
    for _, p := range islandPrograms {
        if p.GetScore() > bestInIsland.GetScore() {
            bestInIsland = p
        }
    }
    
    if rand.Float64() < 0.5 {
        return bestInIsland
    }

    // Random
    return islandPrograms[rand.Intn(len(islandPrograms))]
}
