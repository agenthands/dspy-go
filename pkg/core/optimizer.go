package core

import (
	"context"
	"errors"
)

// Optimizer represents an interface for optimizing DSPy programs.
type Optimizer interface {
	// Compile optimizes a given program using the provided dataset and metric
	Compile(ctx context.Context, program Program, dataset Dataset, metric Metric) (Program, error)
}

// Metric is a function type that evaluates the performance of a program.
type Metric func(expected, actual map[string]interface{}) float64

// Dataset represents a collection of examples for training/evaluation.
type Dataset interface {
	// Next returns the next example in the dataset
	Next() (Example, bool)
	// Reset resets the dataset iterator
	Reset()
}

// Example represents a single training/evaluation example.
type Example struct {
	Inputs  map[string]interface{}
	Outputs map[string]interface{}
}

// DatasetToSlice converts a Dataset to a slice of Examples.
// This is a helper function to avoid repeated iteration patterns.
func DatasetToSlice(dataset Dataset) []Example {
	var examples []Example
	dataset.Reset()
	for {
		example, hasNext := dataset.Next()
		if !hasNext {
			break
		}
		examples = append(examples, example)
	}
	return examples
}

// BaseOptimizer provides a basic implementation of the Optimizer interface.
type BaseOptimizer struct {
	Name string
}

// Compile is a placeholder implementation and should be overridden by specific optimizer implementations.
func (bo *BaseOptimizer) Compile(ctx context.Context, program Program, dataset Dataset, metric Metric) (Program, error) {
	return Program{}, errors.New("Compile method not implemented")
}

// OptimizerFactory is a function type for creating Optimizer instances.
type OptimizerFactory func() (Optimizer, error)

// OptimizerRegistry maintains a registry of available Optimizer implementations.
type OptimizerRegistry struct {
	factories map[string]OptimizerFactory
}

// NewOptimizerRegistry creates a new OptimizerRegistry.
func NewOptimizerRegistry() *OptimizerRegistry {
	return &OptimizerRegistry{
		factories: make(map[string]OptimizerFactory),
	}
}

// Register adds a new Optimizer factory to the registry.
func (r *OptimizerRegistry) Register(name string, factory OptimizerFactory) {
	r.factories[name] = factory
}

// Create instantiates a new Optimizer based on the given name.
func (r *OptimizerRegistry) Create(name string) (Optimizer, error) {
	factory, exists := r.factories[name]
	if !exists {
		return nil, errors.New("unknown Optimizer type: " + name)
	}
	return factory()
}

// CompileOptions represents options for the compilation process.
type CompileOptions struct {
	MaxTrials int
	Teacher   *Program
}

// WithMaxTrials sets the maximum number of trials for optimization.
func WithMaxTrials(n int) func(*CompileOptions) {
	return func(o *CompileOptions) {
		o.MaxTrials = n
	}
}

// WithTeacher sets a teacher program for optimization.
func WithTeacher(teacher *Program) func(*CompileOptions) {
	return func(o *CompileOptions) {
		o.Teacher = teacher
	}
}

// BootstrapFewShot implements a basic few-shot learning optimizer.
type BootstrapFewShot struct {
	BaseOptimizer
	MaxExamples int
}

// NewBootstrapFewShot creates a new BootstrapFewShot optimizer.
func NewBootstrapFewShot(maxExamples int) *BootstrapFewShot {
	return &BootstrapFewShot{
		BaseOptimizer: BaseOptimizer{Name: "BootstrapFewShot"},
		MaxExamples:   maxExamples,
	}
}

// Compile implements the optimization logic for BootstrapFewShot.
func (bfs *BootstrapFewShot) Compile(ctx context.Context, program Program, dataset Dataset, metric Metric) (Program, error) {
	// 1. Setup TraceInterceptor on the Teacher (using original program directly to avoid Clone/Closure issues)
	// We will restore original interceptors after tracing.
	tracer := NewTraceInterceptor()
	originalInterceptors := make(map[string][]ModuleInterceptor)
	
	for name, mod := range program.Modules {
		if interceptable, ok := mod.(InterceptableModule); ok {
			// Save current interceptors
			current := interceptable.GetInterceptors()
			originalInterceptors[name] = current
			
			// Append tracer
			newInterceptors := append(current, tracer.Intercept)
			interceptable.SetInterceptors(newInterceptors)
		}
	}
	
	// Ensure we restore interceptors even if panic occurs
	defer func() {
		for name, mod := range program.Modules {
			if interceptable, ok := mod.(InterceptableModule); ok {
				if original, exists := originalInterceptors[name]; exists {
					interceptable.SetInterceptors(original)
				}
			}
		}
	}()

	// 4. Trace Generation Loop
	dataset.Reset()
	var successfulTraces []TraceEntry
	
	// Iterate through the dataset
	for {
		example, ok := dataset.Next()
		if !ok {
			break
		}

		// Clear previous traces for this run
		tracer.Clear()

		// Execute Teacher (Original Program)
		prediction, err := program.InsertExecState(ctx).Execute(ctx, example.Inputs)
		if err != nil {
			continue // Skip failed executions
		}

		// Evaluate Metric
		score := metric(example.Outputs, prediction)
		
		// If successful (score > 0? or threshold?), we keep the traces as potential demos.
		// For strict correctness, maybe score == 1.0 or based on config.
		if score >= 1.0 { // Assuming 1.0 is max score / success
			traces := tracer.GetTraces()
			successfulTraces = append(successfulTraces, traces...)
		}
	}

	// 3. Create Student as Clone (now that tracing is done)
	student := program.Clone()

	// 5. Bootstrap Demos into Student
	// For each module in Student, find corresponding successful traces and add as demos.
	for name, studentMod := range student.Modules {
		// Only Predict modules support Demos.
		// We need to check if it's a Predict module or supports SetDemos.
		// Interface check: DemoConsumer
		if consumer, ok := studentMod.(DemoConsumer); ok {
            var newDemos []Example
            
            // Find traces for this module name
            for _, trace := range successfulTraces {
                if trace.ModuleName == name || trace.ModuleName == studentMod.GetDisplayName() {
                    // Convert Trace to Example
                    demo := Example{
                        Inputs:  trace.Inputs,
                        Outputs: trace.Outputs,
                    }
                    newDemos = append(newDemos, demo)
                    
                    if len(newDemos) >= bfs.MaxExamples {
                        break
                    }
                }
            }
            
            // Set Demos
            if len(newDemos) > 0 {
                consumer.SetDemos(newDemos)
            }
		}
	}

	return student, nil
}

// Helper to ensure context has execution state
func (p Program) InsertExecState(ctx context.Context) Program {
    // This is just a helper for the test/compile loop, 
    // real execution handles this in Execute.
    return p
}

type ProgressReporter interface {
	Report(stage string, processed, total int)
}
