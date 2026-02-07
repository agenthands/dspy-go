package core

import (
	"context"
	"testing"
)

// TestOptimizerRegistry tests the OptimizerRegistry.
func TestOptimizerRegistry(t *testing.T) {
	registry := NewOptimizerRegistry()

	// Test registering an Optimizer
	registry.Register("test", func() (Optimizer, error) {
		return &MockOptimizer{}, nil
	})

	// Test creating a registered Optimizer
	optimizer, err := registry.Create("test")
	if err != nil {
		t.Errorf("Unexpected error creating Optimizer: %v", err)
	}
	if _, ok := optimizer.(*MockOptimizer); !ok {
		t.Error("Created Optimizer is not of expected type")
	}

	// Test creating an unregistered Optimizer
	_, err = registry.Create("nonexistent")
	if err == nil {
		t.Error("Expected error when creating unregistered Optimizer, got nil")
	}
}

// TestCompileOptions tests the CompileOptions and related functions.
func TestCompileOptions(t *testing.T) {
	opts := &CompileOptions{}

	WithMaxTrials(10)(opts)
	if opts.MaxTrials != 10 {
		t.Errorf("Expected MaxTrials 10, got %d", opts.MaxTrials)
	}

	teacherProgram := &Program{
		Modules: map[string]Module{
			"test": NewModule(NewSignature(
				[]InputField{{Field: Field{Name: "input"}}},
				[]OutputField{{Field: Field{Name: "output"}}},
			)),
		},
		Forward: func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
			return inputs, nil
		},
	}

	WithTeacher(teacherProgram)(opts)
	if opts.Teacher == nil {
		t.Error("Expected Teacher program to be set")
	} else {
		if len(opts.Teacher.Modules) != 1 {
			t.Errorf("Expected 1 module in Teacher program, got %d", len(opts.Teacher.Modules))
		}
		if opts.Teacher.Forward == nil {
			t.Error("Expected Forward function to be set in Teacher program")
		}
	}
}

// TestBootstrapFewShot tests the BootstrapFewShot optimizer.
func TestBootstrapFewShot(t *testing.T) {
	optimizer := NewBootstrapFewShot(5)

	if optimizer.MaxExamples != 5 {
		t.Errorf("Expected MaxExamples 5, got %d", optimizer.MaxExamples)
	}

	// Create a simple program for testing
	// Create a mock Predict module that implements DemoConsumer
	predict := &MockPredict{
		BaseModule: *NewModule(NewSignature(
			[]InputField{{Field: Field{Name: "input"}}},
			[]OutputField{{Field: Field{Name: "output"}}},
		)),
	}
	predict.DisplayName = "test_module"

	// Create a simple program for testing
	program := NewProgram(map[string]Module{
		"test_module": predict,
	}, func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
		// MockPredict embeds BaseModule, so it has ProcessWithInterceptors.
		// We call it directly to allow interceptors to run.
		return predict.ProcessWithInterceptors(ctx, inputs, nil)
	})

	// Create a simple dataset for testing
	dataset := &MockDataset{
		data: []Example{
			{Inputs: map[string]interface{}{"input": "Q1"}, Outputs: map[string]interface{}{"output": "A1"}},
			{Inputs: map[string]interface{}{"input": "Q2"}, Outputs: map[string]interface{}{"output": "A2"}},
		},
	}

	// Create a simple metric for testing
	metric := func(expected, actual map[string]interface{}) float64 {
		return 1.0 // Always return 1.0 (success)
	}

	optimizedProgram, err := optimizer.Compile(context.Background(), program, dataset, metric)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	// Verify demos were added to the optimized program's module
	optMod := optimizedProgram.Modules["test_module"]
	if consumer, ok := optMod.(DemoConsumer); ok {
		demos := consumer.GetDemos()
		if len(demos) != 2 {
			t.Errorf("Expected 2 bootstrapped demos, got %d", len(demos))
		}
		// Verify content of first demo
		if demos[0].Inputs["input"] != "Q1" || demos[0].Outputs["output"] != "A1" {
			t.Errorf("Unexpected demo content: %v", demos[0])
		}
	} else {
		t.Error("Optimized module does not implement DemoConsumer")
	}
}

// MockPredict simulates a Predict module for testing.
type MockPredict struct {
	BaseModule
	Demos []Example
}

func (m *MockPredict) Process(ctx context.Context, inputs map[string]any, opts ...Option) (map[string]any, error) {
	// Simulate "processing" by returning the expected output if we know it (from dataset/trace context)
	// But simply returning inputs is enough for testing TraceInterceptor capture.
	// The TraceInterceptor captures whatever this returns.
	// For "Q1", let's return "A1" to match dataset and pass metric.
	
	inp, _ := inputs["input"].(string)
	if inp == "Q1" {
		return map[string]any{"output": "A1"}, nil
	}
	if inp == "Q2" {
		return map[string]any{"output": "A2"}, nil
	}
	return map[string]any{"output": "unknown"}, nil
}

// ProcessWithInterceptors overrides BaseModule's method to ensure MockPredict.Process is called.
func (m *MockPredict) ProcessWithInterceptors(ctx context.Context, inputs map[string]any, interceptors []ModuleInterceptor, opts ...Option) (map[string]any, error) {
	return m.BaseModule.ProcessWithInterceptorsImpl(ctx, inputs, interceptors, m.Process, opts...)
}

func (m *MockPredict) SetDemos(demos []Example) {
	m.Demos = demos
}

func (m *MockPredict) GetDemos() []Example {
	return m.Demos
}

func (m *MockPredict) Clone() Module {
	return &MockPredict{
		BaseModule: *m.BaseModule.Clone().(*BaseModule),
		Demos:      append([]Example{}, m.Demos...),
	}
}

// MockOptimizer is a mock implementation of the Optimizer interface for testing.
type MockOptimizer struct{}

func (m *MockOptimizer) Compile(ctx context.Context, program Program, dataset Dataset, metric Metric) (Program, error) {
	return program, nil
}

// MockDataset is a mock implementation of the Dataset interface for testing.
type MockDataset struct {
	data  []Example
	index int
}

func (m *MockDataset) Next() (Example, bool) {
	if m.index >= len(m.data) {
		return Example{}, false
	}
	example := m.data[m.index]
	m.index++
	return example, true
}

func (m *MockDataset) Reset() {
	m.index = 0
}
