package context_evolve

import (
    "bytes"
    "context"
    "encoding/json"
    "fmt"
    "os"
    "os/exec"
    "strings"
    "time"
)

// CommandEvaluator implements the Evaluator interface by running an external command.
type CommandEvaluator struct {
    Command        string
    Timeout        time.Duration
    ContentFile    string // Temp file to write content to before running command
}

// NewCommandEvaluator creates a new CommandEvaluator.
func NewCommandEvaluator(command string, timeoutSeconds int) *CommandEvaluator {
    return &CommandEvaluator{
        Command:     command,
        Timeout:     time.Duration(timeoutSeconds) * time.Second,
        ContentFile: "temp_program_content.json", // Default, can be overridden per execution if needed
    }
}

// Evaluate runs the command with the program content and returns metrics.
// The command is expected to read the content file (or stdin) and output JSON metrics to stdout.
func (e *CommandEvaluator) Evaluate(ctx context.Context, program *Program) (map[string]interface{}, error) {
    // Write content to temp file
    // Ideally use a unique temp file per execution to avoid race conditions in parallel
    tmpFile, err := os.CreateTemp("", "program_*.json")
    if err != nil {
        return nil, fmt.Errorf("failed to create temp file: %w", err)
    }
    defer os.Remove(tmpFile.Name()) // Clean up

    if _, err := tmpFile.WriteString(program.Content); err != nil {
        return nil, fmt.Errorf("failed to write content to temp file: %w", err)
    }
    if err := tmpFile.Close(); err != nil {
        return nil, fmt.Errorf("failed to close temp file: %w", err)
    }

    // Prepare command
    // We assume the command takes the filename as the first argument
    // e.g., "python evaluator.py temp_program_123.json"
    parts := strings.Fields(e.Command)
    cmdName := parts[0]
    cmdArgs := append(parts[1:], tmpFile.Name())

    ctx, cancel := context.WithTimeout(ctx, e.Timeout)
    defer cancel()

    cmd := exec.CommandContext(ctx, cmdName, cmdArgs...)
    var stdout, stderr bytes.Buffer
    cmd.Stdout = &stdout
    cmd.Stderr = &stderr

    if err := cmd.Run(); err != nil {
        return nil, fmt.Errorf("command execution failed: %w, stderr: %s", err, stderr.String())
    }

    // Parse output JSON
    var metrics map[string]interface{}
    if err := json.Unmarshal(stdout.Bytes(), &metrics); err != nil {
        // If not JSON, maybe just return raw output as "output" metric?
        // For now, enforce JSON for strictness
        return nil, fmt.Errorf("failed to parse output metrics: %w, output: %s", err, stdout.String())
    }

    return metrics, nil
}
