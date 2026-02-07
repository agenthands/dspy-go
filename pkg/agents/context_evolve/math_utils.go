package context_evolve

import (
	"math"
)

// NormalizeMetrics calculates z-score normalization for a list of programs.
// Returns a slice of maps where each map corresponds to a program's normalized metrics.
func NormalizeMetrics(programs []*Program) []map[string]float64 {
	if len(programs) == 0 {
		return nil
	}

	// 1. Collect all metric names
	metricNames := make(map[string]bool)
	for _, p := range programs {
		for k := range p.Metrics {
			metricNames[k] = true
		}
	}

	// 2. Collect values for each metric
	metricValues := make(map[string][]float64)
	for name := range metricNames {
		metricValues[name] = make([]float64, 0, len(programs))
	}

	for _, p := range programs {
		for name := range metricNames {
			val := 0.0
			if v, ok := p.Metrics[name]; ok {
				switch v := v.(type) {
				case float64:
					val = v
				case int:
					val = float64(v)
				}
			}
			metricValues[name] = append(metricValues[name], val)
		}
	}

	// 3. Calculate mean and stddev for each metric
	metricStats := make(map[string]struct{ mean, std float64 })
	for name, values := range metricValues {
		sum := 0.0
		for _, v := range values {
			sum += v
		}
		mean := sum / float64(len(values))

		sumSqDiff := 0.0
		for _, v := range values {
			diff := v - mean
			sumSqDiff += diff * diff
		}
		std := math.Sqrt(sumSqDiff / float64(len(values)))
		if std < 1e-10 {
			std = 1.0 // Avoid division by zero
		}
		metricStats[name] = struct{ mean, std float64 }{mean, std}
	}

	// 4. Normalize
	normalizedMetrics := make([]map[string]float64, len(programs))
	for i, p := range programs {
		normalized := make(map[string]float64)
		for name := range metricNames {
			val := 0.0
			if v, ok := p.Metrics[name]; ok {
				switch v := v.(type) {
				case float64:
					val = v
				case int:
					val = float64(v)
				}
			}
			stats := metricStats[name]
			normalized[name] = (val - stats.mean) / stats.std
		}
		normalizedMetrics[i] = normalized
	}

	return normalizedMetrics
}

// Classification categories
const (
	ClassImproved = "improved"
	ClassDegraded = "degraded"
	ClassMixed    = "mixed"
)

// ClassifyPrograms classifies programs based on their normalized metrics.
func ClassifyPrograms(programs []*Program, normalizedMetrics []map[string]float64) (improved, degraded, mixed []*Program) {
	epsilon := 1e-10

	for i, p := range programs {
		if i >= len(normalizedMetrics) {
			mixed = append(mixed, p)
			continue
		}
		metrics := normalizedMetrics[i]
		if len(metrics) == 0 {
			mixed = append(mixed, p)
			continue
		}

		allImproved := true
		allDegraded := true

		for _, v := range metrics {
			if v <= epsilon {
				allImproved = false
			}
			if v >= -epsilon {
				allDegraded = false
			}
		}

		if allImproved {
			improved = append(improved, p)
		} else if allDegraded {
			degraded = append(degraded, p)
		} else {
			mixed = append(mixed, p)
		}
	}
	return
}
