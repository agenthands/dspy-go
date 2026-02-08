package context_evolve

import (
	"math"
)

// MetricStats holds statistics for a metric in the population.
type MetricStats struct {
	Mean  float64
	Stdev float64
	Min   float64
	Max   float64
}

// CalculateStats computes the mean and standard deviation for a slice of values.
func CalculateStats(values []float64) MetricStats {
	if len(values) == 0 {
		return MetricStats{}
	}

	var sum, minVal, maxVal float64
	minVal = values[0]
	maxVal = values[0]

	for _, v := range values {
		sum += v
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}

	mean := sum / float64(len(values))

	var sqSum float64
	for _, v := range values {
		diff := v - mean
		sqSum += diff * diff
	}

	stdev := 0.0
	if len(values) > 1 {
		stdev = math.Sqrt(sqSum / float64(len(values)-1))
	}

	return MetricStats{
		Mean:  mean,
		Stdev: stdev,
		Min:   minVal,
		Max:   maxVal,
	}
}

// NormalizeValue calculates the z-score of a value given the stats.
func NormalizeValue(value float64, stats MetricStats) float64 {
	if stats.Stdev == 0 {
		return 0.0 // Avoid division by zero
	}
	return (value - stats.Mean) / stats.Stdev
}

// NormalizeMetrics normalizes a map of metrics against population statistics.
// It returns a map of z-scores.
func NormalizeMetrics(metrics map[string]float64, populationStats map[string]MetricStats) map[string]float64 {
	normalized := make(map[string]float64)
	for key, value := range metrics {
		if stats, ok := populationStats[key]; ok {
			normalized[key] = NormalizeValue(value, stats)
		} else {
			normalized[key] = 0.0 // Default if stats missing
		}
	}
	return normalized
}

// EvolutionCategory classifies a program's performance change relative to a parent.
type EvolutionCategory string

const (
	EvolutionImproved EvolutionCategory = "IMPROVED"
	EvolutionDegraded EvolutionCategory = "DEGRADED"
	EvolutionMixed    EvolutionCategory = "MIXED"
	EvolutionNeutral  EvolutionCategory = "NEUTRAL"
)

// ClassifyEvolution determines how a child program compares to its parent.
// Metrics are assumed to be "higher is better".
func ClassifyEvolution(parentMetrics, childMetrics map[string]float64) EvolutionCategory {
	betterCount := 0
	worseCount := 0
	equalCount := 0
	totalCount := 0

	for key, pVal := range parentMetrics {
		if cVal, ok := childMetrics[key]; ok {
			totalCount++
			if cVal > pVal {
				betterCount++
			} else if cVal < pVal {
				worseCount++
			} else {
				equalCount++
			}
		}
	}

	if totalCount == 0 {
		return EvolutionNeutral
	}

	if betterCount > 0 && worseCount == 0 {
		return EvolutionImproved
	}
	if worseCount > 0 && betterCount == 0 {
		return EvolutionDegraded
	}
	if betterCount == 0 && worseCount == 0 {
		return EvolutionNeutral
	}
	
	return EvolutionMixed
}

// CalculatePopulationStats computes statistics for all metrics across a set of programs.
func CalculatePopulationStats(programs []*Program) map[string]MetricStats {
	// 1. Collect values for each metric
	valuesMap := make(map[string][]float64)
	for _, p := range programs {
		for k, v := range p.Metrics {
			// Handle various numeric types
			var val float64
			switch v := v.(type) {
			case float64:
				val = v
			case int:
				val = float64(v)
			case int64:
				val = float64(v)
			default:
				continue // Skip non-numeric
			}
			valuesMap[k] = append(valuesMap[k], val)
		}
	}

	// 2. Calculate stats for each metric
	statsMap := make(map[string]MetricStats)
	for k, vals := range valuesMap {
		statsMap[k] = CalculateStats(vals)
	}

	return statsMap
}

// NormalizeProgramMetrics normalizes metrics for a list of programs.
// Returns a map ProgramID -> MetricName -> Z-Score.
func NormalizeProgramMetrics(programs []*Program, stats map[string]MetricStats) map[string]map[string]float64 {
	result := make(map[string]map[string]float64)
	for _, p := range programs {
		normalized := make(map[string]float64)
		for k, v := range p.Metrics {
			var val float64
			switch v := v.(type) {
			case float64:
				val = v
			case int:
				val = float64(v)
			case int64:
				val = float64(v)
			default:
				continue
			}

			if stat, ok := stats[k]; ok {
				normalized[k] = NormalizeValue(val, stat)
			}
		}
		result[p.ID] = normalized
	}
	return result
}

// ClassifyPrograms categorizes programs based on their normalized metrics (Z-scores).
// Improved: Sum(Z-scores) > 0.5 (Above average)
// Degraded: Sum(Z-scores) < -0.5 (Below average)
// Mixed/Neutral: -0.5 <= Sum(Z-scores) <= 0.5 (Average)
// This is a simplified heuristic when parent comparison isn't efficient.
func ClassifyPrograms(programs []*Program, normalizedMetrics map[string]map[string]float64) (improved, mixed, degraded []*Program) {
	for _, p := range programs {
		metrics, ok := normalizedMetrics[p.ID]
		if !ok {
			mixed = append(mixed, p)
			continue
		}

		var sumZ float64
		count := 0
		for _, z := range metrics {
			sumZ += z
			count++
		}

		avgZ := 0.0
		if count > 0 {
			avgZ = sumZ / float64(count)
		}

		// Thresholds for classification
		// These heuristics assume "Higher is better" for the metrics.
		// If some metrics are "Lower is better" (e.g. latency), they should be inverted before normalization or handled here.
		// For now, we assume standard maximization.
		if avgZ > 0.2 {
			improved = append(improved, p)
		} else if avgZ < -0.2 {
			degraded = append(degraded, p)
		} else {
			mixed = append(mixed, p)
		}
	}
	return improved, mixed, degraded
}
