"""
Utility functions for formatting output
"""

from typing import Any, Dict


def format_metrics_dict(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format all numeric values in a metrics dictionary to 4 decimal places.
    
    Args:
        metrics: Dictionary of metric names to values
    
    Returns:
        Dictionary with numeric values formatted to 4 decimal places
    """
    if not metrics:
        return {}
    
    formatted_metrics = {}
    for name, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                formatted_metrics[name] = round(float(value), 4)
            except (ValueError, TypeError):
                formatted_metrics[name] = value
        else:
            formatted_metrics[name] = value
    
    return formatted_metrics


def format_metrics_safe(metrics: Dict[str, Any]) -> str:
    """
    Safely format metrics dictionary for logging, handling both numeric and string values.

    Args:
        metrics: Dictionary of metric names to values

    Returns:
        Formatted string representation of metrics
    """
    if not metrics:
        return ""

    formatted_parts = []
    for name, value in metrics.items():
        # Check if value is numeric (int, float)
        if isinstance(value, (int, float)):
            try:
                # Only apply float formatting to numeric values
                formatted_parts.append(f"{name}={value:.4f}")
            except (ValueError, TypeError):
                # Fallback to string representation if formatting fails
                formatted_parts.append(f"{name}={value}")
        else:
            # For non-numeric values (strings, etc.), just convert to string
            formatted_parts.append(f"{name}={value}")

    return ", ".join(formatted_parts)


def format_improvement_safe(parent_metrics: Dict[str, Any], child_metrics: Dict[str, Any]) -> str:
    """
    Safely format improvement metrics for logging.

    Args:
        parent_metrics: Parent program metrics
        child_metrics: Child program metrics

    Returns:
        Formatted string representation of improvements
    """
    if not parent_metrics or not child_metrics:
        return ""

    improvement_parts = []
    for metric, child_value in child_metrics.items():
        if metric in parent_metrics:
            parent_value = parent_metrics[metric]
            # Only calculate improvement for numeric values
            if isinstance(child_value, (int, float)) and isinstance(parent_value, (int, float)):
                try:
                    diff = child_value - parent_value
                    improvement_parts.append(f"{metric}={diff:+.4f}")
                except (ValueError, TypeError):
                    # Skip non-numeric comparisons
                    continue

    return ", ".join(improvement_parts)
