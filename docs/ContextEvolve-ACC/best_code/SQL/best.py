import pandas as pd
from solver import Algorithm
from typing import Tuple, List
from collections import Counter

class Evolved(Algorithm):
    """
    Simplified greedy algorithm that reorders columns to maximize prefix reuse.
    Prioritizes columns with high-frequency values and long string lengths using
    vectorized operations. Removes recursion, threading, and complex control flow
    for improved speed and maintainability.
    """
    
    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    def calculate_column_score(self, col: pd.Series) -> float:
        """Calculate score using vectorized operations for speed and better stability."""
        col = col.dropna()
        if col.empty:
            return 0.0
        
        # Use vectorized string conversion and length calculation
        str_values = col.astype(str)
        lengths = str_values.str.len()
        length_sq = lengths * lengths

        # Get value counts and align with original series
        value_counts = col.value_counts()
        counts = str_values.map(value_counts)

        # Only values appearing more than once contribute: (count - 1) * length_sq
        contributing = (counts - 1).clip(lower=0)
        return float((contributing * length_sq).sum())

    def reorder(
        self,
        df: pd.DataFrame,
        early_stop: int = 0,
        row_stop: int = None,
        col_stop: int = None,
        col_merge: List[List[str]] = [],
        one_way_dep: List[Tuple[str, str]] = [],
        distinct_value_threshold: float = 0.8,
        parallel: bool = True,
    ) -> Tuple[pd.DataFrame, List[List[str]]]:
        """
        Reorder columns to maximize prefix reuse when processing rows sequentially.
        Uses a greedy, non-recursive approach that scores columns based on potential
        contribution to prefix hits (frequency × squared length).
        """
        # Work with a copy to avoid modifying the original
        df = df.copy()

        # Handle column merging with improved logic and robustness
        if col_merge:
            for merge_group in col_merge:
                valid_cols = [col for col in merge_group if col in df.columns]
                if len(valid_cols) > 1:
                    # Create merged column by forward-filling non-null values
                    merged_values = df[valid_cols[0]].copy()
                    for col in valid_cols[1:]:
                        merged_values = merged_values.fillna(df[col])
                    df[valid_cols[0]] = merged_values
                    # Drop the rest of the columns in the group
                    df = df.drop(columns=[col for col in valid_cols[1:]])

        # Early filter: remove columns with too many unique values
        n = len(df)
        threshold = n * distinct_value_threshold
        df_filtered = df[[col for col in df.columns if df[col].nunique() <= threshold]]

        # Score all columns using vectorized computation
        scores = []
        for col in df_filtered.columns:
            score = self.calculate_column_score(df_filtered[col])
            scores.append((col, score))

        # Sort columns by score descending
        ordered_columns = [col for col, _ in sorted(scores, key=lambda x: x[1], reverse=True)]

        # Apply final column order
        result_df = df_filtered[ordered_columns]

        # Use same column ordering for all rows (stable and efficient)
        column_orderings = [ordered_columns] * len(result_df)

        return result_df, column_orderings
