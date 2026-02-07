"""OpenEvolve masker implementation.

This module provides the OpenEvolve masker that implements evolutionary attention patterns.
The current implementation is a bare metal version that returns the previous mask.
"""

from dataclasses import dataclass
from typing import Any, Dict

import torch
from scipy.stats import norm

from sparse_attention_hub.sparse_attention.research_attention.maskers import (
    ResearchMasker,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
    MaskerConfig,
    MaskerRegistry,
)
from sparse_attention_hub.sparse_attention.utils.mask import Mask
from sparse_attention_hub.sparse_attention.utils.mask_attention_utils import (
    _get_num_key_value_groups,
    apply_inv_mask_sum,
    create_sampling_mask_with_per_head_budget,
    repeat_kv,
)


@dataclass
class OpenEvolveMaskerConfig(MaskerConfig):
    """Configuration for OpenEvolveMasker.

    This configuration class inherits from MaskerConfig and provides
    parameters for the attention mechanism evolved out of openevolve.
    empty placeholder
    """

    pass


@MaskerRegistry.register(OpenEvolveMaskerConfig)
class OpenEvolveMasker(ResearchMasker):
    """OpenEvolve masker for evolutionary attention computation.

    This masker implements evolutionary attention patterns that adapt over time.
    The current implementation is a bare metal version that returns the previous mask.

    Attributes:
        evolution_rate: The rate of evolution for attention patterns.
            This value is set from the configuration and controls how quickly
            the attention patterns evolve.

    Important Notes:
        - This is a bare metal implementation that simply returns the previous mask.
        - Future implementations will include evolutionary algorithms for attention pattern optimization.
        - The evolution_rate parameter is currently unused but will be utilized in future versions.

    Example:
        >>> config = OpenEvolveMaskerConfig(evolution_rate=1.0)
        >>> masker = OpenEvolveMasker(config)
        >>> # Use masker.add_mask() to apply evolutionary attention patterns
    """

    def __init__(self, config: OpenEvolveMaskerConfig) -> None:
        """Initialize OpenEvolve masker with configuration.

        Args:
            config: Configuration object containing the evolution rate and other
                parameters for the OpenEvolve masker.

        Raises:
            ValueError: If the evolution_rate in config is negative.
                This validation is performed in the config's __post_init__ method.
        """
        self.base_rate_sampling = 0.01
        self.epsilon = 0.3
        self.delta = 0.3
        self.init_offset = 0.001
        self.local_offset = 0.001

        super().__init__(config)

    def _compute_exp_attention_scores(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        scaling: float,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute exponential attention scores with numerical stability."""
        ngroups = _get_num_key_value_groups(queries, keys)
        keys = repeat_kv(keys, ngroups)
        raw_scores = torch.matmul(queries, keys.transpose(-2, -1)) * scaling
        if attention_mask is not None:
            raw_scores = raw_scores + attention_mask[:, :, :, : keys.shape[-2]]
        max_scores = torch.max(raw_scores, dim=-1, keepdim=True)[0]
        return torch.exp(raw_scores - max_scores)

    def _get_sampling_range(self, seq_len_keys: int) -> tuple[int, int, int]:
        """Get sampling range and validate it.

        Args:
            seq_len_keys: Number of keys in the sequence.

        Returns:
            Tuple of (start_idx, end_idx, sampling_range).

        Raises:
            ValueError: If the computed sampling range is invalid.
        """
        # Compute start index
        if isinstance(self.init_offset, float):
            start_idx: int = int(self.init_offset * seq_len_keys)
        else:
            start_idx = self.init_offset

        # Compute end index
        if isinstance(self.local_offset, float):
            end_idx: int = seq_len_keys - int(self.local_offset * seq_len_keys)
        else:
            end_idx = seq_len_keys - self.local_offset

        sampling_range = end_idx - start_idx

        if sampling_range <= 0:
            raise ValueError(f"Invalid sampling range: {sampling_range}")

        return start_idx, end_idx, sampling_range

    def _get_base_sample_count(self, sampling_range: int) -> int:
        """Get number of base samples based on configuration."""
        # Ensure at least 2 samples since it is used for std estimation
        if isinstance(self.base_rate_sampling, int):
            return max(2, self.base_rate_sampling)
        return max(2, int(self.base_rate_sampling * sampling_range))

    def _get_std_estimate_using_base_sample(
        self,
        expwts: torch.Tensor,
        batch_size: int,
        num_heads: int,
        seq_len_queries: int,
        seq_len_keys: int,
        start_idx: int,
        end_idx: int,
        num_base_samples: int,
        dtype: torch.dtype,
    ) -> tuple[Mask, torch.Tensor]:
        """Get standard deviation estimate using base sampling and create base mask."""
        # Create base sampling indices
        base_row_wise_idx = torch.randint(
            low=start_idx,
            high=end_idx,
            size=(batch_size, num_heads, seq_len_queries, num_base_samples),
            device=expwts.device,
        )

        # Extract values and compute std
        sampled_values = torch.gather(expwts, dim=-1, index=base_row_wise_idx)
        total_rows = batch_size * num_heads * seq_len_queries
        row_sampled_values = sampled_values.view(total_rows, num_base_samples)
        std_estimate = torch.std(row_sampled_values, dim=-1, keepdim=True)
        std_estimate = torch.clamp(std_estimate, min=1e-8)
        std_estimate = std_estimate.view(batch_size, num_heads, seq_len_queries, 1)

        # Create base sampling mask
        sampling_range = end_idx - start_idx
        base_data = torch.full_like(
            base_row_wise_idx, num_base_samples / sampling_range, dtype=expwts.dtype
        )

        base_mask = Mask.create_from_row_wise_idx(
            shape=(batch_size, num_heads, seq_len_queries, seq_len_keys),
            row_wise_idx=base_row_wise_idx,
            data=base_data,
            type="index",
            dtype=dtype,
        )

        return base_mask, std_estimate

    def _compute_adaptive_budget(
        self,
        std_estimate: torch.Tensor,
        estimated_denominator: torch.Tensor,
        sampling_range: int,
    ) -> torch.Tensor:
        """Compute adaptive budget based on statistical bounds."""
        epsilon_allowable_error = self.epsilon * estimated_denominator
        epsilon_allowable_error = torch.clamp(epsilon_allowable_error, min=1e-8)

        budget_numerator = self.delta_ppf * std_estimate * sampling_range
        budget_squared = (budget_numerator / epsilon_allowable_error) ** 2

        # Ensure budget is positive and within bounds
        budget = torch.clamp(
            budget_squared,
            min=1.0,  # Minimum 1 sample
            max=float(sampling_range),  # Maximum sampling_range samples
        ).long()

        return budget

    def add_mask(
        self,
        keys: torch.Tensor,
        queries: torch.Tensor,
        values: torch.Tensor,
        attention_mask: torch.Tensor,
        scaling: float,
        dropout: float,
        sparse_meta_data: Dict[Any, Any],
        previous_mask: Mask,
        **kwargs: Dict[str, Any],
    ) -> Mask:
        """Hybrid Sparse Attention with Statistical Guarantees."""
        if previous_mask.is_full_mask():
            return previous_mask
        
        # Handle grouped queries (multi-query/grouped-query attention)
        ngroups = _get_num_key_value_groups(queries, keys)
        keys = repeat_kv(keys, ngroups)
        B, H, QL, KL = queries.shape[0], queries.shape[1], queries.shape[2], keys.shape[2]
        device, dtype = queries.device, queries.dtype

        # Compute Exponential Scores
        expwts = self._compute_exp_attention_scores(queries, keys, scaling, attention_mask)

        # 1. Improved Log-scaled Top-K Selection for Heavy Hitters
        k_val = int(32 * torch.log2(torch.log2(torch.tensor(KL, device=device).clamp(min=4.0))))
        k = min(max(16, k_val), KL)
        _, topk_indices = torch.topk(expwts, k=k, dim=-1)
        topk_mask = Mask.create_from_row_wise_idx(
            shape=(B, H, QL, KL),
            row_wise_idx=topk_indices,
            data=torch.ones_like(topk_indices, dtype=dtype),
            type="index",
            dtype=dtype
        )

        # 2. Statistical Sampling for Tail Distribution
        base_rate_sampling = 0.012 if KL > 10000 else \
                             0.011 if KL > 7500 else \
                             0.010 if KL > 5000 else \
                             0.009 if KL > 2000 else 0.008

        epsilon = 0.12 if KL > 10000 else \
                  0.15 if KL > 7500 else \
                  0.18 if KL > 5000 else \
                  0.22 if KL > 2000 else 0.28

        delta = 0.18 if KL > 10000 else \
               0.20 if KL > 7500 else \
               0.22 if KL > 5000 else \
               0.25 if KL > 2000 else 0.30

        # Fixed offsets for stability
        init_offset = 0.002
        local_offset = 0.002

        # Pre-compute delta_ppf for efficiency
        delta_ppf = float(norm.ppf(1 - delta))

        # Compute sampling range
        start_idx = int(init_offset * KL)
        end_idx = KL - int(local_offset * KL)
        sampling_range = end_idx - start_idx

        if sampling_range <= 0:
            raise ValueError(f"Invalid sampling range: {sampling_range}")

        # Get base sample count for variance estimation
        num_base_samples = max(2, int(base_rate_sampling * sampling_range))

        # Create base sampling indices for variance estimation
        base_row_wise_idx = torch.randint(
            low=start_idx,
            high=end_idx,
            size=(B, H, QL, num_base_samples),
            device=expwts.device,
        )

        # Extract values and compute std
        sampled_values = torch.gather(expwts, dim=-1, index=base_row_wise_idx)
        total_rows = B * H * QL
        row_sampled_values = sampled_values.view(total_rows, num_base_samples)
        std_estimate = torch.std(row_sampled_values, dim=-1, keepdim=True)
        std_estimate = torch.clamp(std_estimate, min=1e-8)
        std_estimate = std_estimate.view(B, H, QL, 1)

        # Compute denominators and adaptive budget
        static_denominator = apply_inv_mask_sum(expwts, previous_mask)
        base_data = torch.full_like(base_row_wise_idx, num_base_samples / sampling_range, dtype=expwts.dtype)
        base_sampling_mask = Mask.create_from_row_wise_idx(
            shape=(B, H, QL, KL),
            row_wise_idx=base_row_wise_idx,
            data=base_data,
            type="index",
            dtype=dtype
        )
        sampled_denominator = apply_inv_mask_sum(expwts, base_sampling_mask)
        estimated_denominator = static_denominator + sampled_denominator

        # Compute adaptive budget with statistical bounds
        epsilon_allowable_error = epsilon * estimated_denominator
        epsilon_allowable_error = torch.clamp(epsilon_allowable_error, min=1e-8)
        budget_numerator = delta_ppf * std_estimate * sampling_range
        budget_squared = (budget_numerator / epsilon_allowable_error) ** 2

        # Ensure budget is positive and within bounds
        budget = torch.clamp(
            budget_squared,
            min=1.0,  # Minimum 1 sample
            max=float(sampling_range),  # Maximum sampling_range samples
        ).long()

        # Ensure budget is at least the base samples and respects sampling range
        budget = torch.clamp(budget, min=num_base_samples, max=sampling_range)

        # Add global sparsity constraint to prevent excessive density
        # Limit to maximum 1.5% of keys for very long sequences, 3% for shorter ones
        max_allowed_budget = int(KL * (0.015 if KL > 5000 else 0.03))
        budget = torch.min(budget, torch.tensor(max_allowed_budget, device=device))

        # Create sampling probabilities with minimum threshold
        sampling_probabilities = (budget / sampling_range).to(dtype)
        # Ensure minimum probability is adaptive to sequence length
        min_prob = torch.tensor(1.0 / max(KL, 1024), device=device)
        sampling_probabilities = torch.clamp(sampling_probabilities, min=min_prob)

        # Create adaptive sampling mask
        sampling_mask = create_sampling_mask_with_per_head_budget(
            budgets=budget,
            sampling_probability=sampling_probabilities,
            seq_len_keys=KL,
            start_idx=0,
            end_idx=KL,
            dtype=dtype,
        )

        # 3. Cumulative Merging: Combine all components
        # Previous mask + top-k (heavy hitters) + statistical sampling (tail)
        return previous_mask.merge_mask(topk_mask).merge_mask(sampling_mask)

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "OpenEvolveMasker":
        """Create OpenEvolve masker instance from configuration.

        Args:
            config: Configuration for the OpenEvolve masker.

        Returns:
            Instance of the OpenEvolve masker.

        Raises:
            ValueError: If the config type is invalid.
        """
        return cls(config)
