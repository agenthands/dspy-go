# {"balancedness_score": 0.34443350039798354, "speed_score": 0.06530957148302041, "combined_score": 0.20487153594050198}# SPDX-License-Identifier: Apache-2.0
"""
Expert parallelism load balancer (EPLB) for vLLM.

This module implements the core rearrangement algorithm.

The rearrangement algorithm is adapted from
[DeepSeek EPLB](https://github.com/deepseek-ai/eplb).

Please find at [#12](https://github.com/deepseek-ai/EPLB/issues/12) an example
on how the EPLB algorithm works.
"""

import torch


def balanced_packing(weight: torch.Tensor,
                     num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs using Vectorized Snake Round-Robin.
    This replaces the O(N) loop with O(1) vectorized operations for massive speedup.
    """
    num_layers, num_items = weight.shape
    device = weight.device

    # Use ceiling division to ensure we cover all items even if not divisible
    items_per_pack = (num_items + num_packs - 1) // num_packs

    # Sort weights descending
    sorted_indices = torch.argsort(weight, dim=-1, descending=True)

    # Create a Snake Round-Robin pattern: 0,1,2,3,3,2,1,0,0,1,2,3...
    p_ids = torch.arange(num_packs, device=device)
    pattern = p_ids.unsqueeze(0).repeat(items_per_pack, 1)
    pattern[1::2] = pattern[1::2].flip(dims=[1])

    # Flatten and slice to match the exact number of items
    pack_ids_sorted = pattern.flatten()[:num_items].expand(num_layers, -1)

    # Ranks: [0,0,0,0, 1,1,1,1, ...]
    rank_ids_pattern = torch.arange(items_per_pack, device=device).repeat_interleave(num_packs)
    rank_ids_sorted = rank_ids_pattern[:num_items].expand(num_layers, -1)

    pack_index = torch.empty_like(sorted_indices)
    rank_in_pack = torch.empty_like(sorted_indices)

    pack_index.scatter_(1, sorted_indices, pack_ids_sorted)
    rank_in_pack.scatter_(1, sorted_indices, rank_ids_sorted)

    return pack_index, rank_in_pack


def replicate_experts(
        weight: torch.Tensor,
        num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate experts using an optimized D'Hondt method (Min-Max Greedy).
    Minimizes max load by iteratively assigning replicas.
    Complexity reduced from O(L*N*P) to O(L*N*logN) using proportional estimation + deficit sort.
    """
    num_layers, num_log = weight.shape
    device = weight.device

    # 1. Proportional allocation (estimate ideal counts)
    total_weight = weight.sum(dim=1, keepdim=True) + 1e-9
    ideal = (weight * num_phy) / total_weight
    logcnt = ideal.floor().to(torch.int64)

    # 2. Distribute remaining slots based on D'Hondt marginal gain
    current_sum = logcnt.sum(dim=1, keepdim=True)
    deficit = num_phy - current_sum  # [num_layers, 1]

    next_scores = weight / (logcnt + 1).float()
    sorted_scores, sorted_indices = torch.sort(next_scores, descending=True, dim=1)

    rank_idx = torch.arange(num_log, device=device).expand(num_layers, -1)
    mask = rank_idx < deficit

    extra_counts = torch.zeros_like(logcnt)
    extra_counts.scatter_(1, sorted_indices, mask.long())
    logcnt += extra_counts

    # 3. Construct phy2log and rank mapping
    cumsum = logcnt.cumsum(dim=1)

    phy_indices = torch.arange(num_phy, device=device).expand(num_layers, -1)
    phy2log = torch.searchsorted(cumsum, phy_indices, right=True).clamp_(max=num_log - 1)

    zeros = torch.zeros((num_layers, 1), device=device, dtype=cumsum.dtype)
    starts = torch.cat([zeros, cumsum[:, :-1]], dim=1)

    expert_starts = starts.gather(1, phy2log)
    rank = phy_indices - expert_starts

    return phy2log, rank, logcnt


def rebalance_experts_hierarchical(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
):
    """
    Hierarchical expert-parallelism load balancing.
    """
    num_layers, num_logical_experts = weight.shape
    assert num_logical_experts % num_groups == 0
    group_size = num_logical_experts // num_groups
    assert num_groups % num_nodes == 0
    groups_per_node = num_groups // num_nodes
    assert num_gpus % num_nodes == 0
    assert num_physical_experts % num_gpus == 0
    phy_experts_per_gpu = num_physical_experts // num_gpus

    def inverse(perm: torch.Tensor) -> torch.Tensor:
        inv = torch.empty_like(perm)
        inv.scatter_(
            1,
            perm,
            torch.arange(perm.size(1), dtype=torch.int64,
                         device=perm.device).expand(perm.shape),
        )
        return inv

    # Step 1: pack groups to nodes
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = balanced_packing(
        tokens_per_group, num_nodes)

    log2mlog = (
        ((group_pack_index * groups_per_node + group_rank_in_pack) * group_size)
        .unsqueeze(-1)
        + torch.arange(group_size, dtype=torch.int64,
                       device=group_pack_index.device)
    ).flatten(-2)

    mlog2log = inverse(log2mlog)

    # Step 2: replicate experts within nodes
    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes)

    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes)

    # Step 3: pack physical experts to GPUs
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    pack_index, rank_in_pack = balanced_packing(
        tokens_per_phy, num_gpus // num_nodes)

    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    pphy2mlog = phy2mlog.gather(-1, pphy2phy)
    pphy2mlog = (
        pphy2mlog.view(num_layers, num_nodes, -1)
        + torch.arange(
            0,
            num_logical_experts,
            num_logical_experts // num_nodes,
            device=group_pack_index.device,
        ).view(1, -1, 1)
    ).flatten(-2)

    pphy2log = mlog2log.gather(-1, pphy2mlog)
    pphyrank = phyrank.gather(-1, pphy2phy).view(num_layers, -1)
    logcnt = mlogcnt.view(num_layers, -1).gather(-1, log2mlog)

    return pphy2log, pphyrank, logcnt


def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Entry point for expert-parallelism load balancer.
    """
    num_layers, num_logical_experts = weight.shape
    weight = weight.float()

    if num_groups % num_nodes == 0:
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus)
    else:
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus)

    maxlogcnt = int(logcnt.max().item())
    log2phy = torch.full(
        (num_layers, num_logical_experts, maxlogcnt),
        -1,
        dtype=torch.int64,
        device=logcnt.device,
    )

    flat_phy_indices = phy2log * maxlogcnt + phyrank
    phy_ids = torch.arange(num_replicas, device=logcnt.device).expand(num_layers, -1)
    log2phy.view(num_layers, -1).scatter_(1, flat_phy_indices, phy_ids)

    return phy2log, log2phy, logcnt


__all__ = ["rebalance_experts"]
