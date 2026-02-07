import time
import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3


def get_best_schedule(workload, num_seqs):
    """
    Get optimal schedule using Beam Search with Diversity Pruning.

    This approach maintains multiple candidate schedules and prunes based on 
    makespan and transaction footprints to avoid local optima.
    """
    beam_width = 8  # Balanced for time limit

    # Initialize beam with single transactions
    initial_candidates = []
    for i in range(workload.num_txns):
        cost = workload.get_opt_seq_cost([i])
        remaining = set(range(workload.num_txns))
        remaining.remove(i)
        initial_candidates.append((cost, [i], remaining))

    # Sort and take top starters
    initial_candidates.sort(key=lambda x: x[0])
    beam = initial_candidates[:beam_width]

    # Iteratively build the schedule
    for _ in range(workload.num_txns - 1):
        next_gen = []
        for current_cost, seq, remaining in beam:
            for next_txn in remaining:
                new_seq = seq + [next_txn]
                # Incremental cost evaluation
                new_cost = workload.get_opt_seq_cost(new_seq)

                new_remaining = remaining.copy()
                new_remaining.remove(next_txn)
                next_gen.append((new_cost, new_seq, new_remaining))

        # Diversity Pruning: Sort by cost, but keep unique end-transactions if costs are close
        next_gen.sort(key=lambda x: x[0])

        # Filter to keep unique "footprints" (last transaction added) to maintain beam diversity
        seen_ends = set()
        beam = []
        for cand in next_gen:
            last_txn = cand[1][-1]
            if last_txn not in seen_ends:
                beam.append(cand)
                seen_ends.add(last_txn)
            if len(beam) >= beam_width:
                break

    # Return the best complete schedule found
    best_cost, best_schedule, _ = beam[0]
    return best_cost, best_schedule


def get_random_costs():
    start_time = time.time()
    workload_size = 100
    workload = Workload(WORKLOAD_1)

    makespan1, schedule1 = get_best_schedule(workload, 10)
    cost1 = workload.get_opt_seq_cost(schedule1)

    workload2 = Workload(WORKLOAD_2)
    makespan2, schedule2 = get_best_schedule(workload2, 10)
    cost2 = workload2.get_opt_seq_cost(schedule2)

    workload3 = Workload(WORKLOAD_3)
    makespan3, schedule3 = get_best_schedule(workload3, 10)
    cost3 = workload3.get_opt_seq_cost(schedule3)
    print(cost1, cost2, cost3)
    return cost1 + cost2 + cost3, [schedule1, schedule2, schedule3], time.time() - start_time


if __name__ == "__main__":
    makespan, schedule, time = get_random_costs()
    print(f"Makespan: {makespan}, Time: {time}")
