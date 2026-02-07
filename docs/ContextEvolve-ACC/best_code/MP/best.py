GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START
def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.

    Args:
        gpu_num: Number of GPUs
        models: List of models to place

    Returns:
        A placement of models to GPUs
    """

    # 1) Sort models by a hybrid score of load intensity and size to handle heavy models first
    sorted_models = sorted(models, key=lambda m: (m.req_rate / m.slo) * m.model_size, reverse=True)

    # 2) Initialize per-GPU states
    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    free_mem = [GPU_MEM_SIZE for _ in range(gpu_num)]
    total_load = [0.0 for _ in range(gpu_num)]

    # 3) Greedy placement: pick the GPU that results in the minimum peak KVPR
    for model in sorted_models:
        best_idx = -1
        min_kvpr = float('inf')
        m_load = model.req_rate / model.slo

        for gpu_id in range(gpu_num):
            if free_mem[gpu_id] > model.model_size:
                # KVPR = sum(load) / (GPU_MEM_SIZE - sum(model_size))
                projected_kvpr = (total_load[gpu_id] + m_load) / (free_mem[gpu_id] - model.model_size)

                if projected_kvpr < min_kvpr:
                    min_kvpr = projected_kvpr
                    best_idx = gpu_id

        if best_idx == -1:
            raise ValueError(f"Insufficient memory for model size {model.model_size}")

        placement[best_idx].append(model)
        total_load[best_idx] += m_load
        free_mem[best_idx] -= model.model_size

    return placement
# EVOLVE-BLOCK-END


if __name__ == "__main__":
    # Test the algorithm

    from evaluator import generate_test_gpu_models
    from evaluator import calculate_kvcache_pressure
    from evaluator import safe_float
    import numpy as np

    test_cases = generate_test_gpu_models()
    all_kvpr = []
    for i, (gpu_num, gpu_models) in enumerate(test_cases):

        results = compute_model_placement(gpu_num, gpu_models)
        max_kvpr = calculate_kvcache_pressure(results)
        all_kvpr.append(safe_float(max_kvpr))

    avg_kvpr = np.mean(all_kvpr)
    if avg_kvpr != 0:
        avg_kvpr = 1.0 / avg_kvpr

    print(f"Max KVPR: {avg_kvpr:.3f}")
