import logging
import json
import os
import random
import numpy as np
from typing import Dict, Any, List, Union, Optional, Tuple
from openevolve.llm.base import LLMInterface
from openevolve.llm.ensemble import LLMEnsemble
from openevolve.database import Program, ProgramDatabase
from openevolve.config import DebugConfig, HistoryConfig

logger = logging.getLogger(__name__)


class ProgramHistoryLogger:
    """
    Logger for saving program history (abstract and metrics) to JSONL file.
    Each line contains a program's abstract and absolute metrics.
    """
    
    def __init__(self, file_path: str):
        """
        Initialize program history logger.
        
        Args:
            file_path: Path to the history file (JSONL format)
        """
        self.file_path = file_path
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        logger.info(f"Initialized program history logger: {file_path}")
    
    def log_program(
        self,
        program: Program,
        iteration: Optional[int] = None
    ) -> None:
        """
        Log a program's abstract and metrics to the history file.
        
        Args:
            program: Program to log
            iteration: Optional iteration number when this program was created
        """
        if not program.abstract:
            logger.debug(f"Skipping history log: program {program.id} has no abstract")
            return
        
        # Format metrics to ensure all values are JSON-serializable
        try:
            from openevolve.utils.format_utils import format_metrics_dict
            formatted_metrics = format_metrics_dict(program.metrics) if program.metrics else {}
        except ImportError:
            # Fallback if format_utils is not available
            formatted_metrics = {}
            if program.metrics:
                for key, value in program.metrics.items():
                    if isinstance(value, (int, float)):
                        formatted_metrics[key] = float(value)
                    elif isinstance(value, (str, bool, type(None))):
                        formatted_metrics[key] = value
                    else:
                        formatted_metrics[key] = str(value)
        
        # Create history entry
        entry = {
            "program_id": program.id,
            "abstract": program.abstract,
            "metrics": formatted_metrics,
            "iteration": iteration if iteration is not None else program.iteration_found,
            "generation": program.generation,
            "parent_id": program.parent_id,
            "timestamp": program.timestamp,
        }
        
        # Append to JSONL file
        try:
            with open(self.file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write program history entry: {e}")

async def get_summary(
    program: Program, 
    llm: Union[LLMInterface, LLMEnsemble], 
    debug_config: DebugConfig,
    parent_program: Optional[Program] = None,
    parent_abstract: Optional[str] = None
) -> str:
    """
    Generates a high-level summary of the program's logic.
    
    Args:
        program: The program to summarize (child program)
        llm: LLM interface for generation
        debug_config: Debug configuration
        parent_program: Optional parent program for reference
        parent_abstract: Optional parent abstract for reference
    
    Returns:
        A concise summary of the program's core algorithmic approach
    """
    if debug_config.pipeline_debug:
        logger.info('-'*50 + 'Summarization' + '-'*50)

    prompt = {}

    prompt["system"] = (
        "You are an expert code analyst specializing in algorithmic optimization and code comprehension. "
        "Your task is to summarize the core algorithmic idea from code implementations, focusing on "
        "the underlying strategy, data structures, and optimization techniques used in brief and concise manner.")

    prompt["user"] = (
        "# Task Overview:\n"
        "Your goal is to summarize the core algorithmic idea from the child programs.\n\n"

        "# Requirements:\n"
        "1. Concise but dense with technical details.\n"
        "2. Highlight the new or modified aspects (2-4 phrases).\n"
        "3. Keep core shared characteristics from the parent (if available) that are still present in the child (2-4 phrases).\n"
        "4. Each phrase should be less than 8 words.\n\n"
    )
    
    if parent_program is not None and parent_abstract:
        prompt["user"] += (
            # f"# Parent Program Code:\n```python\n{parent_program.code}\n```\n\n"
            f"# Parent Program Abstract:\n{parent_abstract}\n\n"
        )
    prompt["user"] += f"# Child Program Code:\n```python\n{program.code}\n```\n\n"

    prompt["user"] += (
        "# Output Format:\n"
        "Output the summary of the child program directly: \n"
        "- phrase 1\n"
        "- phrase 2\n"
        "- ...\n\n"
    )
    
    if debug_config.get_prompt_debug("summary"):
        logger.info('-'*20 + 'Summary Prompt (System)' + '-'*20 + '\n' + prompt["system"] + '\n' + '-'*50)
        logger.info('-'*20 + 'Summary Prompt (User)' + '-'*20 + '\n' + prompt["user"] + '\n' + '-'*50)

    summary = await llm.generate_with_context(
        system_message=prompt["system"],
        messages=[{"role": "user", "content": prompt["user"]}]
    )

    if debug_config.get_response_debug("summary"):
        logger.info('-'*20 + 'Summary Response' + '-'*20 + '\n' + summary + '\n' + '-'*50)

    return summary.strip()

# async def critique_program(program: Program, llm: Union[LLMInterface, LLMEnsemble], debug_config: DebugConfig) -> str:
#     """
#     Analyzes the program and provides a textual gradient (criticism).
#     """
#     if debug_config.pipeline_debug:
#         logger.info('-'*50 + 'Gradient' + '-'*50)

#     # Handle None values for abstract
#     program_abstract = getattr(program, 'abstract', None) or "No abstract available"
    
#     # Format metrics for better readability (round numeric values to 4 decimal places)
#     from openevolve.utils.format_utils import format_metrics_dict
#     formatted_metrics = format_metrics_dict(program.metrics) if program.metrics else {}
#     metrics_str = json.dumps(formatted_metrics, indent=2) if formatted_metrics else "No metrics available"
    
#     # Analyze which metrics are present and provide targeted guidance
#     metric_guidance = []
#     if program.metrics:
#         if 'latency' in program.metrics or 'time' in str(program.metrics).lower():
#             metric_guidance.append("- **Latency/Time:** Analyze time complexity, identify bottlenecks, and suggest algorithmic optimizations")
#         if 'memory' in str(program.metrics).lower() or 'space' in str(program.metrics).lower():
#             metric_guidance.append("- **Memory/Space:** Analyze space complexity and data structure choices")
#         if 'accuracy' in str(program.metrics).lower() or 'correctness' in str(program.metrics).lower():
#             metric_guidance.append("- **Accuracy/Correctness:** Analyze algorithmic correctness and edge case handling")
#         if 'combined_score' in program.metrics:
#             metric_guidance.append("- **Overall Score:** Prioritize improvements that would have the highest impact on combined_score")
#         if 'efficiency' in str(program.metrics).lower() or 'performance' in str(program.metrics).lower():
#             metric_guidance.append("- **Efficiency/Performance:** Focus on optimization opportunities")
    
#     if not metric_guidance:
#         metric_guidance.append("- Analyze the program's performance characteristics based on the provided metrics")

#     prompt = {}
#     prompt["system"] = (
#         "You are an expert algorithmic strategist and software architect specializing in high-level optimization guidance. "
#         "Your task is to analyze code implementations and provide strategic, high-level improvement directions based on "
#         "evaluation metrics. Focus on **architectural and algorithmic guidance**, not specific code modifications.")
    
#     prompt["user"] = (
#         "# Your Analysis Should Focus On:\n"
#         + "\n".join(metric_guidance) + "\n\n"
        
#         f"# Program Abstract:\n{program_abstract}\n\n"
        
#         f"# Program Code:\n```python\n{program.code}\n```\n\n"
        
#         f"# Evaluation Metrics:\n{metrics_str}\n\n"
        
#         "# Critical Instructions:\n"
#         "1. **Provide HIGH-LEVEL guidance only:** Focus on strategic directions and algorithmic paradigms, not implementation details.\n"
#         "2. **Suggest algorithmic approaches:** Recommend algorithm families or paradigms.\n"
#         "3. **Suggest data structures:** Recommend data structure types.\n"
#         "4. **Suggest architectural patterns:** Recommend code organization approaches.\n"
#         "5. **Suggest optimization strategies:** Recommend high-level optimization approaches.\n"
#         "6. **Avoid specific code changes\n"
#         "7. **Prioritize by impact:** Focus on improvements that would have the highest impact on the evaluation metrics.\n\n"
        
#         "# Output Format:\n"
#         "Provide your critique as high-level strategic guidance:\n"
#         "- Start with the most impactful optimization directions\n"
#         "- Use bullet points for clarity\n"
#         "- Keep each suggestion brief (1-2 sentences)\n"
#         "- Focus on **WHAT** to explore and **WHY** it might help, not **HOW** to implement\n"
#         "- Limit to 1-3 key strategic directions\n"
#         "- Each direction should open up exploration space, not narrow it down\n\n"
#     )

#     if debug_config.get_prompt_debug("gradient"):
#         logger.info('-'*20 + 'Gradient Prompt (System)' + '-'*20 + '\n' + prompt["system"] + '\n' + '-'*50)
#         logger.info('-'*20 + 'Gradient Prompt (User)' + '-'*20 + '\n' + prompt["user"] + '\n' + '-'*50)
    
#     criticism = await llm.generate_with_context(
#         system_message=prompt["system"],
#         messages=[{"role": "user", "content": prompt["user"]}]
#     )

#     if debug_config.get_prompt_debug("gradient"):
#         logger.info('-'*20 + 'Gradient Response' + '-'*20 + '\n' + criticism + '\n' + '-'*50)

#     return criticism.strip()

def normalize_population_metrics(
    programs: List[Program]
) -> List[Dict[str, float]]:
    """
    Normalize metrics for all programs in the population (z-score normalization).
    
    Args:
        programs: List of programs from the population
    
    Returns:
        List of normalized metrics dictionaries (one per program)
    """
    if not programs:
        return []
    
    # Collect all metric names from all programs
    all_metric_names = set()
    for program in programs:
        if program.metrics:
            all_metric_names.update(program.metrics.keys())
    
    if not all_metric_names:
        return [{} for _ in programs]
    
    # Collect metric values for each metric name
    metric_values = {name: [] for name in all_metric_names}
    for program in programs:
        for metric_name in all_metric_names:
            value = program.metrics.get(metric_name, 0.0)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metric_values[metric_name].append(float(value))
            else:
                metric_values[metric_name].append(0.0)
    
    # Calculate mean and std for each metric
    metric_stats = {}
    for metric_name, values in metric_values.items():
        values_array = np.array(values)
        mean = np.mean(values_array)
        std = np.std(values_array)
        # Avoid division by zero
        if std < 1e-10:
            std = 1.0
        metric_stats[metric_name] = {"mean": mean, "std": std}
    
    # Normalize each program's metrics
    normalized_metrics_list = []
    for i, program in enumerate(programs):
        normalized = {}
        for metric_name in all_metric_names:
            value = program.metrics.get(metric_name, 0.0)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                stats = metric_stats[metric_name]
                normalized_value = (float(value) - stats["mean"]) / stats["std"]
                normalized[metric_name] = normalized_value
            else:
                normalized[metric_name] = 0.0
        normalized_metrics_list.append(normalized)
    
    return normalized_metrics_list


def classify_programs_by_normalized_metrics(
    programs: List[Program],
    normalized_metrics: List[Dict[str, float]]
) -> Tuple[List[Program], List[Program], List[Program]]:
    """
    Classify programs into all_improved, all_degraded, and mixed based on normalized metrics.
    
    Args:
        programs: List of programs
        normalized_metrics: List of normalized metrics dictionaries
    
    Returns:
        Tuple of (all_improved_programs, all_degraded_programs, mixed_programs)
    """
    all_improved = []
    all_degraded = []
    mixed = []
    
    epsilon = 1e-10
    
    for program, norm_metrics in zip(programs, normalized_metrics):
        if not norm_metrics:
            mixed.append(program)
            continue
        
        # Get all numeric normalized metric values
        numeric_values = [
            v for v in norm_metrics.values() 
            if isinstance(v, (int, float))
        ]
        
        if not numeric_values:
            mixed.append(program)
            continue
        
        # Classify based on normalized metrics
        if all(v > epsilon for v in numeric_values):
            all_improved.append(program)
        elif all(v < -epsilon for v in numeric_values):
            all_degraded.append(program)
        else:
            mixed.append(program)
    
    return all_improved, all_degraded, mixed


def sample_from_population(
    database: ProgramDatabase,
    batch_size: int,
    weight_all_improved: float = 0.6,
    weight_mixed: float = 0.2,
    weight_all_degraded: float = 0.2,
    debug_config: Optional[DebugConfig] = None
) -> List[Tuple[Program, Dict[str, float]]]:
    """
    Sample programs from the entire population (not just parent's island) with weighted sampling.
    
    Args:
        database: ProgramDatabase instance
        batch_size: Number of programs to sample
        weight_all_improved: Weight for programs where all normalized metrics > 0
        weight_mixed: Weight for programs with mixed normalized metrics
        weight_all_degraded: Weight for programs where all normalized metrics < 0
        debug_config: Optional debug configuration for logging
    
    Returns:
        List of tuples (program, normalized_metrics)
    """
    # Get prompt and response debug flags
    prompt_debug = debug_config.get_prompt_debug("rollout") if debug_config else False
    response_debug = debug_config.get_response_debug("rollout") if debug_config else False
    
    if prompt_debug:
        logger.info('-'*50 + 'Population Sampling (Prompt)' + '-'*50)
    
    # Get all programs from the entire population
    all_programs = list(database.programs.values())
    
    if not all_programs:
        if prompt_debug:
            logger.info("No programs in population, returning empty sample")
        return []
    
    if prompt_debug:
        logger.info(f"Total population size: {len(all_programs)}")
        logger.info(f"Requested batch size: {batch_size}")
    
    if batch_size >= len(all_programs):
        # Return all programs with normalized metrics
        normalized_metrics = normalize_population_metrics(all_programs)
        if prompt_debug:
            logger.info(f"Batch size >= population size, returning all {len(all_programs)} programs")
        return list(zip(all_programs, normalized_metrics)) 

    normalized_metrics = normalize_population_metrics(all_programs)    

    all_improved, all_degraded, mixed = classify_programs_by_normalized_metrics(
        all_programs, normalized_metrics
    )
    
    # Log classification statistics
    if prompt_debug:
        logger.info(f"Classification results:")
        logger.info(f"  - All improved (all normalized metrics > 0): {len(all_improved)} programs")
        logger.info(f"  - Mixed (mixed normalized metrics): {len(mixed)} programs")
        logger.info(f"  - All degraded (all normalized metrics < 0): {len(all_degraded)} programs")
    
    # Normalize weights
    total_weight = weight_all_improved + weight_mixed + weight_all_degraded
    if total_weight > 0:
        weight_all_improved /= total_weight
        weight_mixed /= total_weight
        weight_all_degraded /= total_weight
    
    if prompt_debug:
        logger.info(f"Sampling weights (normalized):")
        logger.info(f"  - All improved: {weight_all_improved:.2%}")
        logger.info(f"  - Mixed: {weight_mixed:.2%}")
        logger.info(f"  - All degraded: {weight_all_degraded:.2%}")
    
    # Calculate target counts for each category
    target_all_improved = max(0, round(batch_size * weight_all_improved))
    target_mixed = max(0, round(batch_size * weight_mixed))
    target_all_degraded = max(0, round(batch_size * weight_all_degraded))
    
    # Adjust if rounding causes mismatch
    total_target = target_all_improved + target_mixed + target_all_degraded
    if total_target != batch_size:
        diff = batch_size - total_target
        # Adjust the largest category
        if target_all_improved >= target_mixed and target_all_improved >= target_all_degraded:
            target_all_improved += diff
        elif target_mixed >= target_all_degraded:
            target_mixed += diff
        else:
            target_all_degraded += diff
    
    if prompt_debug:
        logger.info(f"Target sample counts:")
        logger.info(f"  - All improved: {target_all_improved}")
        logger.info(f"  - Mixed: {target_mixed}")
        logger.info(f"  - All degraded: {target_all_degraded}")
    
    # Sample from each category
    sampled_programs = []
    sampled_indices = set()
    
    # Create index mapping for normalized metrics (use index instead of Program object as key)
    index_to_norm_metrics = dict(enumerate(normalized_metrics))
    
    # Sample all_improved programs
    if all_improved and target_all_improved > 0:
        count = min(target_all_improved, len(all_improved))
        sampled = random.sample(all_improved, count)
        for prog in sampled:
            prog_index = all_programs.index(prog)
            sampled_programs.append((prog, index_to_norm_metrics[prog_index]))
            sampled_indices.add(prog_index)
    
    # Sample mixed programs
    if mixed and target_mixed > 0:
        count = min(target_mixed, len(mixed))
        sampled = random.sample(mixed, count)
        for prog in sampled:
            prog_index = all_programs.index(prog)
            if prog_index not in sampled_indices:
                sampled_programs.append((prog, index_to_norm_metrics[prog_index]))
                sampled_indices.add(prog_index)
    
    # Sample all_degraded programs
    if all_degraded and target_all_degraded > 0:
        count = min(target_all_degraded, len(all_degraded))
        sampled = random.sample(all_degraded, count)
        for prog in sampled:
            prog_index = all_programs.index(prog)
            if prog_index not in sampled_indices:
                sampled_programs.append((prog, index_to_norm_metrics[prog_index]))
                sampled_indices.add(prog_index)
    
    # If we don't have enough, fill with random sampling from remaining
    if len(sampled_programs) < batch_size:
        remaining = batch_size - len(sampled_programs)
        remaining_programs = [
            (prog, index_to_norm_metrics[i])
            for i, prog in enumerate(all_programs)
            if i not in sampled_indices
        ]
        if remaining_programs:
            sampled_programs.extend(
                random.sample(remaining_programs, min(remaining, len(remaining_programs)))
            )

    random.shuffle(sampled_programs)
    
    final_samples = sampled_programs[:batch_size]
    
    # Log sampling results (response debug)
    if response_debug:
        logger.info('-'*50 + 'Population Sampling (Response)' + '-'*50)
        logger.info(f"Final samples: {final_samples}")
        
        # Count actual samples by category
        actual_all_improved = 0
        actual_mixed = 0
        actual_all_degraded = 0
        
        for program, norm_metrics in final_samples:
            numeric_values = [v for v in norm_metrics.values() if isinstance(v, (int, float))]
            epsilon = 1e-10
            if numeric_values:
                if all(v > epsilon for v in numeric_values):
                    actual_all_improved += 1
                elif all(v < -epsilon for v in numeric_values):
                    actual_all_degraded += 1
                else:
                    actual_mixed += 1
        
        logger.info(f"Actual sample distribution:")
        logger.info(f"  - All improved: {actual_all_improved}")
        logger.info(f"  - Mixed: {actual_mixed}")
        logger.info(f"  - All degraded: {actual_all_degraded}")
        
        # Show sample of selected programs
        if final_samples:
            logger.info("Sample of selected programs:")
            for i, (program, norm_metrics) in enumerate(final_samples[:3], 1):
                program_abstract = getattr(program, 'abstract', None) or "No abstract"
                abstract_preview = program_abstract[:100] + "..." if len(program_abstract) > 100 else program_abstract
                logger.info(f"  [{i}] Program {program.id[:8]}...: {abstract_preview}")
                # Show normalized metrics summary
                norm_summary = {k: f"{v:.4f}" for k, v in list(norm_metrics.items())[:3]}
                logger.info(f"      Normalized metrics (sample): {norm_summary}")
        
        logger.info('-'*50 + 'Population Sampling Complete' + '-'*50)
    
    return final_samples


async def get_gradient(
    population_samples: List[Tuple[Program, Dict[str, float]]], 
    llm: Union[LLMInterface, LLMEnsemble], 
    debug_config: DebugConfig,
    use_normalized_metrics: bool = False
) -> str:

    if debug_config.pipeline_debug:
        logger.info('-'*50 + 'Gradient' + '-'*50)

    # Format population samples for the prompt
    history_text = ""
    
    def _format_metric_value(value: float) -> str:
        """
        Format metric values for the gradient prompt.
        - If use_normalized_metrics=True: show sign (+/-) with 4 decimals (z-score space).
        - Else: show raw value with 4 decimals (no forced '+' sign).
        """
        epsilon = 1e-5
        if use_normalized_metrics:
            if abs(value) < epsilon:
                return "0.0000"
            elif value > 0:
                return f"+{value:.4f}"
            else:
                return f"{value:.4f}"
        # Raw metrics
        if abs(value) < epsilon:
            return "0.0000"
        return f"{value:.4f}"
    
    for i, (program, normalized_metrics) in enumerate(population_samples):
        metrics_for_prompt = normalized_metrics if use_normalized_metrics else (program.metrics or {})

        # Determine status: only meaningful in normalized space.
        status = "ABSOLUTE"
        if use_normalized_metrics:
            numeric_values = [
                v for v in normalized_metrics.values()
                if isinstance(v, (int, float))
            ]
            epsilon = 1e-5
            if numeric_values:
                if all(v > epsilon for v in numeric_values):
                    status = "IMPROVEMENT"
                elif all(v < -epsilon for v in numeric_values):
                    status = "REGRESSION"
                else:
                    status = "MIXED"
            else:
                status = "UNKNOWN"
        
        # Format metrics: separate component metrics from combined_score
        component_metrics = {}
        combined_score_value = None
        
        for metric_name, value in metrics_for_prompt.items():
            if metric_name == "combined_score":
                combined_score_value = value
            else:
                component_metrics[metric_name] = value
        
        # Format metrics with proper ordering and signs
        formatted_metrics = []
        
        # Add component metrics first (sorted alphabetically)
        for metric_name in sorted(component_metrics.keys()):
            value = component_metrics[metric_name]
            if isinstance(value, (int, float)):
                formatted_metrics.append(f'"{metric_name}": {_format_metric_value(float(value))}')
        
        # Add combined_score last
        if combined_score_value is not None:
            if isinstance(combined_score_value, (int, float)):
                formatted_metrics.append(f'"combined_score": {_format_metric_value(float(combined_score_value))}')
        
        # Format as JSON-like structure
        if formatted_metrics:
            metrics_str = "{\n " + ",\n ".join(formatted_metrics) + "\n}"
        else:
            metrics_str = "{}"
        
        # Get abstract from program
        program_abstract = getattr(program, 'abstract', None) or "No abstract available"
        
        metrics_label = "Normalized Metrics" if use_normalized_metrics else "Raw Metrics"
        history_text += (
            f"## Attempt {i+1} [{status}]:\n"
            f"Abstract: {program_abstract}\n"
            f"{metrics_label}: {metrics_str}\n\n"
        )

    prompt = {}
    prompt["system"] = (
        "You are an expert algorithmic strategist (Meta-Critic). "
        "Your task is to analyze the recent history of code evolution (both successes and failures) "
        "and synthesize 3-5 brief high-level **Optimization Directions** for the next batch of optimization attempts."
    )

    prompt["user"] = (
        "# Task Overview:\n"
        "Your goal is to synthesize 3-5 brief high-level **Optimization Directions** from the history of optimization for the next batch of attempts.\n\n"

        "# Requirements:\n"
        "1. **Identify Trends:** Identify which directions seem promising vs. dead ends.\n"
        "2. **Potential Ideas:** Though some attempts may failed, the high-level ideas may still be worth exploring.\n"
        "3. **Abstraction:** Do NOT suggest specific code lines. Suggest algorithmic paradigms or architectural patterns.\n"
        "4. **Focus:** The goal is to maximize the evaluation metrics.\n\n"

        "# Population Samples (Metrics):\n"
        f"{history_text}\n"
        
        "Output the directions directly: \n"
    )

    if debug_config.get_prompt_debug("gradient"):
        logger.info('-'*20 + 'Gradient Prompt (System)' + '-'*20 + '\n' + prompt["system"] + '\n' + '-'*50)
        logger.info('-'*20 + 'Gradient Prompt (User)' + '-'*20 + '\n' + prompt["user"] + '\n' + '-'*50)

    
    
    gradient = await llm.generate_with_context(
        system_message=prompt["system"],
        messages=[{"role": "user", "content": prompt["user"]}]
    )

    if debug_config.get_response_debug("gradient"):
        logger.info('-'*20 + 'Gradient Response' + '-'*20 + '\n' + gradient + '\n' + '-'*50)

    return gradient.strip()

async def get_references(
    parent: Program, 
    candidates: list[Program], 
    llm: Union[LLMInterface, LLMEnsemble], 
    debug_config: DebugConfig, 
    k: int = 3,
    candidate_sources: Optional[Dict[str, list[Program]]] = None,
    current_gradient: Optional[str] = None
) -> tuple[list[Program], bool]:
    """
    Selects reference programs from a pre-filtered candidate set based on the parent's
    abstract and gradient. This is a two-stage filtering process:
    1. Pre-filtering: Candidates are selected from island_top_programs, island_previous_programs, and inspirations
    2. LLM selection: LLM selects the best k references from the candidate set
    
    Args:
        parent: The parent program that needs references
        candidates: Pre-filtered candidate programs (from island_top_programs, island_previous_programs, inspirations)
        llm: LLM interface for selection
        debug_config: Debug configuration
        k: Number of references to select
        candidate_sources: Optional dict mapping source names to program lists for context
    
    Returns:
        Tuple of (selected_references, called_llm) where:
        - selected_references: List of selected reference programs
        - called_llm: Boolean indicating whether LLM was actually called (False if no candidates available)
    """
    if debug_config.pipeline_debug:
        logger.info('-'*50 + 'Reference Selection' + '-'*50)
        if candidate_sources:
            for source_name, source_programs in candidate_sources.items():
                logger.info(f"  {source_name}: {len(source_programs)} candidates")

    # Handle None values for abstract and gradient
    parent_abstract = getattr(parent, 'abstract', None) or "No abstract available"

    # parent_code = getattr(parent, 'code', None)
    # if not parent_code or (isinstance(parent_code, str) and not parent_code.strip()):
    #     logger.warning(f"Parent program {parent.id} has no code available. This may indicate a serialization issue.")
    #     parent_program = "No program available"
    # else:
    #     parent_program = parent_code

    # Use provided parent_gradient parameter, fallback to parent.gradient, then to default message
    if current_gradient is None:
        current_gradient = getattr(parent, 'gradient', None)
    current_gradient = current_gradient or "No gradient available"
    
    # Filter out parent from candidates to avoid self-reference
    candidates = [p for p in candidates if p.id != parent.id]
    
    if len(candidates) == 0:
        logger.warning("No candidates available after filtering out parent")
        return [], False  # Return empty list and False (LLM not called)
    
    # If candidates count is less than k, return all candidates without LLM selection
    if len(candidates) < k:
        if debug_config.pipeline_debug:
            logger.info(f"Candidate count ({len(candidates)}) is less than requested k ({k}). Returning all candidates without LLM selection.")
        return candidates, False  # Return all candidates and False (LLM not called)
    
    # Create mapping from integer index to program UUID for easier LLM selection
    index_to_uuid = {i: p.id for i, p in enumerate(candidates)}
    
    # Build candidate list with integer indices, all metrics, and source information
    from openevolve.utils.format_utils import format_metrics_dict
    candidate_list = []
    for i, p in enumerate(candidates):
        abstract = getattr(p, 'abstract', None) or 'N/A'
        formatted_metrics = format_metrics_dict(p.metrics) if p.metrics else {}
        metrics_str = json.dumps(formatted_metrics, indent=2) if formatted_metrics else "No metrics"
        
        candidate_list.append(
            f"**Index {i}**:\n"
            f"- Abstract: {abstract}\n"
            f"- Metrics:\n{metrics_str}\n"
        )
    candidate_abstracts = "\n".join(candidate_list)

    prompt = {}
    prompt["system"] = (
        "You are an expert program selector specializing in evolutionary algorithm optimization. "
        "Your task is to select the best reference programs from a population to guide the evolution "
        "of a parent program based on its current state and improvement needs.")
    
    prompt["user"] = (
        "# Task Overview:\n"
        f"Your goal is to select the best {k} reference programs based on its current state and improvement needs.\n\n"

        "# Parent Program Context:\n"
        f"## Abstract:\n{parent_abstract}\n\n"

        # f"## Program:\n```python\n{parent_program}\n```\n\n"

        f"## Optimization Directions:\n{current_gradient}\n\n"

        f"# Candidate Programs (Total: {len(candidates)}):\n"
        f"{candidate_abstracts}\n\n"

        f"# Selection Criteria:\n"
        f"1. The candidate should logically address or relate to the current optimization directions.\n"
        f"2. Prefer candidates with better evaluation metrics.\n"
        f"3. Avoid selecting multiple programs with identical or similar abstracts. Programs should complement each other.\n"
        f"4. Provide potential, attractive, and innovative ideas.\n"

        f"# Output Format:\n"
        f"You MUST return a JSON object with the following structure:\n"
        f'{{\n'
        f'  "reason": "Brief reasoning statement.",\n'
        f'  "selected_indices": [index1, index2, index3]\n'
        f'}}\n\n'
    )

    if debug_config.get_prompt_debug("sample"):
        logger.info('-'*20 + 'Reference Selection Prompt (System)' + '-'*20 + '\n' + prompt["system"] + '\n' + '-'*50)
        logger.info('-'*20 + 'Reference Selection Prompt (User)' + '-'*20 + '\n' + prompt["user"] + '\n' + '-'*50)

    response = await llm.generate_with_context(
        system_message=prompt["system"],
        messages=[{"role": "user", "content": prompt["user"]}]
    )

    if debug_config.get_response_debug("sample"):
        logger.info('-'*20 + 'Reference Selection Response' + '-'*20 + '\n' + response + '\n' + '-'*50)
    
    try:
        # Parse response as JSON object with thought and selected_indices
        response_data = json.loads(response)
        
        # Handle both old format (list) and new format (object with thought and selected_indices)
        if isinstance(response_data, list):
            # Backward compatibility: if it's a list, use it directly
            selected_indices = response_data
            reasoning = "No reasoning provided (legacy format)"
        elif isinstance(response_data, dict):
            # New format: extract thought and selected_indices
            reasoning = response_data.get("thought", "No reasoning provided")
            selected_indices = response_data.get("selected_indices", [])
            
            if debug_config.get_prompt_debug("sample"):
                logger.info(f"Reference selection reasoning: {reasoning}")
        else:
            raise ValueError("Response is neither a list nor a dict")
        
        # Validate that selected_indices is a list
        if not isinstance(selected_indices, list):
            raise ValueError("selected_indices is not a list")
        
        # Validate length
        if len(selected_indices) != k:
            logger.warning(f"Expected {k} indices, got {len(selected_indices)}")
        
        # Convert indices to UUIDs and then to programs
        selected_programs = []
        seen_indices = set()
        
        for idx in selected_indices:
            if not isinstance(idx, int):
                # Try to convert string numbers to int
                try:
                    idx = int(idx)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid index type in response: {idx} (type: {type(idx)})")
                    continue
            
            # Check for duplicates
            if idx in seen_indices:
                logger.warning(f"Duplicate index {idx} found in selection, skipping")
                continue
            seen_indices.add(idx)
            
            if idx in index_to_uuid:
                uuid = index_to_uuid[idx]
                # Find program by UUID in candidates
                program = next((p for p in candidates if p.id == uuid), None)
                if program:
                    selected_programs.append(program)
                else:
                    logger.warning(f"Program with UUID {uuid} not found in candidates")
            else:
                logger.warning(f"Index {idx} out of range (valid range: 0-{len(candidates)-1})")
        
        # Log reasoning if available
        if debug_config.get_prompt_debug("sample") and isinstance(response_data, dict) and "thought" in response_data:
            logger.info(f"Selection reasoning: {response_data['thought']}")
        
        return selected_programs[:k], True  # Ensure we return at most k programs, and True (LLM was called)
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logger.warning(f"Failed to parse reference selection response: {e}")
        logger.warning(f"Response was: {response}")
        return [], True  # LLM was called but parsing failed

