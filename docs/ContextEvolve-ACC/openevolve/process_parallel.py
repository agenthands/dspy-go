"""
Process-based parallel controller for true parallelism
"""

import asyncio
import logging
import multiprocessing as mp
import os
import pickle
import signal
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, Future
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase
from openevolve.rl_agents import get_references, get_gradient, get_summary, sample_from_population, ProgramHistoryLogger

from openevolve.evaluator import Evaluator
from openevolve.llm.ensemble import LLMEnsemble
from openevolve.prompt.sampler import PromptSampler

from openevolve.config import Config, DatabaseConfig, EvaluatorConfig, LLMConfig, PromptConfig, LLMModelConfig, DebugConfig, MultiAgentConfig, AgentLLMConfig

from openevolve.utils.metrics_utils import safe_numeric_average
from openevolve.utils.code_utils import extract_diffs, apply_diff, format_diff_summary, parse_full_rewrite

logger = logging.getLogger(__name__)


@dataclass
class SerializableResult:
    """Result that can be pickled and sent between processes"""
    child_program_dict: Optional[Dict[str, Any]] = None
    parent_id: Optional[str] = None
    iteration_time: float = 0.0
    prompt: Optional[Dict[str, str]] = None
    llm_response: Optional[str] = None
    artifacts: Optional[Dict[str, Any]] = None
    iteration: int = 0
    error: Optional[str] = None
    token_usage: Optional[Dict[str, Dict[str, int]]] = None  # Token usage per agent: {"evolve": {...}, "sample": {...}, etc.}
    time_usage: Optional[Dict[str, float]] = None  # Time usage per agent in seconds: {"evolve": 1.23, "sample": 0.45, etc.}
    model_token_usage: Optional[Dict[str, Dict[str, int]]] = None  # Token usage per model: {"model_name": {"prompt_tokens": X, ...}, ...}
    model_time_usage: Optional[Dict[str, float]] = None  # Time usage per model in seconds: {"model_name": 1.23, ...}


def _worker_init(config_dict: dict, evaluation_file: str) -> None:
    """Initialize worker process with necessary components"""
    global _worker_config
    global _worker_evaluation_file
    global _worker_evaluator
    global _worker_llm_ensemble
    global _worker_prompt_sampler
    
    # Store config for later use
    # Reconstruct Config object from nested dictionaries
    
    
    # Reconstruct model objects
    models = [LLMModelConfig(**m) for m in config_dict['llm']['models']]
    evaluator_models = [LLMModelConfig(**m) for m in config_dict['llm']['evaluator_models']]
    
    # Create LLM config with models
    llm_dict = config_dict['llm'].copy()
    llm_dict['models'] = models
    llm_dict['evaluator_models'] = evaluator_models
    llm_config = LLMConfig(**llm_dict)
    
    # Helper function to process environment variables
    def process_env_var(value):
        """Process environment variable references like ${VAR_NAME}"""
        if isinstance(value, str):
            value = value.strip()
            if value.startswith("${") and value.endswith("}"):
                env_var_name = value[2:-1]
                env_value = os.environ.get(env_var_name)
                if env_value is not None:
                    return env_value
        return value
    
    # Process environment variables in LLM config
    llm_dict['api_key'] = process_env_var(llm_dict.get('api_key'))
    for model in models:
        if model.api_key:
            model.api_key = process_env_var(model.api_key)
    for model in evaluator_models:
        if model.api_key:
            model.api_key = process_env_var(model.api_key)
    
    # Reconstruct multi-agent config if present
    multi_agent_config = None
    if 'multi_agent' in config_dict:
        multi_agent_dict = config_dict['multi_agent']
        multi_agent_config = MultiAgentConfig()
        for agent_name in ['evolve', 'summary', 'gradient', 'sample']:
            if agent_name in multi_agent_dict:
                agent_dict = multi_agent_dict[agent_name]
                agent_config = AgentLLMConfig()
                
                # Process environment variables for agent-level api_key
                if 'api_key' in agent_dict:
                    agent_dict['api_key'] = process_env_var(agent_dict['api_key'])
                
                for key, value in agent_dict.items():
                    if key == 'models' and isinstance(value, list):
                        # Process environment variables for each model's api_key
                        processed_models = []
                        for m in value:
                            model_dict = dict(m)
                            if 'api_key' in model_dict:
                                model_dict['api_key'] = process_env_var(model_dict['api_key'])
                            processed_models.append(LLMModelConfig(**model_dict))
                        agent_config.models = processed_models
                    elif hasattr(agent_config, key):
                        setattr(agent_config, key, value)
                agent_config.__post_init__()
                setattr(multi_agent_config, agent_name, agent_config)
    
    # Create other configs
    prompt_config = PromptConfig(**config_dict['prompt'])
    database_config = DatabaseConfig(**config_dict['database'])
    evaluator_config = EvaluatorConfig(**config_dict['evaluator'])
    # Handle debug config - use default if not present
    debug_dict = config_dict.get('debug', {})
    debug_config = DebugConfig(**debug_dict)
    
    _worker_config = Config(
        llm=llm_config,
        prompt=prompt_config,
        database=database_config,
        evaluator=evaluator_config,
        debug=debug_config,
        multi_agent=multi_agent_config if multi_agent_config else MultiAgentConfig(),
        **{k: v for k, v in config_dict.items() 
           if k not in ['llm', 'prompt', 'database', 'evaluator', 'debug', 'multi_agent']}
    )
    _worker_evaluation_file = evaluation_file
    
    # These will be lazily initialized on first use
    _worker_evaluator = None
    _worker_llm_ensemble = None  
    _worker_prompt_sampler = None
    _worker_summary_llm = None
    _worker_gradient_llm = None
    _worker_sample_llm = None


def _lazy_init_worker_components():
    """Lazily initialize expensive components on first use"""
    global _worker_evaluator
    global _worker_llm_ensemble
    global _worker_prompt_sampler
    global _worker_summary_llm
    global _worker_gradient_llm
    global _worker_sample_llm
    
    if _worker_llm_ensemble is None:
        # Use evolve agent config if available, otherwise use default
        evolve_config = _worker_config.multi_agent.get_agent_config("evolve", _worker_config.llm)
        _worker_llm_ensemble = LLMEnsemble(evolve_config.models)
    
    # Initialize agent-specific LLMs for RL mode
    if _worker_config.rl_mode:
        if not hasattr(_lazy_init_worker_components, '_rl_agents_initialized'):
            summary_config = _worker_config.multi_agent.get_agent_config("summary", _worker_config.llm)
            gradient_config = _worker_config.multi_agent.get_agent_config("gradient", _worker_config.llm)
            sample_config = _worker_config.multi_agent.get_agent_config("sample", _worker_config.llm)
            
            _worker_summary_llm = LLMEnsemble(summary_config.models)
            _worker_gradient_llm = LLMEnsemble(gradient_config.models)
            _worker_sample_llm = LLMEnsemble(sample_config.models)
            _lazy_init_worker_components._rl_agents_initialized = True
    
    if _worker_prompt_sampler is None:
        _worker_prompt_sampler = PromptSampler(_worker_config.prompt)
    
    if _worker_evaluator is None:

        
        # Create evaluator-specific components
        evaluator_llm = LLMEnsemble(_worker_config.llm.evaluator_models)
        evaluator_prompt = PromptSampler(_worker_config.prompt)
        evaluator_prompt.set_templates("evaluator_system_message")
        
        _worker_evaluator = Evaluator(
            _worker_config.evaluator,
            _worker_evaluation_file,
            evaluator_llm,
            evaluator_prompt,
            database=None  # No shared database in worker
        )


def _run_iteration_worker(
    iteration: int,
    db_snapshot: Dict[str, Any],
    parent_id: str,
    inspiration_ids: List[str]
) -> SerializableResult:
    """Run a single iteration in a worker process"""
    try:
        # Lazy initialization
        _lazy_init_worker_components()
        
        # Reconstruct programs from snapshot
        programs = {
            pid: Program(**prog_dict) 
            for pid, prog_dict in db_snapshot["programs"].items()
        }
        
        parent = programs[parent_id]
        inspirations = [programs[pid] for pid in inspiration_ids if pid in programs]
        
        # Log parent source if pipeline_debug is enabled
        if _worker_config.debug.pipeline_debug:
            parent_source = parent.metadata.get("parent_source", "unknown")
            parent_island_info = parent.metadata.get("island", "unknown")
            logger.info(
                f"Parent source: {parent_source} | "
                f"Parent ID: {parent_id} | "
                f"Parent island: {parent_island_info} | "
                f"Current island: {db_snapshot['current_island']}"
            )
        
        # Get parent artifacts if available
        parent_artifacts = db_snapshot["artifacts"].get(parent_id)
        
        # Get island-specific programs for context
        parent_island = parent.metadata.get("island", db_snapshot["current_island"])
        island_programs = [
            programs[pid] for pid in db_snapshot["islands"][parent_island]
            if pid in programs
        ]
        
        # Sort by metrics for top programs
        island_programs.sort(
            key=lambda p: p.metrics.get("combined_score", safe_numeric_average(p.metrics)),
            reverse=True
        )
        
        # Use config values for limits instead of hardcoding
        island_top_programs = island_programs[:_worker_config.prompt.num_top_programs + _worker_config.prompt.num_diverse_programs]
        top_program_ids = {p.id for p in island_top_programs}
        parent_program_id = parent.id
        
        # Get previous programs: exclude top programs and parent
        # If island has few programs, allow including some top programs (but not parent)
        non_top_programs = [
            p for p in island_programs 
            if p.id not in top_program_ids and p.id != parent_program_id
        ]
        
        if len(non_top_programs) >= _worker_config.prompt.num_top_programs:
            # Enough non-top programs, use them
            island_previous_programs = non_top_programs[:_worker_config.prompt.num_top_programs]
        else:
            # Not enough non-top programs, supplement with top programs (excluding parent)
            island_previous_programs = non_top_programs.copy()
            remaining_slots = _worker_config.prompt.num_top_programs - len(island_previous_programs)
            if remaining_slots > 0:
                # Add top programs that are not the parent
                top_programs_excluding_parent = [
                    p for p in island_top_programs 
                    if p.id != parent_program_id
                ]
                # Take only what we need, avoiding duplicates
                for p in top_programs_excluding_parent:
                    if len(island_previous_programs) >= _worker_config.prompt.num_top_programs:
                        break
                    if p.id not in {prev.id for prev in island_previous_programs}:
                        island_previous_programs.append(p)
        
        # Initialize token usage and time usage tracking early (needed for RL mode)
        token_usage = {}
        time_usage = {}
        
        if _worker_config.rl_mode:
            # Stage 1: Pre-filter candidates from island_top_programs, island_previous_programs, and inspirations
            candidate_set = {}
            all_source_programs = island_top_programs + island_previous_programs + inspirations
            
            if _worker_config.debug.pipeline_debug:
                top_ids = {p.id for p in island_top_programs}
                previous_ids = {p.id for p in island_previous_programs}
                inspiration_ids = {p.id for p in inspirations}
                overlap_top_previous = top_ids & previous_ids
                overlap_top_inspiration = top_ids & inspiration_ids
                overlap_previous_inspiration = previous_ids & inspiration_ids
                if overlap_top_previous or overlap_top_inspiration or overlap_previous_inspiration:
                    logger.debug(
                        f"Candidate source overlaps: "
                        f"top∩previous={len(overlap_top_previous)}, "
                        f"top∩inspiration={len(overlap_top_inspiration)}, "
                        f"previous∩inspiration={len(overlap_previous_inspiration)}"
                    )
            
            for p in all_source_programs:
                if p.id != parent_program_id: 
                    candidate_set[p.id] = p
            
            candidates = list(candidate_set.values())
            
            # Debug: log filtering results
            if _worker_config.debug.pipeline_debug:
                filtered_out_parent = sum(1 for p in all_source_programs if p.id == parent_program_id)
                duplicates_removed = len(all_source_programs) - len(candidate_set) - filtered_out_parent
                if filtered_out_parent > 0 or duplicates_removed > 0:
                    logger.debug(
                        f"Candidate filtering: "
                        f"total_source={len(all_source_programs)}, "
                        f"filtered_parent={filtered_out_parent}, "
                        f"duplicates_removed={duplicates_removed}, "
                        f"final_candidates={len(candidates)}"
                    )
            
            candidate_sources = {
                "island_top_programs": island_top_programs,
                "island_previous_programs": island_previous_programs,
                "inspirations": inspirations
            }
            
            # Stage 2: LLM selects k references from the pre-filtered candidate set
            # Use sample agent LLM if available, otherwise use default
            _lazy_init_worker_components()  # Ensure RL agents are initialized
            # Check if RL agents were initialized by checking if the global variable is set
            sample_llm = _worker_sample_llm if (_worker_config.rl_mode and _worker_sample_llm is not None) else _worker_llm_ensemble
            num_references = _worker_config.prompt.num_references
            
            references, called_llm = asyncio.run(get_references(
                parent, 
                candidates, 
                sample_llm, 
                _worker_config.debug,
                k=num_references,  # Number of references to select (from config)
                candidate_sources=candidate_sources,
                current_gradient=db_snapshot.get("current_gradient")  # Pass the consolidated gradient
            ))
            
            # Record token usage and time usage for sample agent (only if LLM was actually called)
            if called_llm:
                sample_usage = sample_llm.get_last_usage()
                sample_time = sample_llm.get_last_time()
                token_usage["sample"] = sample_usage
                time_usage["sample"] = sample_time
                if _worker_config.debug.pipeline_debug:
                    logger.info(f"Token usage (sample): prompt={sample_usage['prompt_tokens']}, completion={sample_usage['completion_tokens']}, total={sample_usage['total_tokens']}")
                    logger.info(f"Time usage (sample): {sample_time:.3f}s")
            else:
                # No LLM call, set token usage and time usage to 0
                token_usage["sample"] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                time_usage["sample"] = 0.0
                if _worker_config.debug.pipeline_debug:
                    logger.info(f"Token usage (sample): prompt=0, completion=0, total=0 (no LLM call - no candidates available)")
                    logger.info(f"Time usage (sample): 0.000s (no LLM call)")
            reference_programs_dict = [p.to_dict() for p in references]
            
            # Convert candidate_sources to dict format for reference_sources
            reference_sources = {
                "island_previous_programs": [p.to_dict() for p in island_previous_programs],
                "island_top_programs": [p.to_dict() for p in island_top_programs],
                "inspirations": [p.to_dict() for p in inspirations]
            }
            
            # RL mode: Only pass reference_programs, not previous_programs/top_programs/inspirations
            prompt = _worker_prompt_sampler.build_prompt(
                current_program=parent.code,
                parent_program=parent.code,
                program_metrics=parent.metrics,
                previous_programs=[],  # Not used in RL mode
                top_programs=[],  # Not used in RL mode
                inspirations=[],  # Not used in RL mode
                language=_worker_config.language,
                evolution_round=iteration,
                diff_based_evolution=_worker_config.diff_based_evolution,
                program_artifacts=parent_artifacts,
                parent_gradient=db_snapshot.get("current_gradient"), # Use consolidated gradient
                reference_programs=reference_programs_dict,
                reference_sources=reference_sources,
            )
        else:
            # Non-RL mode: Use traditional evolution history
            prompt = _worker_prompt_sampler.build_prompt(
                current_program=parent.code,
                parent_program=parent.code,
                program_metrics=parent.metrics,
                previous_programs=[p.to_dict() for p in island_previous_programs],
                top_programs=[p.to_dict() for p in island_top_programs],
                inspirations=[p.to_dict() for p in inspirations],
                language=_worker_config.language,
                evolution_round=iteration,
                diff_based_evolution=_worker_config.diff_based_evolution,
                program_artifacts=parent_artifacts,
                parent_gradient=parent.gradient if parent.gradient else None,
                reference_programs=[],  # Empty in non-RL mode
                reference_sources=None,  # None in non-RL mode
            )
        
        iteration_start = time.time()
        
        if _worker_config.debug.pipeline_debug:
            logger.info('-'*50 + 'Evolution' + '-'*50)
        if _worker_config.debug.get_prompt_debug("evolve"):
            logger.info('-'*20 + 'Evolution Prompt (System)' + '-'*20 + '\n' + prompt["system"] + '\n' + '-'*50)
            logger.info('-'*20 + 'Evolution Prompt (User)' + '-'*20 + '\n' + prompt["user"] + '\n' + '-'*50)

        # Generate code modification (sync wrapper for async)
        llm_response = asyncio.run(
            _worker_llm_ensemble.generate_with_context(
                system_message=prompt["system"],
                messages=[{"role": "user", "content": prompt["user"]}],
            )
        )
        
        # Record token usage and time usage for evolve agent (always track, not just for debug)
        evolve_usage = _worker_llm_ensemble.get_last_usage()
        evolve_time = _worker_llm_ensemble.get_last_time()
        token_usage["evolve"] = evolve_usage
        time_usage["evolve"] = evolve_time
        if _worker_config.debug.pipeline_debug:
            logger.info(f"Token usage (evolve): prompt={evolve_usage['prompt_tokens']}, completion={evolve_usage['completion_tokens']}, total={evolve_usage['total_tokens']}")
            logger.info(f"Time usage (evolve): {evolve_time:.3f}s")

        if _worker_config.debug.get_response_debug("evolve"):
            logger.info('-'*20 + 'Evolution Response' + '-'*20 + '\n' + llm_response + '\n' + '-'*50)
        
        # Parse response based on evolution mode
        if _worker_config.diff_based_evolution:

            
            diff_blocks = extract_diffs(llm_response)
            if not diff_blocks:
                return SerializableResult(
                    error=f"No valid diffs found in response",
                    iteration=iteration
                )
            
            child_code = apply_diff(parent.code, llm_response)
            changes_summary = format_diff_summary(diff_blocks)
        else:
            new_code = parse_full_rewrite(llm_response, _worker_config.language)
            if not new_code:
                return SerializableResult(
                    error=f"No valid code found in response",
                    iteration=iteration
                )
            
            child_code = new_code
            changes_summary = "Full rewrite"
        
        # Check code length
        if len(child_code) > _worker_config.max_code_length:
            return SerializableResult(
                error=f"Generated code exceeds maximum length ({len(child_code)} > {_worker_config.max_code_length})",
                iteration=iteration
            )
        
        # Evaluate the child program
        child_id = str(uuid.uuid4())
        child_metrics = asyncio.run(
            _worker_evaluator.evaluate_program(child_code, child_id)
        )
        
        # Get artifacts
        artifacts = _worker_evaluator.get_pending_artifacts(child_id)
        
        # Create child program
        child_program = Program(
            id=child_id,
            code=child_code,
            language=_worker_config.language,
            parent_id=parent.id,
            generation=parent.generation + 1,
            metrics=child_metrics,
            iteration_found=iteration,
            metadata={
                "changes": changes_summary,
                "parent_metrics": parent.metrics,
                "island": parent_island,
            },
            provenance={
                "parent_abstract": parent.abstract,
                "parent_gradient": parent.gradient,
            }
        )

        # RL-enhanced features
        if _worker_config.rl_mode:
            _lazy_init_worker_components()
            # Use agent-specific LLMs if available
            summary_llm = _worker_summary_llm if (_worker_config.rl_mode and _worker_summary_llm is not None) else _worker_llm_ensemble
            
            # Summarize
            summary = asyncio.run(get_summary(
                child_program, 
                summary_llm, 
                _worker_config.debug,
                parent_program=parent,
                parent_abstract=parent.abstract if parent.abstract else None
            ))
            child_program.abstract = summary
            
            # Record token usage and time usage for summary agent (always track, not just for debug)
            summary_usage = summary_llm.get_last_usage()
            summary_time = summary_llm.get_last_time()
            token_usage["summary"] = summary_usage
            time_usage["summary"] = summary_time
            if _worker_config.debug.pipeline_debug:
                logger.info(f"Token usage (summary): prompt={summary_usage['prompt_tokens']}, completion={summary_usage['completion_tokens']}, total={summary_usage['total_tokens']}")
                logger.info(f"Time usage (summary): {summary_time:.3f}s")
        
        iteration_time = time.time() - iteration_start
        
        # Log token usage and time usage summary if pipeline_debug is enabled
        if _worker_config.debug.pipeline_debug and token_usage:
            total_prompt = sum(u.get("prompt_tokens", 0) for u in token_usage.values())
            total_completion = sum(u.get("completion_tokens", 0) for u in token_usage.values())
            total_tokens = sum(u.get("total_tokens", 0) for u in token_usage.values())
            total_time = sum(time_usage.values()) if time_usage else 0.0
            logger.info(f"=== Iteration {iteration} Token Usage Summary ===")
            for agent_name, usage in token_usage.items():
                agent_time = time_usage.get(agent_name, 0.0)
                logger.info(f"  {agent_name}: prompt={usage['prompt_tokens']}, completion={usage['completion_tokens']}, total={usage['total_tokens']}, time={agent_time:.3f}s")
            logger.info(f"  TOTAL: prompt={total_prompt}, completion={total_completion}, total={total_tokens}, time={total_time:.3f}s")
            logger.info("=" * 50)
        
        # Collect per-model token and time stats from all worker LLM ensembles
        model_token_usage = {}
        model_time_usage = {}
        
        # Collect from main evolution LLM ensemble
        if _worker_llm_ensemble:
            for model_name, stats in _worker_llm_ensemble.get_token_stats().items():
                if model_name not in model_token_usage:
                    model_token_usage[model_name] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                model_token_usage[model_name]["prompt_tokens"] += stats.get("prompt_tokens", 0)
                model_token_usage[model_name]["completion_tokens"] += stats.get("completion_tokens", 0)
                model_token_usage[model_name]["total_tokens"] += stats.get("total_tokens", 0)
            for model_name, model_time in _worker_llm_ensemble.get_time_stats().items():
                model_time_usage[model_name] = model_time_usage.get(model_name, 0.0) + model_time
        
        # Collect from RL agent LLM ensembles
        if _worker_config.rl_mode:
            for ensemble in [_worker_summary_llm, _worker_sample_llm]:
                if ensemble:
                    for model_name, stats in ensemble.get_token_stats().items():
                        if model_name not in model_token_usage:
                            model_token_usage[model_name] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                        model_token_usage[model_name]["prompt_tokens"] += stats.get("prompt_tokens", 0)
                        model_token_usage[model_name]["completion_tokens"] += stats.get("completion_tokens", 0)
                        model_token_usage[model_name]["total_tokens"] += stats.get("total_tokens", 0)
                    for model_name, model_time in ensemble.get_time_stats().items():
                        model_time_usage[model_name] = model_time_usage.get(model_name, 0.0) + model_time
        
        return SerializableResult(
            child_program_dict=child_program.to_dict(),
            parent_id=parent.id,
            iteration_time=iteration_time,
            prompt=prompt,
            llm_response=llm_response,
            artifacts=artifacts,
            iteration=iteration,
            token_usage=token_usage if token_usage else None,
            time_usage=time_usage if time_usage else None,
            model_token_usage=model_token_usage if model_token_usage else None,
            model_time_usage=model_time_usage if model_time_usage else None
        )
        
    except Exception as e:
        logger.exception(f"Error in worker iteration {iteration}")
        return SerializableResult(
            error=str(e),
            iteration=iteration
        )


class ProcessParallelController:
    """Controller for process-based parallel evolution"""
    
    def __init__(self, config: Config, evaluation_file: str, database: ProgramDatabase, history_logger: Optional["ProgramHistoryLogger"] = None, curve_file_path: Optional[str] = None, initial_curve_data: Optional[List[Dict[str, Any]]] = None):
        self.config = config
        self.evaluation_file = evaluation_file
        self.database = database
        self.history_logger = history_logger
        self.curve_file_path = curve_file_path
        
        self.executor: Optional[ProcessPoolExecutor] = None
        self.shutdown_event = mp.Event()
        
        # Number of worker processes
        self.num_workers = config.evaluator.parallel_evaluations
        
        # Initialize cumulative token and time usage tracking (always, not just for pipeline_debug)
        self.cumulative_token_usage: Dict[str, Dict[str, int]] = {}
        self.cumulative_time_usage: Dict[str, float] = {}
        
        # Initialize cumulative per-model token and time usage tracking
        self.cumulative_model_token_usage: Dict[str, Dict[str, int]] = {}
        self.cumulative_model_time_usage: Dict[str, float] = {}
        
        # Initialize curve data for tracking best solution per iteration
        # Start with initial curve data if provided (e.g., from iteration 0)
        self.curve_data: List[Dict[str, Any]] = initial_curve_data.copy() if initial_curve_data else []
        
        # --- New attributes for periodic gradient synthesis ---
        self.current_gradient: Optional[str] = "No gradient generated yet. Perform broad exploration."
        self.gradient_llm: Optional[LLMEnsemble] = None
        if self.config.rl_mode:
            
            gradient_config = self.config.multi_agent.get_agent_config("gradient", self.config.llm)
            self.gradient_llm = LLMEnsemble(gradient_config.models)
            # Get gradient interval from gradient agent config, fallback to default
            gradient_agent_config = self.config.multi_agent.gradient
            self.gradient_interval = (
                gradient_agent_config.interval 
                if gradient_agent_config and gradient_agent_config.interval is not None
                else getattr(self.config, "gradient_interval", 10)
            )
            # Get batch size from history config, fallback to default
            history_config = self.config.multi_agent.history
            self.gradient_batch_size = (
                history_config.rollout_batch_size
                if history_config
                else getattr(self.config, "gradient_batch_size", 20)
            )
            # Track gradient token usage and time usage
            self.gradient_token_usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            self.gradient_time_usage = 0.0
        # --- End of new attributes ---
        
        logger.info(f"Initialized process parallel controller with {self.num_workers} workers")
    
    def _serialize_config(self, config: Config) -> dict:
        """Serialize config object to a dictionary that can be pickled"""
        # Manual serialization to handle nested objects properly
        result = {
            'llm': {
                'models': [asdict(m) for m in config.llm.models],
                'evaluator_models': [asdict(m) for m in config.llm.evaluator_models],
                'api_base': config.llm.api_base,
                'api_key': config.llm.api_key,
                'temperature': config.llm.temperature,
                'top_p': config.llm.top_p,
                'max_tokens': config.llm.max_tokens,
                'timeout': config.llm.timeout,
                'retries': config.llm.retries,
                'retry_delay': config.llm.retry_delay,
            },
            'prompt': asdict(config.prompt),
            'database': asdict(config.database),
            'evaluator': asdict(config.evaluator),
            'debug': asdict(config.debug),
            'max_iterations': config.max_iterations,
            'checkpoint_interval': config.checkpoint_interval,
            'log_level': config.log_level,
            'log_dir': config.log_dir,
            'random_seed': config.random_seed,
            'diff_based_evolution': config.diff_based_evolution,
            'rl_mode': config.rl_mode,
            'max_code_length': config.max_code_length,
            'language': config.language,
        }
        
        # Serialize multi-agent config if present
        if config.multi_agent:
            multi_agent_dict = {}
            for agent_name in ['evolve', 'summary', 'gradient', 'sample']:
                agent_config = getattr(config.multi_agent, agent_name, None)
                if agent_config:
                    multi_agent_dict[agent_name] = {
                        'api_base': agent_config.api_base,
                        'api_key': agent_config.api_key,
                        'temperature': agent_config.temperature,
                        'top_p': agent_config.top_p,
                        'max_tokens': agent_config.max_tokens,
                        'timeout': agent_config.timeout,
                        'retries': agent_config.retries,
                        'retry_delay': agent_config.retry_delay,
                        'interval': agent_config.interval,
                        'norm': agent_config.norm,
                        'models': [asdict(m) for m in agent_config.models],
                    }
            if multi_agent_dict:
                result['multi_agent'] = multi_agent_dict
        
        return result
    
    def start(self) -> None:
        """Start the process pool"""
        # Convert config to dict for pickling
        # We need to be careful with nested dataclasses
        config_dict = self._serialize_config(self.config)
        
        # Create process pool with initializer
        self.executor = ProcessPoolExecutor(
            max_workers=self.num_workers,
            initializer=_worker_init,
            initargs=(config_dict, self.evaluation_file)
        )
        
        logger.info(f"Started process pool with {self.num_workers} processes")
    
    def stop(self) -> None:
        """Stop the process pool"""
        self.shutdown_event.set()
        
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None
        
        logger.info("Stopped process pool")
    
    def request_shutdown(self) -> None:
        """Request graceful shutdown"""
        logger.info("Graceful shutdown requested...")
        self.shutdown_event.set()
    
    def _create_database_snapshot(self) -> Dict[str, Any]:
        """Create a serializable snapshot of the database state"""
        # Only include necessary data for workers
        snapshot = {
            "programs": {
                pid: prog.to_dict() 
                for pid, prog in self.database.programs.items()
            },
            "islands": [
                list(island) for island in self.database.islands
            ],
            "current_island": self.database.current_island,
            "current_gradient": self.current_gradient, # Pass the consolidated gradient
            "artifacts": {},  # Will be populated selectively
        }
        
        # Include artifacts for programs that might be selected
        # IMPORTANT: This limits artifacts (execution outputs/errors) to first 100 programs only.
        # This does NOT affect program code - all programs are fully serialized above.
        # With max_artifact_bytes=20KB and population_size=1000, artifacts could be 20MB total,
        # which would significantly slow worker process initialization. The limit of 100 keeps
        # artifact data under 2MB while still providing execution context for recent programs.
        # Workers can still evolve properly as they have access to ALL program code.
        for pid in list(self.database.programs.keys())[:100]:
            artifacts = self.database.get_artifacts(pid)
            if artifacts:
                snapshot["artifacts"][pid] = artifacts
        
        return snapshot
    
    async def _update_gradient(self):
        logger.info("Synthesizing new strategic gradient from history...")
        # Gradient metrics display mode (normalized vs raw)
        use_normalized_metrics = False
        if self.config.multi_agent and self.config.multi_agent.gradient:
            use_normalized_metrics = bool(self.config.multi_agent.gradient.norm)
        # Get rollout weights from history config, fallback to defaults
        history_config = self.config.multi_agent.history
        weight_all_improved = (
            history_config.rollout_weight_all_improved
            if history_config
            else getattr(self.config, "rollout_weight_all_improved", 0.6)
        )
        weight_mixed = (
            history_config.rollout_weight_mixed
            if history_config
            else getattr(self.config, "rollout_weight_mixed", 0.2)
        )
        weight_all_degraded = (
            history_config.rollout_weight_all_degraded
            if history_config
            else getattr(self.config, "rollout_weight_all_degraded", 0.2)
        )
        
        # Sample from entire population instead of history buffer
        population_samples = sample_from_population(
            self.database,
            self.gradient_batch_size,
            weight_all_improved=weight_all_improved,
            weight_mixed=weight_mixed,
            weight_all_degraded=weight_all_degraded,
            debug_config=self.config.debug
        )
        
        if not population_samples:
            logger.warning("Not enough population to synthesize a new gradient.")
            return
            
        try:
            new_gradient = await get_gradient(
                population_samples, self.gradient_llm, self.config.debug, use_normalized_metrics=use_normalized_metrics
            )
            self.current_gradient = new_gradient
            
            # Record token usage and time usage for gradient agent (always track, not just for debug)
            gradient_usage = self.gradient_llm.get_last_usage()
            gradient_time = self.gradient_llm.get_last_time()
            self.gradient_token_usage["prompt_tokens"] += gradient_usage.get("prompt_tokens", 0)
            self.gradient_token_usage["completion_tokens"] += gradient_usage.get("completion_tokens", 0)
            self.gradient_token_usage["total_tokens"] += gradient_usage.get("total_tokens", 0)
            if not hasattr(self, 'gradient_time_usage'):
                self.gradient_time_usage = 0.0
            self.gradient_time_usage += gradient_time
            if self.config.debug.pipeline_debug:
                logger.info(f"Token usage (gradient): prompt={gradient_usage['prompt_tokens']}, completion={gradient_usage['completion_tokens']}, total={gradient_usage['total_tokens']}")
                logger.info(f"Time usage (gradient): {gradient_time:.3f}s")
            
            logger.info(f"Successfully updated strategic gradient:\n{new_gradient}")
        except Exception as e:
            logger.error(f"Failed to update strategic gradient: {e}")
    
    def _update_curve_file(self, iteration: int) -> None:
        """
        Update the curve file with the best solution's metrics for the current iteration.
        This method is called after each iteration to dynamically update the evolution history.
        """
        if not self.curve_file_path:
            return
        
        # Get the best program from database
        best_program = None
        if self.database.best_program_id:
            best_program = self.database.get(self.database.best_program_id)
        
        if not best_program:
            best_program = self.database.get_best_program()
        
        if not best_program or not best_program.metrics:
            return
        
        # Format metrics (ensure all values are JSON serializable)
        formatted_metrics = {}
        for key, value in best_program.metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                formatted_metrics[key] = float(value)
            elif isinstance(value, (str, bool, type(None))):
                formatted_metrics[key] = value
            else:
                formatted_metrics[key] = str(value)
        
        # Create or update the curve entry for this iteration
        # Check if we already have an entry for this iteration
        existing_entry_idx = None
        for idx, entry in enumerate(self.curve_data):
            if entry.get("iteration") == iteration:
                existing_entry_idx = idx
                break
        
        curve_entry = {
            "iteration": iteration,
            "metrics": formatted_metrics
        }
        
        if existing_entry_idx is not None:
            # Update existing entry
            self.curve_data[existing_entry_idx] = curve_entry
        else:
            # Add new entry
            self.curve_data.append(curve_entry)
        
        # Sort by iteration
        self.curve_data.sort(key=lambda x: x.get("iteration", 0))
        
        # Save to file
        try:
            import json
            with open(self.curve_file_path, "w") as f:
                json.dump(self.curve_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to update curve file: {e}")

    async def run_evolution(
        self,
        start_iteration: int,
        max_iterations: int,
        target_score: Optional[float] = None,
        checkpoint_callback=None,
    ):
        """Run evolution with process-based parallelism"""
        if not self.executor:
            raise RuntimeError("Process pool not started")
        
        total_iterations = start_iteration + max_iterations
        
        logger.info(
            f"Starting process-based evolution from iteration {start_iteration} "
            f"for {max_iterations} iterations (total: {total_iterations})"
        )
        
        # Track pending futures
        pending_futures: Dict[int, Future] = {}
        batch_size = min(self.num_workers * 2, max_iterations)
        
        # Reset cumulative token and time usage tracking for this evolution run
        self.cumulative_token_usage = {}
        self.cumulative_time_usage = {}
        self.cumulative_model_token_usage = {}
        self.cumulative_model_time_usage = {}
        
        # Submit initial batch
        for i in range(start_iteration, min(start_iteration + batch_size, total_iterations)):
            future = self._submit_iteration(i)
            if future:
                pending_futures[i] = future
        
        next_iteration = start_iteration + batch_size
        completed_iterations = 0
        should_stop = False  # Flag to indicate we should stop processing
        
        # Island management
        programs_per_island = max(1, max_iterations // (self.config.database.num_islands * 10))
        current_island_counter = 0
        
        # Process results as they complete
        while (
            pending_futures 
            and completed_iterations < max_iterations
            and not self.shutdown_event.is_set()
            and not should_stop
        ):
            # Find completed futures
            completed_iteration = None
            for iteration, future in list(pending_futures.items()):
                if future.done():
                    completed_iteration = iteration
                    break
            
            if completed_iteration is None:
                await asyncio.sleep(0.01)
                continue
            
            # Process completed result
            future = pending_futures.pop(completed_iteration)
            
            try:
                result = future.result()
                
                # Update gradient periodically
                if self.config.rl_mode and completed_iteration > start_iteration and completed_iteration % self.gradient_interval == 0:
                    await self._update_gradient()
                
                if result.error:
                    logger.warning(f"Iteration {completed_iteration} error: {result.error}")
                elif result.child_program_dict:
                    # Accumulate token usage and time usage (always track for final summary)
                    if result.token_usage:
                        for agent_name, usage in result.token_usage.items():
                            if agent_name not in self.cumulative_token_usage:
                                self.cumulative_token_usage[agent_name] = {
                                    "prompt_tokens": 0,
                                    "completion_tokens": 0,
                                    "total_tokens": 0,
                                }
                            self.cumulative_token_usage[agent_name]["prompt_tokens"] += usage.get("prompt_tokens", 0)
                            self.cumulative_token_usage[agent_name]["completion_tokens"] += usage.get("completion_tokens", 0)
                            self.cumulative_token_usage[agent_name]["total_tokens"] += usage.get("total_tokens", 0)
                    
                    if result.time_usage:
                        for agent_name, agent_time in result.time_usage.items():
                            self.cumulative_time_usage[agent_name] = self.cumulative_time_usage.get(agent_name, 0.0) + agent_time
                    
                    # Accumulate per-model token usage and time usage
                    if result.model_token_usage:
                        for model_name, usage in result.model_token_usage.items():
                            if model_name not in self.cumulative_model_token_usage:
                                self.cumulative_model_token_usage[model_name] = {
                                    "prompt_tokens": 0,
                                    "completion_tokens": 0,
                                    "total_tokens": 0,
                                }
                            self.cumulative_model_token_usage[model_name]["prompt_tokens"] += usage.get("prompt_tokens", 0)
                            self.cumulative_model_token_usage[model_name]["completion_tokens"] += usage.get("completion_tokens", 0)
                            self.cumulative_model_token_usage[model_name]["total_tokens"] += usage.get("total_tokens", 0)
                    
                    if result.model_time_usage:
                        for model_name, model_time in result.model_time_usage.items():
                            self.cumulative_model_time_usage[model_name] = self.cumulative_model_time_usage.get(model_name, 0.0) + model_time
                    
                    # Reconstruct program from dict
                    child_program = Program(**result.child_program_dict)
                    
                    # Add to database
                    self.database.add(child_program, iteration=completed_iteration)

                    # Log program to history file (abstract and absolute metrics)
                    if self.history_logger:
                        self.history_logger.log_program(child_program, iteration=completed_iteration)
                    
                    # Store artifacts
                    if result.artifacts:
                        self.database.store_artifacts(child_program.id, result.artifacts)
                    
                    # Log prompts
                    if result.prompt:
                        self.database.log_prompt(
                            template_key=(
                                "full_rewrite_user" 
                                if not self.config.diff_based_evolution 
                                else "diff_user"
                            ),
                            program_id=child_program.id,
                            prompt=result.prompt,
                            responses=[result.llm_response] if result.llm_response else []
                        )
                    
                    # Island management
                    if completed_iteration > start_iteration and current_island_counter >= programs_per_island:
                        self.database.next_island()
                        current_island_counter = 0
                        logger.debug(f"Switched to island {self.database.current_island}")
                    
                    current_island_counter += 1
                    self.database.increment_island_generation()
                    
                    # Check migration
                    if self.database.should_migrate():
                        logger.info(f"Performing migration at iteration {completed_iteration}")
                        self.database.migrate_programs()
                        self.database.log_island_status()
                    
                    # Log progress
                    logger.info(
                        f"Iteration {completed_iteration}: "
                        f"Program {child_program.id} "
                        f"(parent: {result.parent_id}) "
                        f"completed in {result.iteration_time:.2f}s"
                    )
                    
                    if child_program.metrics:
                        # Format all metrics
                        metrics_str = ", ".join([
                            f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}"
                            for k, v in child_program.metrics.items()
                        ])
                        
                        if self.config.debug.simple_debug:
                            # In simple_debug mode, still show all metrics but highlight combined_score
                            if "combined_score" in child_program.metrics:
                                logger.info(f"Metrics: {metrics_str} (combined_score={child_program.metrics['combined_score']:.4f})")
                            else:
                                logger.info(f"Metrics: {metrics_str}")
                        else:
                            logger.info(f"Metrics: {metrics_str}")
                        
                        # Check if this is the first program without combined_score
                        if not hasattr(self, '_warned_about_combined_score'):
                            self._warned_about_combined_score = False
                        
                        if "combined_score" not in child_program.metrics and not self._warned_about_combined_score:
                            avg_score = safe_numeric_average(child_program.metrics)
                            logger.warning(
                                f"⚠️  No 'combined_score' metric found in evaluation results. "
                                f"Using average of all numeric metrics ({avg_score:.4f}) for evolution guidance. "
                                f"For better evolution results, please modify your evaluator to return a 'combined_score' "
                                f"metric that properly weights different aspects of program performance."
                            )
                            self._warned_about_combined_score = True
                    
                    # Check for new best
                    if self.database.best_program_id == child_program.id:
                        logger.info(
                            f"🌟 New best solution found at iteration {completed_iteration}: "
                            f"{child_program.id}"
                        )
                    
                    # Update curve file with best solution metrics for this iteration
                    self._update_curve_file(completed_iteration)
                    
                    # Checkpoint callback
                    # Don't checkpoint at iteration 0 (that's just the initial program)
                    if completed_iteration > 0 and completed_iteration % self.config.checkpoint_interval == 0:
                        logger.info(f"Checkpoint interval reached at iteration {completed_iteration}")
                        self.database.log_island_status()
                        if checkpoint_callback:
                            checkpoint_callback(completed_iteration)
                    
                    # Check target score
                    if target_score is not None and child_program.metrics:
                        numeric_metrics = [
                            v for v in child_program.metrics.values()
                            if isinstance(v, (int, float))
                        ]
                        if numeric_metrics:
                            avg_score = sum(numeric_metrics) / len(numeric_metrics)
                            if avg_score >= target_score:
                                logger.info(
                                    f"Target score {target_score} reached at iteration {completed_iteration}"
                                )
                                should_stop = True
                                break
                
            except Exception as e:
                logger.error(f"Error processing result from iteration {completed_iteration}: {e}")
            
            completed_iterations += 1
            
            # Check if we've reached max_iterations
            if completed_iterations >= max_iterations:
                logger.info(f"Reached max_iterations ({max_iterations}), stopping new iteration submissions")
                should_stop = True
            
            # Submit next iteration
            if next_iteration < total_iterations and not self.shutdown_event.is_set() and not should_stop:
                future = self._submit_iteration(next_iteration)
                if future:
                    pending_futures[next_iteration] = future
                    next_iteration += 1
        
        # Handle shutdown or completion
        if self.shutdown_event.is_set():
            logger.info("Shutdown requested, canceling remaining evaluations...")
            for future in pending_futures.values():
                future.cancel()
        elif completed_iterations >= max_iterations:
            logger.info(f"Completed {completed_iterations} iterations (max_iterations={max_iterations})")
            # Cancel any remaining pending futures that haven't started yet
            # Note: We don't cancel futures that are already running to avoid leaving workers in bad state
            logger.info(f"Waiting for {len(pending_futures)} remaining evaluations to complete...")
            # Wait a reasonable time for remaining futures, then cancel if still pending
            import time
            wait_start = time.time()
            max_wait_time = 300  # Wait up to 5 minutes for remaining evaluations
            while pending_futures and (time.time() - wait_start) < max_wait_time:
                await asyncio.sleep(1)
                # Remove completed futures
                for iteration in list(pending_futures.keys()):
                    if pending_futures[iteration].done():
                        try:
                            pending_futures[iteration].result()  # Consume result to avoid warnings
                        except Exception:
                            pass
                        del pending_futures[iteration]
            if pending_futures:
                logger.warning(f"Still have {len(pending_futures)} pending futures after waiting, canceling...")
                for future in pending_futures.values():
                    if not future.done():
                        future.cancel()
        
        logger.info("Evolution completed")
        
        # Include gradient token usage if RL mode is enabled
        if self.config.rl_mode and hasattr(self, 'gradient_token_usage'):
            self.cumulative_token_usage["gradient"] = self.gradient_token_usage.copy()
        
        # Include gradient time usage if RL mode is enabled
        if self.config.rl_mode and hasattr(self, 'gradient_time_usage'):
            self.cumulative_time_usage["gradient"] = self.gradient_time_usage
        
        # Print cumulative token usage summary if pipeline_debug is enabled
        if self.config.debug.pipeline_debug and self.cumulative_token_usage:
            logger.info("=" * 70)
            logger.info("CUMULATIVE TOKEN USAGE SUMMARY (All Iterations)")
            logger.info("=" * 70)
            
            total_prompt = sum(u.get("prompt_tokens", 0) for u in self.cumulative_token_usage.values())
            total_completion = sum(u.get("completion_tokens", 0) for u in self.cumulative_token_usage.values())
            total_tokens = sum(u.get("total_tokens", 0) for u in self.cumulative_token_usage.values())
            
            for agent_name, usage in sorted(self.cumulative_token_usage.items()):
                logger.info(f"  {agent_name:15s}: prompt={usage['prompt_tokens']:8d}, completion={usage['completion_tokens']:8d}, total={usage['total_tokens']:8d}")
            
            logger.info("-" * 70)
            logger.info(f"  {'TOTAL':15s}: prompt={total_prompt:8d}, completion={total_completion:8d}, total={total_tokens:8d}")
            logger.info("=" * 70)
        
        # Print cumulative time usage summary if pipeline_debug is enabled
        if self.config.debug.pipeline_debug and self.cumulative_time_usage:
            logger.info("=" * 70)
            logger.info("CUMULATIVE TIME USAGE SUMMARY (All Iterations)")
            logger.info("=" * 70)
            
            total_time = 0.0
            for agent_name, agent_time in sorted(self.cumulative_time_usage.items()):
                logger.info(f"  {agent_name:15s}: {agent_time:8.3f}s ({agent_time/60:6.2f}min)")
                total_time += agent_time
            
            logger.info("-" * 70)
            logger.info(f"  {'TOTAL':15s}: {total_time:8.3f}s ({total_time/60:6.2f}min)")
            logger.info("=" * 70)
        
        return self.database.get_best_program()
    
    def _submit_iteration(self, iteration: int) -> Optional[Future]:
        """Submit an iteration to the process pool"""
        try:
            # Sample parent and inspirations
            # Prefer num_inspiration_programs, then num_inspirations, then database config
            num_inspirations = (
                getattr(self.config.prompt, 'num_inspiration_programs', None) or 
                getattr(self.config.prompt, 'num_inspirations', None) or 
                getattr(self.config.database, 'num_inspirations', 5)
            )
            parent, inspirations = self.database.sample(n=num_inspirations)
            
            # Create database snapshot
            db_snapshot = self._create_database_snapshot()
            
            # Submit to process pool
            future = self.executor.submit(
                _run_iteration_worker,
                iteration,
                db_snapshot,
                parent.id,
                [insp.id for insp in inspirations]
            )
            
            return future
            
        except Exception as e:
            logger.error(f"Error submitting iteration {iteration}: {e}")
            return None