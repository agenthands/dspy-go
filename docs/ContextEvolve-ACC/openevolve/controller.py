"""
Main controller for OpenEvolve
"""

import asyncio
import json
import logging
import os
import signal
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from openevolve.config import Config, load_config
from openevolve.database import Program, ProgramDatabase
from openevolve.evaluator import Evaluator
from openevolve.llm.ensemble import LLMEnsemble
from openevolve.prompt.sampler import PromptSampler
from openevolve.process_parallel import ProcessParallelController
from openevolve.utils.code_utils import (
    extract_code_language,
)
from openevolve.utils.format_utils import (
    format_metrics_safe,
    format_improvement_safe,
)
from openevolve.rl_agents import get_summary, ProgramHistoryLogger

logger = logging.getLogger(__name__)


def _format_metrics(metrics: Dict[str, Any]) -> str:
    """Safely format metrics, handling both numeric and string values"""
    formatted_parts = []
    for name, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                formatted_parts.append(f"{name}={value:.4f}")
            except (ValueError, TypeError):
                formatted_parts.append(f"{name}={value}")
        else:
            formatted_parts.append(f"{name}={value}")
    return ", ".join(formatted_parts)


def _format_improvement(improvement: Dict[str, Any]) -> str:
    """Safely format improvement metrics"""
    formatted_parts = []
    for name, diff in improvement.items():
        if isinstance(diff, (int, float)) and not isinstance(diff, bool):
            try:
                formatted_parts.append(f"{name}={diff:+.4f}")
            except (ValueError, TypeError):
                formatted_parts.append(f"{name}={diff}")
        else:
            formatted_parts.append(f"{name}={diff}")
    return ", ".join(formatted_parts)


class OpenEvolve:
    """
    Main controller for OpenEvolve

    Orchestrates the evolution process, coordinating between the prompt sampler,
    LLM ensemble, evaluator, and program database.

    Features:
    - Tracks the absolute best program across evolution steps
    - Ensures the best solution is not lost during the MAP-Elites process
    - Always includes the best program in the selection process for inspiration
    - Maintains detailed logs and metadata about improvements
    """

    def __init__(
        self,
        initial_program_path: str,
        evaluation_file: str,
        config_path: Optional[str] = None,
        config: Optional[Config] = None,
        output_dir: Optional[str] = None,
        history_file_path: Optional[str] = None,
    ):
        # Load configuration
        if config is not None:
            # Use provided Config object directly
            self.config = config
        else:
            # Load from file or use defaults
            self.config = load_config(config_path)

        # Set up output directory
        self.output_dir = output_dir or os.path.join(
            os.path.dirname(initial_program_path), "openevolve_output"
        )
        os.makedirs(self.output_dir, exist_ok=True)

        # Generate a consistent timestamp for all output files
        self.run_timestamp = time.strftime('%Y%m%d_%H%M%S')

        # Set up logging
        self._setup_logging()

        # Set random seed for reproducibility if specified
        if self.config.random_seed is not None:
            import random
            import numpy as np
            import hashlib

            # Set global random seeds
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)

            # Create hash-based seeds for different components
            base_seed = str(self.config.random_seed).encode("utf-8")
            llm_seed = int(hashlib.md5(base_seed + b"llm").hexdigest()[:8], 16) % (2**31)

            # Propagate seed to LLM configurations
            self.config.llm.random_seed = llm_seed
            for model_cfg in self.config.llm.models:
                if not hasattr(model_cfg, "random_seed") or model_cfg.random_seed is None:
                    model_cfg.random_seed = llm_seed
            for model_cfg in self.config.llm.evaluator_models:
                if not hasattr(model_cfg, "random_seed") or model_cfg.random_seed is None:
                    model_cfg.random_seed = llm_seed

            logger.info(f"Set random seed to {self.config.random_seed} for reproducibility")
            # logger.debug(f"Generated LLM seed: {llm_seed}")

        # Load initial program
        self.initial_program_path = initial_program_path
        self.initial_program_code = self._load_initial_program()
        if not self.config.language:
            self.config.language = extract_code_language(self.initial_program_code)

        # Extract file extension from initial program
        self.file_extension = os.path.splitext(initial_program_path)[1]
        if not self.file_extension:
            # Default to .py if no extension found
            self.file_extension = ".py"
        else:
            # Make sure it starts with a dot
            if not self.file_extension.startswith("."):
                self.file_extension = f".{self.file_extension}"

        # Initialize components
        # Use multi-agent config if available, otherwise use default llm config
        evolve_config = self.config.multi_agent.get_agent_config("evolve", self.config.llm)
        self.llm_ensemble = LLMEnsemble(evolve_config.models)
        
        self.llm_evaluator_ensemble = LLMEnsemble(self.config.llm.evaluator_models)
        
        # Initialize agent-specific LLM ensembles for RL mode
        if self.config.rl_mode:
            summary_config = self.config.multi_agent.get_agent_config("summary", self.config.llm)
            gradient_config = self.config.multi_agent.get_agent_config("gradient", self.config.llm)
            sample_config = self.config.multi_agent.get_agent_config("sample", self.config.llm)
            
            self.summary_llm = LLMEnsemble(summary_config.models)
            self.gradient_llm = LLMEnsemble(gradient_config.models)
            self.sample_llm = LLMEnsemble(sample_config.models)
        else:
            self.summary_llm = None
            self.gradient_llm = None
            self.sample_llm = None

        self.prompt_sampler = PromptSampler(self.config.prompt)
        self.evaluator_prompt_sampler = PromptSampler(self.config.prompt)
        self.evaluator_prompt_sampler.set_templates("evaluator_system_message")

        # Pass random seed to database if specified
        if self.config.random_seed is not None:
            self.config.database.random_seed = self.config.random_seed

        self.database = ProgramDatabase(self.config.database)

        self.evaluator = Evaluator(
            self.config.evaluator,
            evaluation_file,
            self.llm_evaluator_ensemble,
            self.evaluator_prompt_sampler,
            database=self.database,
        )
        self.evaluation_file = evaluation_file

        logger.info(f"Initialized OpenEvolve with {initial_program_path}")

        # Initialize program history logger
        if history_file_path:
            # Use provided history file path (for resuming)
            history_path = history_file_path
            logger.info(f"Using specified history file: {history_path}")
        else:
            # Generate new timestamped history file
            history_dir = os.path.join(self.output_dir, "history")
            os.makedirs(history_dir, exist_ok=True)
            history_filename = f"evolution_history_{self.run_timestamp}.jsonl"
            history_path = os.path.join(history_dir, history_filename)
        self.history_logger = ProgramHistoryLogger(history_path)
        
        # Initialize usage and curve directories
        self.usage_dir = os.path.join(self.output_dir, "usage")
        os.makedirs(self.usage_dir, exist_ok=True)
        self.curve_dir = os.path.join(self.output_dir, "curve")
        os.makedirs(self.curve_dir, exist_ok=True)
        
        # Initialize curve file path for dynamic updates
        self.curve_file_path = os.path.join(self.curve_dir, f"evolution_history_{self.run_timestamp}.json")
        
        # Initialize curve data list to track best solution per iteration
        self.curve_data = []

        # Initialize improved parallel processing components
        self.parallel_controller = None

    def _setup_logging(self) -> None:
        """Set up logging"""
        log_dir = self.config.log_dir or os.path.join(self.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        # Set up root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.log_level))

        # Silence verbose SDK/http debug logs (e.g., OpenAI "Request options")
        for noisy_logger in ("openai", "httpx", "httpcore", "stainless", "urllib3"):
            logging.getLogger(noisy_logger).setLevel(logging.WARNING)

        # Add file handler - use shared timestamp for consistent naming
        log_file = os.path.join(log_dir, f"openevolve_{self.run_timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        root_logger.addHandler(file_handler)

        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        root_logger.addHandler(console_handler)

        logger.info(f"Logging to {log_file}")

    def _load_initial_program(self) -> str:
        """Load the initial program from file"""
        with open(self.initial_program_path, "r") as f:
            return f.read()

    async def run(
        self,
        iterations: Optional[int] = None,
        target_score: Optional[float] = None,
        checkpoint_path: Optional[str] = None,
    ) -> Optional[Program]:
        """
        Run the evolution process with improved parallel processing

        Args:
            iterations: Maximum number of iterations (uses config if None)
            target_score: Target score to reach (continues until reached if specified)
            checkpoint_path: Path to resume from checkpoint

        Returns:
            Best program found
        """
        max_iterations = iterations or self.config.max_iterations

        # Determine starting iteration
        start_iteration = 0
        is_resuming = False
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)
            start_iteration = self.database.last_iteration + 1
            is_resuming = True
            logger.info(f"Resuming from checkpoint at iteration {start_iteration}")
        else:
            start_iteration = self.database.last_iteration

        # History buffer removed - population serves as replay buffer

        # Only add initial program if starting fresh (not resuming from checkpoint)
        should_add_initial = (
            start_iteration == 0
            and len(self.database.programs) == 0
            and not any(
                p.code == self.initial_program_code for p in self.database.programs.values()
            )
        )

        if should_add_initial:
            logger.info("Adding initial program to database")
            initial_program_id = str(uuid.uuid4())

            # Evaluate the initial program
            initial_metrics = await self.evaluator.evaluate_program(
                self.initial_program_code, initial_program_id
            )

            initial_program = Program(
                id=initial_program_id,
                code=self.initial_program_code,
                language=self.config.language,
                metrics=initial_metrics,
                iteration_found=start_iteration,
            )

            self.database.add(initial_program)
            
            # Log initial program to history
            if hasattr(self, 'history_logger') and self.history_logger:
                self.history_logger.log_program(initial_program, iteration=start_iteration)
            
            # Save initial program metrics to curve file (iteration 0)
            self._save_initial_curve_data(initial_program, start_iteration)
            
            # Generate abstract for initial program if RL mode is enabled and abstract is missing
            if self.config.rl_mode and not initial_program.abstract:
                logger.info("Generating abstract for initial program...")
                try:
                    summary = await get_summary(
                        initial_program,
                        self.summary_llm,
                        self.config.debug,
                        parent_program=None,
                        parent_abstract=None
                    )
                    initial_program.abstract = summary
                    # Update the program in database with the abstract
                    self.database.programs[initial_program.id].abstract = summary
                    logger.info(f"Successfully generated abstract for initial program")
                    # Update history entry if abstract was generated
                    if hasattr(self, 'history_logger') and self.history_logger:
                        self.history_logger.log_program(initial_program, iteration=start_iteration)
                except Exception as e:
                    logger.warning(f"Failed to generate abstract for initial program: {e}")
            
            # Check if combined_score is present in the metrics
            if "combined_score" not in initial_metrics:
                # Calculate average of numeric metrics
                numeric_metrics = [
                    v for v in initial_metrics.values() 
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                ]
                if numeric_metrics:
                    avg_score = sum(numeric_metrics) / len(numeric_metrics)
                    logger.warning(
                        f"⚠️  No 'combined_score' metric found in evaluation results. "
                        f"Using average of all numeric metrics ({avg_score:.4f}) for evolution guidance. "
                        f"For better evolution results, please modify your evaluator to return a 'combined_score' "
                        f"metric that properly weights different aspects of program performance."
                    )
        else:
            logger.info(
                f"Skipping initial program addition (resuming from iteration {start_iteration} "
                f"with {len(self.database.programs)} existing programs)"
            )

        # Initialize improved parallel processing
        try:
            self.parallel_controller = ProcessParallelController(
                self.config, self.evaluation_file, self.database, self.history_logger,
                curve_file_path=self.curve_file_path,
                initial_curve_data=self.curve_data
            )

            # Set up signal handlers for graceful shutdown
            def signal_handler(signum, frame):
                logger.info(f"Received signal {signum}, initiating graceful shutdown...")
                self.parallel_controller.request_shutdown()

                # Set up a secondary handler for immediate exit if user presses Ctrl+C again
                def force_exit_handler(signum, frame):
                    logger.info("Force exit requested - terminating immediately")
                    import sys

                    sys.exit(0)

                signal.signal(signal.SIGINT, force_exit_handler)

            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

            self.parallel_controller.start()

            # When starting from iteration 0, we've already done the initial program evaluation
            # So we need to adjust the start_iteration for the actual evolution
            evolution_start = start_iteration
            evolution_iterations = max_iterations
            
            # If we just added the initial program at iteration 0, start evolution from iteration 1
            if should_add_initial and start_iteration == 0:
                evolution_start = 1
                # User expects max_iterations evolutionary iterations AFTER the initial program
                # So we don't need to reduce evolution_iterations
                
            # Run evolution with improved parallel processing and checkpoint callback
            await self._run_evolution_with_checkpoints(
                evolution_start, evolution_iterations, target_score
            )
            
            # Save usage statistics to JSON file BEFORE cleaning up parallel_controller
            # (since cumulative_token_usage and cumulative_time_usage are stored there)
            self._save_usage_stats()

        finally:
            # Clean up parallel processing resources
            if self.parallel_controller:
                self.parallel_controller.stop()
                self.parallel_controller = None

        # Log token usage statistics for all LLM calls
        self._log_token_stats()

        # Get the best program
        best_program = None
        if self.database.best_program_id:
            best_program = self.database.get(self.database.best_program_id)
            logger.info(f"Using tracked best program: {self.database.best_program_id}")

        if best_program is None:
            best_program = self.database.get_best_program()
            logger.info("Using calculated best program (tracked program not found)")

        # Check if there's a better program by combined_score that wasn't tracked
        if best_program and "combined_score" in best_program.metrics:
            best_by_combined = self.database.get_best_program(metric="combined_score")
            if (
                best_by_combined
                and best_by_combined.id != best_program.id
                and "combined_score" in best_by_combined.metrics
            ):
                # If the combined_score of this program is significantly better, use it instead
                if (
                    best_by_combined.metrics["combined_score"]
                    > best_program.metrics["combined_score"] + 0.02
                ):
                    logger.warning(
                        f"Found program with better combined_score: {best_by_combined.id}"
                    )
                    logger.warning(
                        f"Score difference: {best_program.metrics['combined_score']:.4f} vs "
                        f"{best_by_combined.metrics['combined_score']:.4f}"
                    )
                    best_program = best_by_combined

        if best_program:
            logger.info(
                f"Evolution complete. Best program has metrics: "
                f"{format_metrics_safe(best_program.metrics)}"
            )
            self._save_best_program(best_program)
            return best_program
        else:
            logger.warning("No valid programs found during evolution")
            return None

    def _log_iteration(
        self,
        iteration: int,
        parent: Program,
        child: Program,
        elapsed_time: float,
    ) -> None:
        """
        Log iteration progress

        Args:
            iteration: Iteration number
            parent: Parent program
            child: Child program
            elapsed_time: Elapsed time in seconds
        """
        # Calculate improvement using safe formatting
        improvement_str = format_improvement_safe(parent.metrics, child.metrics)

        logger.info(
            f"Iteration {iteration+1}: Child {child.id} from parent {parent.id} "
            f"in {elapsed_time:.2f}s. Metrics: "
            f"{format_metrics_safe(child.metrics)} "
            f"(Δ: {improvement_str})"
        )

    def _save_checkpoint(self, iteration: int) -> None:
        """
        Save a checkpoint

        Args:
            iteration: Current iteration number
        """
        checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Create specific checkpoint directory
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{iteration}")
        os.makedirs(checkpoint_path, exist_ok=True)

        # Save the database
        self.database.save(checkpoint_path, iteration)

        # Save the best program found so far
        best_program = None
        if self.database.best_program_id:
            best_program = self.database.get(self.database.best_program_id)
        else:
            best_program = self.database.get_best_program()

        if best_program:
            # Save the best program at this checkpoint
            best_program_path = os.path.join(checkpoint_path, f"best_program{self.file_extension}")
            with open(best_program_path, "w") as f:
                f.write(best_program.code)

            # Save metrics
            best_program_info_path = os.path.join(checkpoint_path, "best_program_info.json")
            with open(best_program_info_path, "w") as f:
                json.dump(
                    {
                        "id": best_program.id,
                        "generation": best_program.generation,
                        "iteration": best_program.iteration_found,
                        "current_iteration": iteration,
                        "metrics": best_program.metrics,
                        "language": best_program.language,
                        "timestamp": best_program.timestamp,
                        "saved_at": time.time(),
                    },
                    f,
                    indent=2,
                )

            logger.info(
                f"Saved best program at checkpoint {iteration} with metrics: "
                f"{format_metrics_safe(best_program.metrics)}"
            )

        logger.info(f"Saved checkpoint at iteration {iteration} to {checkpoint_path}")

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load state from a checkpoint directory"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint directory {checkpoint_path} not found")

        logger.info(f"Loading checkpoint from {checkpoint_path}")
        self.database.load(checkpoint_path)
        logger.info(f"Checkpoint loaded successfully (iteration {self.database.last_iteration})")

    async def _run_evolution_with_checkpoints(
        self, start_iteration: int, max_iterations: int, target_score: Optional[float]
    ) -> None:
        """Run evolution with checkpoint saving support"""
        logger.info(f"Using island-based evolution with {self.config.database.num_islands} islands")
        self.database.log_island_status()

        # Run the evolution process with checkpoint callback
        await self.parallel_controller.run_evolution(
            start_iteration, max_iterations, target_score, checkpoint_callback=self._save_checkpoint
        )

        # Check if shutdown was requested
        if self.parallel_controller.shutdown_event.is_set():
            logger.info("Evolution stopped due to shutdown request")
            return

        # Save final checkpoint if needed
        # Note: start_iteration here is the evolution start (1 for fresh start, not 0)
        # max_iterations is the number of evolution iterations to run
        final_iteration = start_iteration + max_iterations - 1
        if final_iteration > 0 and final_iteration % self.config.checkpoint_interval == 0:
            self._save_checkpoint(final_iteration)

    def _save_best_program(self, program: Optional[Program] = None) -> None:
        """
        Save the best program

        Args:
            program: Best program (if None, uses the tracked best program)
        """
        # If no program is provided, use the tracked best program from the database
        if program is None:
            if self.database.best_program_id:
                program = self.database.get(self.database.best_program_id)
            else:
                # Fallback to calculating best program if no tracked best program
                program = self.database.get_best_program()

        if not program:
            logger.warning("No best program found to save")
            return

        best_dir = os.path.join(self.output_dir, "best")
        os.makedirs(best_dir, exist_ok=True)

        # Use the extension from the initial program file
        filename = f"best_program{self.file_extension}"
        code_path = os.path.join(best_dir, filename)

        with open(code_path, "w") as f:
            f.write(program.code)

        # Save complete program info including metrics
        info_path = os.path.join(best_dir, "best_program_info.json")
        with open(info_path, "w") as f:
            json.dump(
                {
                    "id": program.id,
                    "generation": program.generation,
                    "iteration": program.iteration_found,
                    "timestamp": program.timestamp,
                    "parent_id": program.parent_id,
                    "metrics": program.metrics,
                    "language": program.language,
                    "saved_at": time.time(),
                },
                f,
                indent=2,
            )

        logger.info(f"Saved best program to {code_path} with program info to {info_path}")
    
    def _save_initial_curve_data(self, program: Program, iteration: int) -> None:
        """
        Save the initial program's metrics to the curve file.
        This is called after the initial program is evaluated (iteration 0).
        
        Args:
            program: The initial program with metrics
            iteration: The iteration number (typically 0)
        """
        if not program.metrics:
            return
        
        # Format metrics (ensure all values are JSON serializable)
        formatted_metrics = {}
        for key, value in program.metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                formatted_metrics[key] = float(value)
            elif isinstance(value, (str, bool, type(None))):
                formatted_metrics[key] = value
            else:
                formatted_metrics[key] = str(value)
        
        # Create curve entry for this iteration
        curve_entry = {
            "iteration": iteration,
            "metrics": formatted_metrics
        }
        
        # Add to curve data
        self.curve_data.append(curve_entry)
        
        # Save to file
        try:
            with open(self.curve_file_path, "w") as f:
                json.dump(self.curve_data, f, indent=2)
            logger.info(f"Saved initial curve data to {self.curve_file_path}")
        except Exception as e:
            logger.warning(f"Failed to save initial curve data: {e}")
    
    def _log_token_stats(self) -> None:
        """Log token usage statistics for all LLM calls"""
        # Get token stats from main LLM ensemble
        main_stats = self.llm_ensemble.get_token_stats()
        
        # Get token stats from evaluator LLM ensemble if available
        evaluator_stats = {}
        if self.llm_evaluator_ensemble:
            evaluator_stats = self.llm_evaluator_ensemble.get_token_stats()

        
        # Calculate and log totals
        total_input = sum(s['prompt_tokens'] for s in main_stats.values()) + sum(s['prompt_tokens'] for s in evaluator_stats.values())
        total_output = sum(s['completion_tokens'] for s in main_stats.values()) + sum(s['completion_tokens'] for s in evaluator_stats.values())
        total_all = sum(s['total_tokens'] for s in main_stats.values()) + sum(s['total_tokens'] for s in evaluator_stats.values())
        
        logger.info("=" * 80)
        logger.info("Total Token Usage Across All Models:")
        logger.info(f"Total Input Tokens: {total_input:,}")
        logger.info(f"Total Output Tokens: {total_output:,}")
        logger.info(f"Total Tokens: {total_all:,}")
        logger.info("=" * 80)
    
    def _save_usage_stats(self) -> None:
        """Save token and time usage statistics to a JSON file"""
        # Initialize usage data structure
        usage_data = {
            "token": {
                "multi-agent": {
                    "evolve": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    "gradient": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    "summary": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    "sample": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                },
                "models": {}
            },
            "time": {
                "multi-agent": {
                    "evolve": 0.0,
                    "gradient": 0.0,
                    "summary": 0.0,
                    "sample": 0.0,
                },
                "models": {}
            }
        }
        
        # Collect multi-agent token and time usage from parallel_controller
        if hasattr(self, 'parallel_controller') and self.parallel_controller is not None:
            # Get cumulative token usage per agent
            if hasattr(self.parallel_controller, 'cumulative_token_usage') and self.parallel_controller.cumulative_token_usage:
                for agent_name, agent_usage in self.parallel_controller.cumulative_token_usage.items():
                    if agent_name in usage_data["token"]["multi-agent"]:
                        usage_data["token"]["multi-agent"][agent_name] = {
                            "prompt_tokens": agent_usage.get("prompt_tokens", 0),
                            "completion_tokens": agent_usage.get("completion_tokens", 0),
                            "total_tokens": agent_usage.get("total_tokens", 0),
                        }
            
            # Get gradient token usage separately (it's tracked in main process)
            if hasattr(self.parallel_controller, 'gradient_token_usage') and self.parallel_controller.gradient_token_usage:
                usage_data["token"]["multi-agent"]["gradient"] = {
                    "prompt_tokens": self.parallel_controller.gradient_token_usage.get("prompt_tokens", 0),
                    "completion_tokens": self.parallel_controller.gradient_token_usage.get("completion_tokens", 0),
                    "total_tokens": self.parallel_controller.gradient_token_usage.get("total_tokens", 0),
                }
            
            # Get cumulative time usage per agent
            if hasattr(self.parallel_controller, 'cumulative_time_usage') and self.parallel_controller.cumulative_time_usage:
                for agent_name, agent_time in self.parallel_controller.cumulative_time_usage.items():
                    if agent_name in usage_data["time"]["multi-agent"]:
                        usage_data["time"]["multi-agent"][agent_name] = agent_time
            
            # Get gradient time usage separately
            if hasattr(self.parallel_controller, 'gradient_time_usage'):
                usage_data["time"]["multi-agent"]["gradient"] = self.parallel_controller.gradient_time_usage
            
            # Collect per-model token and time usage from worker processes
            if hasattr(self.parallel_controller, 'cumulative_model_token_usage') and self.parallel_controller.cumulative_model_token_usage:
                for model_name, usage in self.parallel_controller.cumulative_model_token_usage.items():
                    if model_name not in usage_data["token"]["models"]:
                        usage_data["token"]["models"][model_name] = {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                        }
                    usage_data["token"]["models"][model_name]["prompt_tokens"] += usage.get("prompt_tokens", 0)
                    usage_data["token"]["models"][model_name]["completion_tokens"] += usage.get("completion_tokens", 0)
                    usage_data["token"]["models"][model_name]["total_tokens"] += usage.get("total_tokens", 0)
            
            if hasattr(self.parallel_controller, 'cumulative_model_time_usage') and self.parallel_controller.cumulative_model_time_usage:
                for model_name, model_time in self.parallel_controller.cumulative_model_time_usage.items():
                    if model_name not in usage_data["time"]["models"]:
                        usage_data["time"]["models"][model_name] = 0.0
                    usage_data["time"]["models"][model_name] += model_time
        
        # Also collect per-model stats from main process LLM ensembles
        # (gradient LLM runs in main process, and initial program summary)
        main_process_ensembles = []
        
        # Gradient LLM (runs in main process)
        if hasattr(self, 'gradient_llm') and self.gradient_llm:
            main_process_ensembles.append(self.gradient_llm)
        
        # Summary LLM in main process (for initial program)
        if hasattr(self, 'summary_llm') and self.summary_llm:
            main_process_ensembles.append(self.summary_llm)
        
        # Aggregate token stats and time stats per model from main process ensembles
        for ensemble in main_process_ensembles:
            token_stats = ensemble.get_token_stats()
            time_stats = ensemble.get_time_stats()
            
            for model_name, stats in token_stats.items():
                if model_name not in usage_data["token"]["models"]:
                    usage_data["token"]["models"][model_name] = {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    }
                usage_data["token"]["models"][model_name]["prompt_tokens"] += stats.get("prompt_tokens", 0)
                usage_data["token"]["models"][model_name]["completion_tokens"] += stats.get("completion_tokens", 0)
                usage_data["token"]["models"][model_name]["total_tokens"] += stats.get("total_tokens", 0)
            
            for model_name, model_time in time_stats.items():
                if model_name not in usage_data["time"]["models"]:
                    usage_data["time"]["models"][model_name] = 0.0
                usage_data["time"]["models"][model_name] += model_time
        
        # Save to JSON file
        usage_file_path = os.path.join(self.usage_dir, f"usage_{self.run_timestamp}.json")
        try:
            with open(usage_file_path, "w") as f:
                json.dump(usage_data, f, indent=2)
            logger.info(f"Saved usage statistics to {usage_file_path}")
        except Exception as e:
            logger.warning(f"Failed to save usage statistics: {e}")
