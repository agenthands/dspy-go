"""
Configuration handling for OpenEvolve
"""

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class LLMModelConfig:
    """Configuration for a single LLM model"""

    # API configuration
    api_base: str = None
    api_key: Optional[str] = None
    name: str = None

    # Weight for model in ensemble
    weight: float = 1.0

    # Generation parameters
    system_message: Optional[str] = None
    temperature: float = None
    top_p: float = None
    max_tokens: int = None

    # Request parameters
    timeout: int = None
    retries: int = None
    retry_delay: int = None

    # Reproducibility
    random_seed: Optional[int] = None


@dataclass
class AgentLLMConfig:
    """Configuration for a single agent's LLM"""
    
    # API configuration
    api_base: Optional[str] = None  # If None, will inherit from default llm config
    api_key: Optional[str] = None  # If None, will inherit from default llm config
    
    # Model configuration
    models: List[LLMModelConfig] = field(default_factory=lambda: [])
    
    # Generation parameters (if None, inherit from default)
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout: Optional[int] = None
    retries: Optional[int] = None
    retry_delay: Optional[int] = None
    
    # Agent-specific parameters
    interval: Optional[int] = None  # For gradient agent: how often to update (in iterations)
    norm: Optional[bool] = None  # For gradient agent: whether to use normalized metrics (z-score) in gradient synthesis prompt
    
    def __post_init__(self):
        """Post-initialization to set up model configurations"""
        # If no models specified, create empty list (will use default)
        if not self.models:
            self.models = []


@dataclass
class HistoryConfig:
    """Configuration for population-based rollout (replay buffer)"""
    
    rollout_batch_size: int = 20  # How many population samples to use for gradient synthesis
    rollout_weight_all_improved: float = 0.6  # Weight for programs where all normalized metrics > 0
    rollout_weight_mixed: float = 0.2  # Weight for programs with mixed normalized metrics
    rollout_weight_all_degraded: float = 0.2  # Weight for programs where all normalized metrics < 0


@dataclass
class MultiAgentConfig:
    """Configuration for multi-agent LLM setup"""
    
    # Individual agent configurations
    evolve: Optional[AgentLLMConfig] = None  # For code evolution
    summary: Optional[AgentLLMConfig] = None  # For program summarization
    gradient: Optional[AgentLLMConfig] = None  # For program critique/gradient
    sample: Optional[AgentLLMConfig] = None  # For reference selection (get_references)
    history: Optional[HistoryConfig] = None  # For population-based rollout (replay buffer) configuration
    
    def get_agent_config(self, agent_name: str, default_llm_config: "LLMConfig") -> "LLMConfig":
        """
        Get LLM config for a specific agent, merging with default if needed.
        
        Args:
            agent_name: Name of the agent ('evolve', 'summary', 'gradient', 'sample')
            default_llm_config: Default LLM config to inherit from
            
        Returns:
            LLMConfig instance configured for the agent
        """
        agent_config = getattr(self, agent_name, None)
        
        if agent_config is None:
            # Use default config
            return default_llm_config
        
        # Create a new LLMConfig based on default, then override with agent-specific settings
        merged_config = LLMConfig()
        
        # Copy default values
        merged_config.api_base = default_llm_config.api_base
        merged_config.api_key = default_llm_config.api_key
        merged_config.temperature = default_llm_config.temperature
        merged_config.top_p = default_llm_config.top_p
        merged_config.max_tokens = default_llm_config.max_tokens
        merged_config.timeout = default_llm_config.timeout
        merged_config.retries = default_llm_config.retries
        merged_config.retry_delay = default_llm_config.retry_delay
        merged_config.random_seed = default_llm_config.random_seed
        
        # Override with agent-specific values if provided
        if agent_config.api_base is not None:
            merged_config.api_base = agent_config.api_base
        if agent_config.api_key is not None:
            merged_config.api_key = agent_config.api_key
        if agent_config.temperature is not None:
            merged_config.temperature = agent_config.temperature
        if agent_config.top_p is not None:
            merged_config.top_p = agent_config.top_p
        if agent_config.max_tokens is not None:
            merged_config.max_tokens = agent_config.max_tokens
        if agent_config.timeout is not None:
            merged_config.timeout = agent_config.timeout
        if agent_config.retries is not None:
            merged_config.retries = agent_config.retries
        if agent_config.retry_delay is not None:
            merged_config.retry_delay = agent_config.retry_delay
        
        # Use agent-specific models if provided, otherwise use default
        if agent_config.models:
            merged_config.models = agent_config.models.copy()
            # Apply shared config to models
            shared_config = {
                "api_base": merged_config.api_base,
                "api_key": merged_config.api_key,
                "temperature": merged_config.temperature,
                "top_p": merged_config.top_p,
                "max_tokens": merged_config.max_tokens,
                "timeout": merged_config.timeout,
                "retries": merged_config.retries,
                "retry_delay": merged_config.retry_delay,
                "random_seed": merged_config.random_seed,
            }
            merged_config.update_model_params(shared_config)
        else:
            # Use default models
            merged_config.models = default_llm_config.models.copy()
        
        # Call __post_init__ to ensure models are properly initialized
        merged_config.__post_init__()
        
        return merged_config


@dataclass
class LLMConfig(LLMModelConfig):
    """Configuration for LLM models"""

    # API configuration
    api_base: str = "https://api.openai.com/v1"

    # Generation parameters
    system_message: Optional[str] = "system_message"
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 4096

    # Request parameters
    timeout: int = 60
    retries: int = 3
    retry_delay: int = 5

    # n-model configuration for evolution LLM ensemble
    models: List[LLMModelConfig] = field(
        default_factory=lambda: [
            LLMModelConfig(name="gpt-4o-mini", weight=0.8),
            LLMModelConfig(name="gpt-4o", weight=0.2),
        ]
    )

    # n-model configuration for evaluator LLM ensemble
    evaluator_models: List[LLMModelConfig] = field(default_factory=lambda: [])

    # Backwardes compatibility with primary_model(_weight) options
    primary_model: str = None
    primary_model_weight: float = None
    secondary_model: str = None
    secondary_model_weight: float = None

    def __post_init__(self):
        """Post-initialization to set up model configurations"""
        # Handle backward compatibility for primary_model(_weight) and secondary_model(_weight).
        if (self.primary_model or self.primary_model_weight) and len(self.models) < 1:
            # Ensure we have a primary model
            self.models.append(LLMModelConfig())
        if self.primary_model:
            self.models[0].name = self.primary_model
        if self.primary_model_weight:
            self.models[0].weight = self.primary_model_weight

        if (self.secondary_model or self.secondary_model_weight) and len(self.models) < 2:
            # Ensure we have a second model
            self.models.append(LLMModelConfig())
        if self.secondary_model:
            self.models[1].name = self.secondary_model
        if self.secondary_model_weight:
            self.models[1].weight = self.secondary_model_weight

        # If no evaluator models are defined, use the same models as for evolution
        if not self.evaluator_models or len(self.evaluator_models) < 1:
            self.evaluator_models = self.models.copy()

        # Update models with shared configuration values
        shared_config = {
            "api_base": self.api_base,
            "api_key": self.api_key,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "retries": self.retries,
            "retry_delay": self.retry_delay,
            "random_seed": self.random_seed,
        }
        self.update_model_params(shared_config)

    def update_model_params(self, args: Dict[str, Any], overwrite: bool = False) -> None:
        """Update model parameters for all models"""
        for model in self.models + self.evaluator_models:
            for key, value in args.items():
                if overwrite or getattr(model, key, None) is None:
                    setattr(model, key, value)


@dataclass
class PromptConfig:
    """Configuration for prompt generation"""

    template_dir: Optional[str] = None
    system_message: str = "system_message"
    evaluator_system_message: str = "evaluator_system_message"

    # Number of examples to include in the prompt
    num_top_programs: int = 3
    num_diverse_programs: int = 2
    num_inspirations: int = 5  # Number of inspiration programs to sample (deprecated, use num_inspiration_programs)
    num_inspiration_programs: int = 5  # Number of inspiration programs to include
    num_previous_programs: int = 3  # Number of previous attempts to include
    num_references: int = 3  # Number of reference programs for RL mode (get_references)

    # Template stochasticity
    use_template_stochasticity: bool = True
    template_variations: Dict[str, List[str]] = field(default_factory=dict)

    # Meta-prompting
    use_meta_prompting: bool = False
    meta_prompt_weight: float = 0.1

    # Artifact rendering
    include_artifacts: bool = True
    max_artifact_bytes: int = 20 * 1024  # 20KB in prompt
    artifact_security_filter: bool = True

    # Feature extraction and program labeling
    suggest_simplification_after_chars: Optional[int] = 500  # Suggest simplifying if program exceeds this many characters
    include_changes_under_chars: Optional[int] = 100  # Include change descriptions in features if under this length
    concise_implementation_max_lines: Optional[int] = 10  # Label as "concise" if program has this many lines or fewer
    comprehensive_implementation_min_lines: Optional[int] = 50  # Label as "comprehensive" if program has this many lines or more
    
    # Backward compatibility - deprecated
    code_length_threshold: Optional[int] = None  # Deprecated: use suggest_simplification_after_chars


@dataclass
class DatabaseConfig:
    """Configuration for the program database"""

    # General settings
    db_path: Optional[str] = None  # Path to store database on disk
    in_memory: bool = True

    # Prompt and response logging to programs/<id>.json
    log_prompts: bool = True

    # Evolutionary parameters
    population_size: int = 1000
    archive_size: int = 100
    num_islands: int = 5

    # Selection parameters
    elite_selection_ratio: float = 0.1
    exploration_ratio: float = 0.2
    exploitation_ratio: float = 0.7
    num_inspirations: int = 5  # Number of inspiration programs to sample (can be overridden by PromptConfig)
    diversity_metric: str = "edit_distance"  # Options: "edit_distance", "feature_based"

    # Feature map dimensions for MAP-Elites
    # Default to complexity and diversity for better exploration
    feature_dimensions: List[str] = field(default_factory=lambda: ["complexity", "diversity"])
    feature_bins: Union[int, Dict[str, int]] = 10  # Can be int (all dims) or dict (per-dim)
    diversity_reference_size: int = 20  # Size of reference set for diversity calculation

    # Migration parameters for island-based evolution
    migration_interval: int = 50  # Migrate every N generations
    migration_rate: float = 0.1  # Fraction of population to migrate

    # Random seed for reproducible sampling
    random_seed: Optional[int] = 42

    # Artifact storage
    artifacts_base_path: Optional[str] = None  # Defaults to db_path/artifacts
    artifact_size_threshold: int = 32 * 1024  # 32KB threshold
    cleanup_old_artifacts: bool = True
    artifact_retention_days: int = 30

    # Cemetery configuration
    cemetery_size: int = 100  # Max number of discarded programs to keep in the cemetery
    cemetery_sampling_weight: float = 0.0  # Probability (0.0 to 1.0) of sampling a parent from the cemetery


@dataclass
class EvaluatorConfig:
    """Configuration for program evaluation"""

    # General settings
    timeout: int = 300  # Maximum evaluation time in seconds
    max_retries: int = 3

    # Resource limits for evaluation
    memory_limit_mb: Optional[int] = None
    cpu_limit: Optional[float] = None

    # Evaluation strategies
    cascade_evaluation: bool = True
    cascade_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.75, 0.9])

    # Parallel evaluation
    parallel_evaluations: int = 1
    distributed: bool = False

    # LLM-based feedback
    use_llm_feedback: bool = False
    llm_feedback_weight: float = 0.1

    # Artifact handling
    enable_artifacts: bool = True
    max_artifact_storage: int = 100 * 1024 * 1024  # 100MB per program


@dataclass
@dataclass
class DebugConfig:
    """Configuration for debugging"""
    simple_debug: bool = False
    pipeline_debug: bool = False
    prompt_debug: bool = False  # Global prompt debug (for backward compatibility)
    
    # Per-agent prompt debug switches
    evolve_prompt_debug: Optional[bool] = None  # If None, uses prompt_debug
    summary_prompt_debug: Optional[bool] = None  # If None, uses prompt_debug
    gradient_prompt_debug: Optional[bool] = None  # If None, uses prompt_debug
    sample_prompt_debug: Optional[bool] = None  # If None, uses prompt_debug
    rollout_prompt_debug: Optional[bool] = None  # If None, uses prompt_debug
    
    # Response debug switches
    response_debug: bool = False  # Global response debug
    evolve_response_debug: Optional[bool] = None  # If None, uses response_debug
    summary_response_debug: Optional[bool] = None  # If None, uses response_debug
    gradient_response_debug: Optional[bool] = None  # If None, uses response_debug
    sample_response_debug: Optional[bool] = None  # If None, uses response_debug
    rollout_response_debug: Optional[bool] = None  # If None, uses response_debug
    
    def get_prompt_debug(self, agent: str) -> bool:
        """
        Get prompt debug setting for a specific agent.
        
        Args:
            agent: Agent name ('evolve', 'summary', 'gradient', 'sample', 'rollout')
        
        Returns:
            True if prompt debug is enabled for this agent
        """
        agent_attr = f"{agent}_prompt_debug"
        if hasattr(self, agent_attr):
            agent_value = getattr(self, agent_attr)
            if agent_value is not None:
                return agent_value
        # Fallback to global prompt_debug
        return self.prompt_debug
    
    def get_response_debug(self, agent: str) -> bool:
        """
        Get response debug setting for a specific agent.
        
        Args:
            agent: Agent name ('evolve', 'summary', 'gradient', 'sample', 'rollout')
        
        Returns:
            True if response debug is enabled for this agent
        """
        agent_attr = f"{agent}_response_debug"
        if hasattr(self, agent_attr):
            agent_value = getattr(self, agent_attr)
            if agent_value is not None:
                return agent_value
        # Fallback to global response_debug
        return self.response_debug


@dataclass
class Config:
    """Master configuration for OpenEvolve"""

    # General settings
    max_iterations: int = 10000
    checkpoint_interval: int = 100
    log_level: str = "INFO"
    log_dir: Optional[str] = None
    random_seed: Optional[int] = 42
    language: str = None

    # Component configurations
    llm: LLMConfig = field(default_factory=LLMConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    multi_agent: MultiAgentConfig = field(default_factory=MultiAgentConfig)

    # Evolution settings
    diff_based_evolution: bool = True
    rl_mode: bool = False
    
    # RL mode settings
    gradient_interval: int = 10  # How often (in iterations) to update the strategic gradient
    gradient_batch_size: int = 20  # How many population samples to use for gradient synthesis
    
    # Population rollout weights (for weighted sampling based on normalized metrics)
    # These weights control the proportion of different types of population programs in gradient synthesis
    # Programs are classified based on normalized metrics (z-score normalized)
    rollout_weight_all_improved: float = 0.6  # Weight for programs where all normalized metrics > 0
    rollout_weight_mixed: float = 0.2  # Weight for programs with mixed normalized metrics
    rollout_weight_all_degraded: float = 0.2  # Weight for programs where all normalized metrics < 0
    max_code_length: int = 10000

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from a YAML file"""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create configuration from a dictionary"""
        # Handle nested configurations
        config = Config()

        # Update top-level fields
        for key, value in config_dict.items():
            if key not in ["llm", "prompt", "database", "evaluator"] and hasattr(config, key):
                setattr(config, key, value)

        # Helper function to process environment variables in model configs
        def process_env_vars_in_model(model_dict):
            """Process environment variables in model configuration."""
            if "api_key" in model_dict:
                api_key = str(model_dict["api_key"]).strip()
                if api_key.startswith("${") and api_key.endswith("}"):
                    env_var_name = api_key[2:-1]
                    api_key_env = os.environ.get(env_var_name)
                    if api_key_env is not None:
                        model_dict["api_key"] = api_key_env
            return model_dict
        
        # Update nested configs
        if "llm" in config_dict:
            llm_dict = config_dict["llm"]
            
            if "models" in llm_dict:
                # Process environment variables for each model before creating LLMModelConfig
                processed_models = []
                for m in llm_dict["models"]:
                    processed_model = process_env_vars_in_model(dict(m))
                    processed_models.append(LLMModelConfig(**processed_model))
                llm_dict["models"] = processed_models
                
            if "evaluator_models" in llm_dict:
                # Process environment variables for each evaluator model
                processed_eval_models = []
                for m in llm_dict["evaluator_models"]:
                    processed_model = process_env_vars_in_model(dict(m))
                    processed_eval_models.append(LLMModelConfig(**processed_model))
                llm_dict["evaluator_models"] = processed_eval_models
                
            # Also handle top-level api_key for backward compatibility
            api_key = llm_dict.get("api_key", "").strip()
            if api_key.startswith("${") and api_key.endswith("}"):
                api_key_env = os.environ.get(api_key[2:-1])
                if api_key_env is not None:
                    llm_dict["api_key"] = api_key_env
            config.llm = LLMConfig(**llm_dict)
        if "prompt" in config_dict:
            prompt_dict = config_dict["prompt"].copy()
            # Handle compatibility: if num_inspiration_programs is provided, use it for num_inspirations too
            if "num_inspiration_programs" in prompt_dict and "num_inspirations" not in prompt_dict:
                prompt_dict["num_inspirations"] = prompt_dict["num_inspiration_programs"]
            # Also handle reverse: if num_inspirations is provided but not num_inspiration_programs
            elif "num_inspirations" in prompt_dict and "num_inspiration_programs" not in prompt_dict:
                prompt_dict["num_inspiration_programs"] = prompt_dict["num_inspirations"]
            config.prompt = PromptConfig(**prompt_dict)
        if "database" in config_dict:
            config.database = DatabaseConfig(**config_dict["database"])
        if "debug" in config_dict:
            config.debug = DebugConfig(**config_dict["debug"])

        # Ensure database inherits the random seed if not explicitly set
        if config.database.random_seed is None and config.random_seed is not None:
            config.database.random_seed = config.random_seed
        if "evaluator" in config_dict:
            config.evaluator = EvaluatorConfig(**config_dict["evaluator"])
        
        # Parse multi-agent configuration
        if "multi_agent" in config_dict:
            multi_agent_dict = config_dict["multi_agent"]
            multi_agent_config = MultiAgentConfig()
            
            # Helper function to parse agent config
            def parse_agent_config(agent_dict: Dict[str, Any]) -> AgentLLMConfig:
                """Parse a single agent's LLM configuration"""
                agent_config = AgentLLMConfig()
                
                if "api_base" in agent_dict:
                    agent_config.api_base = agent_dict["api_base"]
                if "api_key" in agent_dict:
                    # Process environment variables for api_key
                    api_key = str(agent_dict["api_key"]).strip()
                    if api_key.startswith("${") and api_key.endswith("}"):
                        env_var_name = api_key[2:-1]
                        api_key_env = os.environ.get(env_var_name)
                        if api_key_env is not None:
                            agent_config.api_key = api_key_env
                        else:
                            agent_config.api_key = api_key  # Keep original if env var not found
                    else:
                        agent_config.api_key = api_key
                if "temperature" in agent_dict:
                    agent_config.temperature = agent_dict["temperature"]
                if "top_p" in agent_dict:
                    agent_config.top_p = agent_dict["top_p"]
                if "max_tokens" in agent_dict:
                    agent_config.max_tokens = agent_dict["max_tokens"]
                if "timeout" in agent_dict:
                    agent_config.timeout = agent_dict["timeout"]
                if "retries" in agent_dict:
                    agent_config.retries = agent_dict["retries"]
                if "retry_delay" in agent_dict:
                    agent_config.retry_delay = agent_dict["retry_delay"]
                if "interval" in agent_dict:
                    agent_config.interval = agent_dict["interval"]
                if "norm" in agent_dict:
                    agent_config.norm = agent_dict["norm"]
                
                # Parse models if provided
                if "models" in agent_dict:
                    models_list = agent_dict["models"]
                    if isinstance(models_list, list):
                        agent_config.models = []
                        for model_dict in models_list:
                            # Process environment variables
                            processed_model = process_env_vars_in_model(model_dict.copy())
                            agent_config.models.append(LLMModelConfig(**processed_model))
                    elif isinstance(models_list, dict):
                        # Single model as dict
                        processed_model = process_env_vars_in_model(models_list.copy())
                        agent_config.models = [LLMModelConfig(**processed_model)]
                elif "primary_model" in agent_dict or "secondary_model" in agent_dict:
                    # Backward compatibility: support primary_model/secondary_model
                    if "primary_model" in agent_dict:
                        primary = LLMModelConfig(
                            name=agent_dict["primary_model"],
                            weight=agent_dict.get("primary_model_weight", 0.8)
                        )
                        agent_config.models.append(primary)
                    if "secondary_model" in agent_dict:
                        secondary = LLMModelConfig(
                            name=agent_dict["secondary_model"],
                            weight=agent_dict.get("secondary_model_weight", 0.2)
                        )
                        agent_config.models.append(secondary)
                
                agent_config.__post_init__()
                return agent_config
            
            # Parse each agent configuration
            for agent_name in ["evolve", "summary", "gradient", "sample"]:
                if agent_name in multi_agent_dict:
                    setattr(multi_agent_config, agent_name, parse_agent_config(multi_agent_dict[agent_name]))
            
            # Parse history configuration if provided
            if "history" in multi_agent_dict:
                history_dict = multi_agent_dict["history"]
                history_config = HistoryConfig(
                    rollout_batch_size=history_dict.get("rollout_batch_size", 20),
                    rollout_weight_all_improved=history_dict.get("rollout_weight_all_improved", 0.6),
                    rollout_weight_mixed=history_dict.get("rollout_weight_mixed", 0.2),
                    rollout_weight_all_degraded=history_dict.get("rollout_weight_all_degraded", 0.2),
                )
                multi_agent_config.history = history_config
            
            config.multi_agent = multi_agent_config

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary"""
        return {
            # General settings
            "max_iterations": self.max_iterations,
            "checkpoint_interval": self.checkpoint_interval,
            "log_level": self.log_level,
            "log_dir": self.log_dir,
            "random_seed": self.random_seed,
            # Component configurations
            "llm": {
                "models": self.llm.models,
                "evaluator_models": self.llm.evaluator_models,
                "api_base": self.llm.api_base,
                "temperature": self.llm.temperature,
                "top_p": self.llm.top_p,
                "max_tokens": self.llm.max_tokens,
                "timeout": self.llm.timeout,
                "retries": self.llm.retries,
                "retry_delay": self.llm.retry_delay,
            },
            "prompt": {
                "template_dir": self.prompt.template_dir,
                "system_message": self.prompt.system_message,
                "evaluator_system_message": self.prompt.evaluator_system_message,
                "num_top_programs": self.prompt.num_top_programs,
                "num_diverse_programs": self.prompt.num_diverse_programs,
                "num_inspirations": self.prompt.num_inspirations,
                "num_references": self.prompt.num_references,
                "use_template_stochasticity": self.prompt.use_template_stochasticity,
                "template_variations": self.prompt.template_variations,
                # Note: meta-prompting features not implemented
                # "use_meta_prompting": self.prompt.use_meta_prompting,
                # "meta_prompt_weight": self.prompt.meta_prompt_weight,
            },
            "database": {
                "db_path": self.database.db_path,
                "in_memory": self.database.in_memory,
                "population_size": self.database.population_size,
                "archive_size": self.database.archive_size,
                "num_islands": self.database.num_islands,
                "elite_selection_ratio": self.database.elite_selection_ratio,
                "exploration_ratio": self.database.exploration_ratio,
                "exploitation_ratio": self.database.exploitation_ratio,
                "num_inspirations": self.database.num_inspirations,
                # Note: diversity_metric fixed to "edit_distance"
                # "diversity_metric": self.database.diversity_metric,
                "feature_dimensions": self.database.feature_dimensions,
                "feature_bins": self.database.feature_bins,
                "migration_interval": self.database.migration_interval,
                "migration_rate": self.database.migration_rate,
                "random_seed": self.database.random_seed,
                "log_prompts": self.database.log_prompts,
            },
            "evaluator": {
                "timeout": self.evaluator.timeout,
                "max_retries": self.evaluator.max_retries,
                # Note: resource limits not implemented
                # "memory_limit_mb": self.evaluator.memory_limit_mb,
                # "cpu_limit": self.evaluator.cpu_limit,
                "cascade_evaluation": self.evaluator.cascade_evaluation,
                "cascade_thresholds": self.evaluator.cascade_thresholds,
                "parallel_evaluations": self.evaluator.parallel_evaluations,
                # Note: distributed evaluation not implemented
                # "distributed": self.evaluator.distributed,
                "use_llm_feedback": self.evaluator.use_llm_feedback,
                "llm_feedback_weight": self.evaluator.llm_feedback_weight,
            },
            "debug": {
                "simple_debug": self.debug.simple_debug,
                "pipeline_debug": self.debug.pipeline_debug,
                "prompt_debug": self.debug.prompt_debug,
                "evolve_prompt_debug": self.debug.evolve_prompt_debug,
                "summary_prompt_debug": self.debug.summary_prompt_debug,
                "gradient_prompt_debug": self.debug.gradient_prompt_debug,
                "sample_prompt_debug": self.debug.sample_prompt_debug,
                "response_debug": self.debug.response_debug,
                "evolve_response_debug": self.debug.evolve_response_debug,
                "summary_response_debug": self.debug.summary_response_debug,
                "gradient_response_debug": self.debug.gradient_response_debug,
                "sample_response_debug": self.debug.sample_response_debug,
            },
            # Multi-agent configuration
            "multi_agent": {
                "evolve": {
                    "api_base": self.multi_agent.evolve.api_base if self.multi_agent.evolve else None,
                    "api_key": self.multi_agent.evolve.api_key if self.multi_agent.evolve else None,
                    "models": [asdict(m) for m in self.multi_agent.evolve.models] if self.multi_agent.evolve else [],
                    "temperature": self.multi_agent.evolve.temperature if self.multi_agent.evolve else None,
                    "top_p": self.multi_agent.evolve.top_p if self.multi_agent.evolve else None,
                    "max_tokens": self.multi_agent.evolve.max_tokens if self.multi_agent.evolve else None,
                } if self.multi_agent.evolve else None,
                "summary": {
                    "api_base": self.multi_agent.summary.api_base if self.multi_agent.summary else None,
                    "api_key": self.multi_agent.summary.api_key if self.multi_agent.summary else None,
                    "models": [asdict(m) for m in self.multi_agent.summary.models] if self.multi_agent.summary else [],
                    "temperature": self.multi_agent.summary.temperature if self.multi_agent.summary else None,
                    "top_p": self.multi_agent.summary.top_p if self.multi_agent.summary else None,
                    "max_tokens": self.multi_agent.summary.max_tokens if self.multi_agent.summary else None,
                } if self.multi_agent.summary else None,
                "gradient": {
                    "api_base": self.multi_agent.gradient.api_base if self.multi_agent.gradient else None,
                    "api_key": self.multi_agent.gradient.api_key if self.multi_agent.gradient else None,
                    "models": [asdict(m) for m in self.multi_agent.gradient.models] if self.multi_agent.gradient else [],
                    "temperature": self.multi_agent.gradient.temperature if self.multi_agent.gradient else None,
                    "top_p": self.multi_agent.gradient.top_p if self.multi_agent.gradient else None,
                    "max_tokens": self.multi_agent.gradient.max_tokens if self.multi_agent.gradient else None,
                    "interval": self.multi_agent.gradient.interval if self.multi_agent.gradient else None,
                    "norm": self.multi_agent.gradient.norm if self.multi_agent.gradient else None,
                } if self.multi_agent.gradient else None,
                "sample": {
                    "api_base": self.multi_agent.sample.api_base if self.multi_agent.sample else None,
                    "api_key": self.multi_agent.sample.api_key if self.multi_agent.sample else None,
                    "models": [asdict(m) for m in self.multi_agent.sample.models] if self.multi_agent.sample else [],
                    "temperature": self.multi_agent.sample.temperature if self.multi_agent.sample else None,
                    "top_p": self.multi_agent.sample.top_p if self.multi_agent.sample else None,
                    "max_tokens": self.multi_agent.sample.max_tokens if self.multi_agent.sample else None,
                } if self.multi_agent.sample else None,
                "history": {
                    "rollout_batch_size": self.multi_agent.history.rollout_batch_size if self.multi_agent.history else None,
                    "rollout_weight_all_improved": self.multi_agent.history.rollout_weight_all_improved if self.multi_agent.history else None,
                    "rollout_weight_mixed": self.multi_agent.history.rollout_weight_mixed if self.multi_agent.history else None,
                    "rollout_weight_all_degraded": self.multi_agent.history.rollout_weight_all_degraded if self.multi_agent.history else None,
                } if self.multi_agent.history else None,
            },
            # Evolution settings
            "diff_based_evolution": self.diff_based_evolution,
            "rl_mode": self.rl_mode,
            "max_code_length": self.max_code_length,
            # Keep these for backward compatibility, but prefer multi_agent config
            "gradient_interval": self.gradient_interval,
            "gradient_batch_size": self.gradient_batch_size,
            "rollout_weight_all_improved": self.rollout_weight_all_improved,
            "rollout_weight_mixed": self.rollout_weight_mixed,
            "rollout_weight_all_degraded": self.rollout_weight_all_degraded,
        }

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to a YAML file"""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """Load configuration from a YAML file or use defaults"""
    if config_path and os.path.exists(config_path):
        config = Config.from_yaml(config_path)
    else:
        config = Config()

        # Use environment variables if available
        api_key = os.environ.get("OPENAI_API_KEY")
        api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")

        config.llm.update_model_params({"api_key": api_key, "api_base": api_base})

    # Make the system message available to the individual models, in case it is not provided from the prompt sampler
    config.llm.update_model_params({"system_message": config.prompt.system_message})

    return config
