"""
Model ensemble for LLMs
"""

import asyncio
import logging
import random
from typing import Dict, List, Optional, Tuple

from openevolve.llm.base import LLMInterface
from openevolve.llm.openai import OpenAILLM
from openevolve.config import LLMModelConfig

logger = logging.getLogger(__name__)


class LLMEnsemble:
    """Ensemble of LLMs"""

    def __init__(self, models_cfg: List[LLMModelConfig]):
        self.models_cfg = models_cfg

        # Initialize models from the configuration
        self.models = [OpenAILLM(model_cfg) for model_cfg in models_cfg]

        # Extract and normalize model weights
        self.weights = [model.weight for model in models_cfg]
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

        # Set up random state for deterministic model selection
        self.random_state = random.Random()
        # Initialize with seed from first model's config if available
        if (
            models_cfg
            and hasattr(models_cfg[0], "random_seed")
            and models_cfg[0].random_seed is not None
        ):
            self.random_state.seed(models_cfg[0].random_seed)
            logger.debug(
                f"LLMEnsemble: Set random seed to {models_cfg[0].random_seed} for deterministic model selection"
            )

        # Only log if we have multiple models or this is the first ensemble
        if len(models_cfg) > 1 or not hasattr(logger, "_ensemble_logged"):
            logger.info(
                f"Initialized LLM ensemble with models: "
                + ", ".join(
                    f"{model.name} (weight: {weight:.2f})"
                    for model, weight in zip(models_cfg, self.weights)
                )
            )
            logger._ensemble_logged = True

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using a randomly selected model based on weights"""
        model = self._sample_model()
        return await model.generate(prompt, **kwargs)

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate text using a system message and conversational context"""
        model = self._sample_model()
        self._last_sampled_model = model  # Store for token usage retrieval
        return await model.generate_with_context(system_message, messages, **kwargs)

    def _sample_model(self) -> LLMInterface:
        """Sample a model from the ensemble based on weights"""
        index = self.random_state.choices(range(len(self.models)), weights=self.weights, k=1)[0]
        sampled_model = self.models[index]
        logger.info(f"Sampled model: {vars(sampled_model)['model']}")
        return sampled_model
    
    def get_last_usage(self) -> Dict[str, int]:
        """
        Get token usage from the last API call.
        Returns usage from the last sampled model.
        """
        # Get usage from the last sampled model (if it's an OpenAILLM)
        if hasattr(self, '_last_sampled_model'):
            if isinstance(self._last_sampled_model, OpenAILLM):
                return self._last_sampled_model.get_last_usage()
        # Fallback: try to get from any model
        for model in self.models:
            if isinstance(model, OpenAILLM):
                return model.get_last_usage()
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    def get_token_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Get cumulative token usage statistics for all models in the ensemble
        
        Returns:
            Dictionary mapping model name to token statistics with keys:
            - prompt_tokens: Total input tokens
            - completion_tokens: Total output tokens
            - total_tokens: Total tokens
        """
        stats = {}
        for model in self.models:
            if isinstance(model, OpenAILLM):
                model_name = model.model
                stats[model_name] = model.get_token_stats()
        return stats
    
    def get_last_time(self) -> float:
        """
        Get time usage from the last API call.
        Returns time from the last sampled model.
        """
        # Get time from the last sampled model (if it's an OpenAILLM)
        if hasattr(self, '_last_sampled_model'):
            if isinstance(self._last_sampled_model, OpenAILLM):
                return self._last_sampled_model.get_last_time()
        # Fallback: try to get from any model
        for model in self.models:
            if isinstance(model, OpenAILLM):
                return model.get_last_time()
        return 0.0
    
    def get_time_stats(self) -> Dict[str, float]:
        """
        Get cumulative time usage statistics for all models in the ensemble
        
        Returns:
            Dictionary mapping model name to time statistics (in seconds)
        """
        stats = {}
        for model in self.models:
            if isinstance(model, OpenAILLM):
                model_name = model.model
                stats[model_name] = model.get_time_stats()
        return stats

    async def generate_multiple(self, prompt: str, n: int, **kwargs) -> List[str]:
        """Generate multiple texts in parallel"""
        tasks = [self.generate(prompt, **kwargs) for _ in range(n)]
        return await asyncio.gather(*tasks)

    async def parallel_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for multiple prompts in parallel"""
        tasks = [self.generate(prompt, **kwargs) for prompt in prompts]
        return await asyncio.gather(*tasks)

    async def generate_all_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate text using a all available models and average their returned metrics"""
        responses = []
        for model in self.models:
            responses.append(await model.generate_with_context(system_message, messages, **kwargs))
        return responses
