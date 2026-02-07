"""
OpenAI API interface for LLMs
"""

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import openai

from openevolve.config import LLMConfig
from openevolve.llm.base import LLMInterface

logger = logging.getLogger(__name__)


class OpenAILLM(LLMInterface):
    """LLM interface using OpenAI-compatible APIs"""

    def __init__(
        self,
        model_cfg: Optional[dict] = None,
    ):
        self.model = model_cfg.name
        self.system_message = model_cfg.system_message
        self.temperature = model_cfg.temperature
        self.top_p = model_cfg.top_p
        self.max_tokens = model_cfg.max_tokens
        self.timeout = model_cfg.timeout
        self.retries = model_cfg.retries
        self.retry_delay = model_cfg.retry_delay
        self.api_base = model_cfg.api_base
        self.api_key = model_cfg.api_key
        self.random_seed = getattr(model_cfg, "random_seed", None)

        # Validate API key for OpenRouter
        if self.api_base and "openrouter.ai" in self.api_base:
            # Check if api_key looks like an unprocessed environment variable reference
            if isinstance(self.api_key, str) and self.api_key.startswith("${") and self.api_key.endswith("}"):
                env_var_name = self.api_key[2:-1]
                env_value = os.environ.get(env_var_name)
                if env_value:
                    logger.warning(
                        f"API key for {self.model} appears to be an unprocessed environment variable "
                        f"reference '{self.api_key}'. Attempting to resolve from environment..."
                    )
                    self.api_key = env_value
                else:
                    raise ValueError(
                        f"OpenRouter API key environment variable '{env_var_name}' is not set. "
                        f"Please set the {env_var_name} environment variable or provide the API key directly."
                    )
            elif not self.api_key or not self.api_key.strip():
                raise ValueError(
                    "OpenRouter API key is required but not provided. "
                    "Please set the API key in your config file or environment variable."
                )
            # OpenRouter requires HTTP-Referer header (optional but recommended)
            default_headers = {
                "HTTP-Referer": "https://github.com/openevolve/openevolve",
                "X-Title": "OpenEvolve"
            }
        else:
            default_headers = None
        
        # Set up API client
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            default_headers=default_headers,
        )
        
        # Track token usage for last API call
        self._last_usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        
        # Track cumulative token usage across all API calls
        self.token_stats: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        
        # Track time usage for last API call
        self._last_time: float = 0.0
        
        # Track cumulative time usage across all API calls
        self.time_stats: float = 0.0

        # Only log unique models to reduce duplication
        if not hasattr(logger, "_initialized_models"):
            logger._initialized_models = set()

        if self.model not in logger._initialized_models:
            logger.info(f"Initialized OpenAI LLM with model: {self.model}")
            logger._initialized_models.add(self.model)

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt"""
        return await self.generate_with_context(
            system_message=self.system_message,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    async def generate_with_context(
        self, system_message: Optional[str], messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate text using a system message and conversational context"""
        # Prepare messages - only add system message if it's not None or empty
        formatted_messages = []
        if system_message and system_message.strip():
            formatted_messages.append({"role": "system", "content": system_message})
        formatted_messages.extend(messages)

        # Set up generation parameters
        # Check if model requires max_completion_tokens (o-series and gpt-5 models)
        model_lower = str(self.model).lower()
        uses_completion_tokens = (
            self.api_base == "https://api.openai.com/v1" and 
            (model_lower.startswith("o") or "gpt-5" in model_lower)
        )
        
        if uses_completion_tokens:
            # For o-series and GPT-5 models that use response format API
            params = {
                "model": self.model,
                "messages": formatted_messages,
                "max_completion_tokens": kwargs.get("max_tokens", self.max_tokens),
            }
            # GPT-5 only supports temperature=1 (default)
            if "gpt-5" in model_lower:
                # GPT-5 requires temperature to be 1 (or omitted for default)
                # Don't include temperature/top_p parameters for GPT-5
                pass
            elif model_lower.startswith("o"):
                # o-series models don't support temperature/top_p with max_completion_tokens
                pass
        else:
            # For standard models
            params = {
                "model": self.model,
                "messages": formatted_messages,
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", self.top_p),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            }

        # Add seed parameter for reproducibility if configured
        # Skip seed parameter for Google AI Studio endpoint as it doesn't support it
        seed = kwargs.get("seed", self.random_seed)
        if seed is not None:
            if self.api_base == "https://generativelanguage.googleapis.com/v1beta/openai/":
                logger.warning(
                    "Skipping seed parameter as Google AI Studio endpoint doesn't support it. "
                    "Reproducibility may be limited."
                )
            else:
                params["seed"] = seed

        # Attempt the API call with retries
        retries = kwargs.get("retries", self.retries)
        retry_delay = kwargs.get("retry_delay", self.retry_delay)
        timeout = kwargs.get("timeout", self.timeout)

        for attempt in range(retries + 1):
            try:
                call_start_time = time.time()
                content, usage = await asyncio.wait_for(self._call_api(params), timeout=timeout)
                call_end_time = time.time()
                call_duration = call_end_time - call_start_time
                
                # Store usage info for later retrieval if needed
                self._last_usage = usage
                self._last_time = call_duration
                
                # Accumulate token usage statistics
                self.token_stats["prompt_tokens"] += usage.get("prompt_tokens", 0)
                self.token_stats["completion_tokens"] += usage.get("completion_tokens", 0)
                self.token_stats["total_tokens"] += usage.get("total_tokens", 0)
                
                # Accumulate time statistics
                self.time_stats += call_duration
                
                return content
            except asyncio.TimeoutError:
                if attempt < retries:
                    logger.warning(f"Timeout on attempt {attempt + 1}/{retries + 1}. Retrying...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with timeout")
                    raise
            except Exception as e:
                # Check if this is an authentication error (401, 403) - don't retry these
                error_str = str(e).lower()
                is_auth_error = (
                    "401" in error_str or 
                    "403" in error_str or 
                    "unauthorized" in error_str or
                    "forbidden" in error_str or
                    "authentication" in error_str or
                    "api key" in error_str or
                    "no cookie auth" in error_str
                )
                
                if is_auth_error:
                    # Authentication errors won't be fixed by retrying - fail immediately
                    logger.error(
                        f"Authentication error (attempt {attempt + 1}): {str(e)}. "
                        f"Please check your API key configuration. Not retrying."
                    )
                    raise
                elif attempt < retries:
                    logger.warning(
                        f"Error on attempt {attempt + 1}/{retries + 1}: {str(e)}. Retrying..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with error: {str(e)}")
                    raise

    async def _call_api(self, params: Dict[str, Any]) -> Tuple[str, Dict[str, int]]:
        """
        Make the actual API call
        
        Returns:
            Tuple of (content, usage_dict) where usage_dict contains:
            - prompt_tokens: int
            - completion_tokens: int
            - total_tokens: int
        """
        # Use asyncio to run the blocking API call in a thread pool
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: self.client.chat.completions.create(**params)
        )
        # Logging of system prompt, user message and response content
        logger = logging.getLogger(__name__)
        # logger.debug(f"API parameters: {params}")
        
        # Check if response is valid
        if not response or not hasattr(response, 'choices'):
            raise ValueError("Invalid API response: missing 'choices' attribute")
        
        if not response.choices or len(response.choices) == 0:
            raise ValueError("Invalid API response: 'choices' is empty")
        
        choice = response.choices[0]
        if not choice or not hasattr(choice, 'message'):
            raise ValueError("Invalid API response: missing 'message' attribute in choice")
        
        message = choice.message
        if not message or not hasattr(message, 'content'):
            raise ValueError("Invalid API response: missing 'content' attribute in message")
        
        content = message.content
        if content is None:
            raise ValueError("Invalid API response: 'content' is None")
        
        # Extract token usage from response
        usage_dict = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        if hasattr(response, 'usage') and response.usage:
            usage_dict["prompt_tokens"] = getattr(response.usage, 'prompt_tokens', 0) or 0
            usage_dict["completion_tokens"] = getattr(response.usage, 'completion_tokens', 0) or 0
            usage_dict["total_tokens"] = getattr(response.usage, 'total_tokens', 0) or 0
        
        # logger.debug(f"API response: {content}")
        # logger.debug(f"Token usage: {usage_dict}")
        return content, usage_dict
    
    def get_last_usage(self) -> Dict[str, int]:
        """Get token usage from the last API call"""
        return self._last_usage.copy()
    
    def get_token_stats(self) -> Dict[str, int]:
        """Get cumulative token usage statistics for this model"""
        return self.token_stats.copy()
    
    def get_last_time(self) -> float:
        """Get time usage from the last API call in seconds"""
        return self._last_time
    
    def get_time_stats(self) -> float:
        """Get cumulative time usage statistics in seconds"""
        return self.time_stats
