"""HuggingFace adapter implementation for sparse attention."""

import random
import string
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from ..sparse_attention.base import SparseAttention, SparseAttentionConfig
from ..sparse_attention.research_attention.base import ResearchAttention
from .base import ModelAdapter, Request, RequestResponse

INT_MAX = 2**31 - 1


class ModelAdapterHF(ModelAdapter):
    """ModelAdapter for HuggingFace integration. Provides concrete implementations for huggingface's
    transformer library.
    """

    def __init__(
        self,
        model_name: str,
        sparse_attention_config: Optional[SparseAttentionConfig],
        model_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        """Initialize HuggingFace adapter.

        Args:
            model_name: Name of the HuggingFace model to use
            sparse_attention_config: Configuration for sparse attention. If None, adapter runs in dense-only mode.
            model_kwargs: Additional keyword arguments for model creation
            device: Device to run the model on TODO: support dynamic and multipledevice placement
            tokenizer_kwargs: Additional keyword arguments for tokenizer creation
        """
        super().__init__(model_name, sparse_attention_config, **kwargs)
        self._registered_attention_name: Optional[str] = None
        self._custom_attention_fn: Optional[Callable] = None
        self.model_kwargs = model_kwargs or {}
        self.tokenizer_kwargs = tokenizer_kwargs or {}

        # more useful parameters to store
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.torch_dtype = self.model_kwargs.get("torch_dtype", torch.float32)

        # Handle dense-only mode when sparse_attention_config is None
        self._sparse_attention_available: bool = sparse_attention_config is not None

        # create model and tokenizer
        # Support local model path, HuggingFace cache, and mirror
        import os
        from pathlib import Path
        
        # Check if model_name is a local path
        is_local_path = os.path.exists(self.model_name) and os.path.isdir(self.model_name)
        
        # Check HuggingFace cache directories
        hf_home = os.environ.get("HF_HOME")
        transformers_cache = os.environ.get("TRANSFORMERS_CACHE")
        default_cache = os.path.expanduser("~/.cache/huggingface")
        
        # Try to find model in cache
        model_in_cache = False
        if not is_local_path:
            # Check common cache locations
            cache_dirs = []
            if hf_home:
                cache_dirs.append(Path(hf_home) / "hub")
            if transformers_cache:
                cache_dirs.append(Path(transformers_cache))
            cache_dirs.append(Path(default_cache) / "hub")
            
            # Transformers uses a specific cache structure: models--org--model_name
            cache_model_name = self.model_name.replace("/", "--")
            for cache_dir in cache_dirs:
                if cache_dir.exists():
                    # Look for the model in cache
                    model_cache_path = cache_dir / f"models--{cache_model_name}"
                    if model_cache_path.exists():
                        # Check for snapshots directory
                        snapshots_dir = model_cache_path / "snapshots"
                        if snapshots_dir.exists():
                            # Get the latest snapshot
                            snapshots = list(snapshots_dir.iterdir())
                            if snapshots:
                                latest_snapshot = max(snapshots, key=lambda p: p.stat().st_mtime)
                                if (latest_snapshot / "config.json").exists():
                                    print(f"Found model in cache: {latest_snapshot}")
                                    model_in_cache = True
                                    break
        
        # Prepare kwargs for from_pretrained
        model_load_kwargs = self.model_kwargs.copy()
        tokenizer_load_kwargs = self.tokenizer_kwargs.copy()
        
        # If local path, use local_files_only
        if is_local_path:
            model_load_kwargs["local_files_only"] = True
            tokenizer_load_kwargs["local_files_only"] = True
            print(f"Loading model from local path: {self.model_name}")
        elif model_in_cache:
            # Try to use cache first, but allow fallback to download if needed
            print(f"Using cached model, will download missing files if needed")
        else:
            # Check for HuggingFace mirror via environment variable
            hf_endpoint = os.environ.get("HF_ENDPOINT")
            if hf_endpoint:
                # Use mirror endpoint
                model_load_kwargs["trust_remote_code"] = model_load_kwargs.get("trust_remote_code", True)
                # Note: transformers library uses HF_ENDPOINT environment variable automatically
                print(f"Using HuggingFace endpoint: {hf_endpoint}")
            else:
                print(f"Model not found in cache. Will download from HuggingFace Hub.")
                print(f"  To use local cache, set HF_HOME or TRANSFORMERS_CACHE environment variable")
                print(f"  To use mirror, set HF_ENDPOINT environment variable (e.g., https://hf-mirror.com)")
        
        # Try to load model with retry logic for network issues
        max_retries = 3
        retry_delay = 2
        for attempt in range(max_retries):
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, **model_load_kwargs
                )
                print(f"✓ Model loaded successfully")
                break
            except Exception as e:
                error_str = str(e)
                if "Network is unreachable" in error_str or "Failed to establish" in error_str:
                    if attempt < max_retries - 1:
                        import time
                        print(f"⚠ Network error loading model (attempt {attempt + 1}/{max_retries}): {error_str[:200]}")
                        print(f"  Retrying in {retry_delay}s...")
                        print(f"  Tip: Set HF_ENDPOINT for mirror or use local model path")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        print(f"❌ Failed to load model after {max_retries} attempts")
                        print(f"  Solutions:")
                        print(f"  1. Download model manually and set SPARSE_ATTENTION_MODEL_PATH to local path")
                        print(f"  2. Set HF_ENDPOINT environment variable to use mirror (e.g., export HF_ENDPOINT=https://hf-mirror.com)")
                        print(f"  3. Ensure network connectivity to huggingface.co")
                        raise
                else:
                    raise
        
        # Load tokenizer
        retry_delay = 2  # Reset delay for tokenizer
        for attempt in range(max_retries):
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, **tokenizer_load_kwargs
                )
                print(f"✓ Tokenizer loaded successfully")
                break
            except Exception as e:
                error_str = str(e)
                if "Network is unreachable" in error_str or "Failed to establish" in error_str:
                    if attempt < max_retries - 1:
                        import time
                        print(f"⚠ Network error loading tokenizer (attempt {attempt + 1}/{max_retries}): {error_str[:200]}")
                        print(f"  Retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        print(f"❌ Failed to load tokenizer after {max_retries} attempts")
                        raise
                else:
                    raise
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # TODO: support dynamic and multipledevice placement
        self.model.to(self.device)
        self.random_separator = "".join(
            random.choices(string.ascii_lowercase + string.digits, k=100)
        )

    def __del__(self) -> None:
        """Clean up registered attention functions when the adapter is destroyed."""
        self._cleanup_attention_registration()

    def process_request(
        self,
        request: Request,
        generation_kwargs: Dict[str, Any],
        request_kwargs: Dict[str, Any],
        **kwargs: Dict[str, Any],
    ) -> RequestResponse:
        """Processes request with optimized tokenization but independent question processing.
        Context is tokenized once but each question is processed independently to avoid KV cache contamination.

        Args:
            request: The request to process

        Returns:
            response: The response to the request
        """
        max_context_length: int = request_kwargs.get("max_context_length", INT_MAX)

        questions: List[str] = (
            request.questions
            if isinstance(request.questions, list)
            else [request.questions]
        )
        context: str = request.context

        context, questions = self._preprocess_context_and_questions(context, questions)

        context_tokens = self.tokenizer.encode(context, return_tensors="pt")
        context_tokens = context_tokens[
            :, :max_context_length
        ]  # truncate context to max_context_length
        if self.device is not None:
            context_tokens = context_tokens.to(self.device)
        print(f"Context tokens: {context_tokens.shape}")
        responses: List[str] = []

        self.model.eval()
        with torch.no_grad():
            for question in questions:
                sparse_meta_data: Dict[str, Any] = {}

                question_tokens = self.tokenizer.encode(question, return_tensors="pt")
                if self.device is not None:
                    question_tokens = question_tokens.to(self.device)

                context_outputs = self.model(
                    context_tokens,
                    past_key_values=None,
                    use_cache=True,
                    sparse_meta_data=sparse_meta_data,
                )

                if self._sparse_attention_available:
                    with self.enable_sparse_mode():
                        response_text = self._generate_response(
                            question_tokens,
                            context_outputs,
                            sparse_meta_data,
                            generation_kwargs,
                            **kwargs,
                        )
                        responses.append(response_text)
                else:
                    # Dense-only mode: process questions with dense attention
                    response_text = self._generate_response(
                        question_tokens,
                        context_outputs,
                        sparse_meta_data,
                        generation_kwargs,
                        **kwargs,
                    )
                    responses.append(response_text)

        if isinstance(request.questions, str):
            return RequestResponse(responses=responses[0])
        else:
            return RequestResponse(responses=responses)

    def _preprocess_context_and_questions(
        self, context: str, questions: List[str]
    ) -> Tuple[str, List[str]]:
        """Preprocess the context and questions -- apply chat template if needed

        Args:
            context: The context to preprocess
            questions: The questions to preprocess
        """
        context = context + self.random_separator
        if self.tokenizer.chat_template is not None:
            context = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": context}],
                tokenize=False,
                add_generation_prompt=True,
            )
        new_context = context.split(self.random_separator)[0]
        new_questions = [
            question + context.split(self.random_separator)[1] for question in questions
        ]
        return new_context, new_questions

    def get_custom_attention_function(
        self, sparse_attention: SparseAttention
    ) -> Callable:
        """Returns custom_attention_fn callable with the correct signature required for HuggingFace.

        Args:
            sparse_attention: The sparse attention instance

        Returns:
            custom_attention_fn: Callable with correct signature for HuggingFace
        """

        def custom_attention_callable(
            module: torch.nn.Module,
            queries: torch.Tensor,
            keys: torch.Tensor,
            values: torch.Tensor,
            attention_mask: Optional[torch.Tensor],
            scaling: float = 1.0,
            dropout: float = 0.0,
            **kwargs: Dict[str, Any],
        ):
            """Custom attention callable for HuggingFace integration."""
            if hasattr(module, "layer_idx"):
                layer_idx = getattr(module, "layer_idx", None)
                if layer_idx is not None:
                    kwargs["layer_idx"] = layer_idx

            if "sparse_meta_data" in kwargs:
                sparse_meta_data: Dict[Any, Any] = kwargs["sparse_meta_data"]
                kwargs.pop("sparse_meta_data", None)
            else:
                raise ValueError(
                    "sparse_meta_data must be provided while calling model.forward()"
                )

            return sparse_attention.custom_attention(
                module=module,
                queries=queries,
                keys=keys,
                values=values,
                attention_mask=attention_mask,
                scaling=scaling,
                dropout=dropout,
                sparse_meta_data=sparse_meta_data,
                **kwargs,
            )

        return custom_attention_callable

    def _generate_unique_attention_name(self) -> str:
        """Generate a unique name not present in ALL_ATTENTION_FUNCTIONS."""
        base_name: str = "sparse_attention"
        existing_keys: List[str] = (
            ALL_ATTENTION_FUNCTIONS.valid_keys()
            + ALL_MASK_ATTENTION_FUNCTIONS.valid_keys()
        )

        while True:
            suffix: str = "".join(
                random.choices(string.ascii_lowercase + string.digits, k=8)
            )
            name: str = f"{base_name}_{suffix}"

            if name not in existing_keys:
                return name

    def _ensure_attention_registered(self) -> str:
        """Ensure custom attention function is registered and return the name.
        Caches the registration to avoid repeated registration overhead.

        Returns:
            The name of the registered attention function
        """
        if self._registered_attention_name is None:
            if not self._sparse_attention_available or self.sparse_attention is None:
                raise RuntimeError(
                    "Cannot register attention function: sparse attention is not available"
                )
            self._custom_attention_fn = self.get_custom_attention_function(
                self.sparse_attention
            )
            self._registered_attention_name = self._generate_unique_attention_name()

            from transformers.masking_utils import eager_mask

            ALL_ATTENTION_FUNCTIONS.register(
                self._registered_attention_name, self._custom_attention_fn
            )
            if isinstance(self.sparse_attention, ResearchAttention):
                ALL_MASK_ATTENTION_FUNCTIONS.register(
                    self._registered_attention_name, eager_mask
                )
            else:
                raise NotImplementedError(
                    "Sparse attention is not supported for this model yet"
                )

        return self._registered_attention_name

    def _cleanup_attention_registration(self) -> None:
        """Clean up registered attention functions."""
        if (
            self._registered_attention_name is not None
            and self._registered_attention_name in ALL_ATTENTION_FUNCTIONS.valid_keys()
        ):
            ALL_ATTENTION_FUNCTIONS._global_mapping.pop(self._registered_attention_name)
        if (
            self._registered_attention_name is not None
            and self._registered_attention_name
            in ALL_MASK_ATTENTION_FUNCTIONS.valid_keys()
        ):
            ALL_MASK_ATTENTION_FUNCTIONS._global_mapping.pop(
                self._registered_attention_name
            )
        self._registered_attention_name = None
        self._custom_attention_fn = None

    @contextmanager
    def enable_sparse_mode(self) -> Generator[None, None, None]:
        """Context manager to temporarily enable sparse attention mode.

        Yields:
            None
        """
        # If sparse attention is not available, raise an error
        if not self._sparse_attention_available:
            raise RuntimeError(
                "Cannot enable sparse mode: sparse attention is not available"
            )

        # Store original implementations to restore later
        original_implementations: Dict[str, str] = {}

        # First, store the original implementations before registering custom attention
        for name, module in self.model.named_modules():
            if hasattr(module, "config") and hasattr(
                module.config, "_attn_implementation"
            ):
                original_implementations[name] = module.config._attn_implementation

        # Ensure custom attention function is registered (reuse if already registered)
        custom_attention_name: str = self._ensure_attention_registered()

        try:
            # Switch to sparse attention
            for name, module in self.model.named_modules():
                if hasattr(module, "config") and hasattr(
                    module.config, "_attn_implementation"
                ):
                    module.config._attn_implementation = custom_attention_name

            yield

        finally:
            # Restore original implementations
            for name, module in self.model.named_modules():
                if name in original_implementations:
                    module.config._attn_implementation = original_implementations[name]

    def _generate_response(
        self,
        question_tokens: torch.Tensor,
        context_outputs: Any,
        sparse_meta_data: Dict[str, Any],
        generation_kwargs: Dict[str, Any],
        **kwargs: Dict[str, Any],
    ) -> str:
        """Generate text response using greedy decoding based on kvpress pipeline approach.

        Args:
            question_tokens: The tokenized question
            context_outputs: The model outputs from processing the context
            max_new_tokens: Maximum number of new tokens to generate

        Returns:
            Generated text response

        TODO:
            move to huggingface genera`te() to leverage all possible generations
            pass generation_kwargs appropriately
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized")

        max_new_tokens: int = generation_kwargs.get("max_new_tokens", 50)  # type: ignore
        context_length: int = context_outputs.past_key_values.get_seq_length()

        position_ids = torch.arange(
            context_length,
            context_length + question_tokens.shape[1],
            device=self.model.device,
        ).unsqueeze(0)

        with torch.no_grad():
            question_outputs = self.model(
                input_ids=question_tokens,
                past_key_values=context_outputs.past_key_values,
                position_ids=position_ids,
                num_logits_to_keep=1,
                sparse_meta_data=sparse_meta_data,
            )

        position_ids = position_ids[:, -1:] + 1
        generated_ids = [question_outputs.logits[0, -1].argmax()]

        should_stop_token_ids = self.model.generation_config.eos_token_id
        if not isinstance(should_stop_token_ids, list):
            should_stop_token_ids = [should_stop_token_ids]

        for i in range(max_new_tokens - 1):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=generated_ids[-1].unsqueeze(0).unsqueeze(0),
                    past_key_values=context_outputs.past_key_values,
                    position_ids=position_ids + i,
                    sparse_meta_data=sparse_meta_data,
                )
                # TODO: support other forms of decoding
                new_id = outputs.logits[0, -1].argmax()
                generated_ids.append(new_id)

                if new_id.item() in should_stop_token_ids:
                    break

        answer: str = self.tokenizer.decode(
            torch.stack(generated_ids), skip_special_tokens=True
        )
        return answer
