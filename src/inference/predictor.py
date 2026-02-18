"""Inference using vLLM for efficient text generation."""

from typing import Optional, List
from vllm import LLM
from vllm.sampling_params import SamplingParams
from vllm.lora.request import LoRARequest


class VLLMPredictor:
    """Simple predictor using vLLM for efficient inference."""

    def __init__(
        self,
        model_path: str,
        lora_adapter_path: Optional[str] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
    ):
        """
        Initialize the vLLM predictor.

        Args:
            model_path: HuggingFace model path or local path to base model
            lora_adapter_path: Optional path to LoRA adapter
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use
            max_model_len: Maximum sequence length (None for model default)
        """
        self.model_path = model_path
        self.lora_adapter_path = lora_adapter_path
        self.lora_request = None

        llm_kwargs = {
            "model": model_path,
            "tokenizer_mode": "mistral",
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "trust_remote_code": True,
        }

        if lora_adapter_path:
            llm_kwargs["enable_lora"] = True
            self.lora_request = LoRARequest("finetuned", 1, lora_adapter_path)

        self.llm = LLM(**llm_kwargs)

    def generate(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        use_lora: Optional[bool] = None,
    ) -> str:
        """
        Generate a response for a single message.

        Args:
            message: User message/prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            use_lora: Override LoRA usage. None=default behavior,
                      True=force LoRA, False=force base model only.

        Returns:
            Raw generated text
        """
        messages = self._build_messages(message, system_prompt)

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        chat_kwargs = {"sampling_params": sampling_params}
        lora = self._resolve_lora(use_lora)
        if lora:
            chat_kwargs["lora_request"] = lora

        outputs = self.llm.chat(messages, **chat_kwargs)
        return outputs[0].outputs[0].text

    def generate_batch(
        self,
        messages: List[str],
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        use_lora: Optional[bool] = None,
    ) -> List[str]:
        """
        Generate responses for multiple messages.

        Args:
            messages: List of user messages
            system_prompt: Optional system prompt (applied to all)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            use_lora: Override LoRA usage. None=default behavior,
                      True=force LoRA, False=force base model only.

        Returns:
            List of raw generated texts
        """
        conversations = [self._build_messages(msg, system_prompt) for msg in messages]

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        chat_kwargs = {"sampling_params": sampling_params}
        lora = self._resolve_lora(use_lora)
        if lora:
            chat_kwargs["lora_request"] = lora

        outputs = self.llm.chat(conversations, **chat_kwargs)
        return [output.outputs[0].text for output in outputs]

    def _resolve_lora(self, use_lora: Optional[bool] = None) -> Optional[LoRARequest]:
        """Resolve whether to use LoRA for this request.

        Args:
            use_lora: None=default (use if configured), True=force LoRA,
                      False=force base model only.

        Returns:
            LoRARequest if LoRA should be used, None otherwise.
        """
        if use_lora is None:
            return self.lora_request
        if use_lora:
            return self.lora_request
        return None

    def _build_messages(
        self, message: str, system_prompt: Optional[str] = None
    ) -> List[dict]:
        """Build message list for chat API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})
        return messages
