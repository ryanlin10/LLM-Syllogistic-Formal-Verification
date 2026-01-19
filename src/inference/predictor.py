"""Inference using vLLM for efficient text generation."""

from typing import Optional, List
from vllm import LLM
from vllm.sampling_params import SamplingParams


class VLLMPredictor:
    """Simple predictor using vLLM for efficient inference."""

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
    ):
        """
        Initialize the vLLM predictor.

        Args:
            model_path: HuggingFace model path or local path
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use
            max_model_len: Maximum sequence length (None for model default)
        """
        self.model_path = model_path
        self.llm = LLM(
            model=model_path,
            tokenizer_mode="mistral",
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
        )

    def generate(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate a response for a single message.

        Args:
            message: User message/prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter

        Returns:
            Raw generated text
        """
        messages = self._build_messages(message, system_prompt)

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        outputs = self.llm.chat(messages, sampling_params=sampling_params)
        return outputs[0].outputs[0].text

    def generate_batch(
        self,
        messages: List[str],
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> List[str]:
        """
        Generate responses for multiple messages.

        Args:
            messages: List of user messages
            system_prompt: Optional system prompt (applied to all)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter

        Returns:
            List of raw generated texts
        """
        conversations = [self._build_messages(msg, system_prompt) for msg in messages]

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        outputs = self.llm.chat(conversations, sampling_params=sampling_params)
        return [output.outputs[0].text for output in outputs]

    def _build_messages(
        self, message: str, system_prompt: Optional[str] = None
    ) -> List[dict]:
        """Build message list for chat API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})
        return messages
