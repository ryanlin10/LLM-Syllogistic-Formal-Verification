"""
Transformers-based inference utility for experiments 1-3.
Loads the model once and provides batch generation, using the same approach as the training pipeline.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SYSTEM_PROMPT = (
    "You are a logical reasoning assistant. "
    "Given the following premises, derive their valid conclusion."
)


def build_messages(user_message: str, system_prompt: str = SYSTEM_PROMPT) -> list:
    """Build messages list for apply_chat_template."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]


class TransformersPredictor:
    def __init__(
        self,
        model_name: str,
        adapter_path: Optional[str] = None,
        use_4bit: bool = True,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        print(f"Loading model: {model_name} (4-bit={use_4bit})")
        load_kwargs = dict(
            device_map="auto",
            trust_remote_code=True,
        )
        if use_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16

        # Mistral Small 3.2 uses Mistral3ForConditionalGeneration (multimodal base).
        # Try this class first; fall back to AutoModelForCausalLM.
        try:
            from transformers import Mistral3ForConditionalGeneration
            # Use Flash Attention 2 for faster inference
            fa2_kwargs = dict(**load_kwargs)
            try:
                fa2_kwargs["attn_implementation"] = "flash_attention_2"
                self.model = Mistral3ForConditionalGeneration.from_pretrained(
                    model_name, **fa2_kwargs
                )
                print("  Loaded as Mistral3ForConditionalGeneration (Flash Attention 2)")
            except Exception:
                self.model = Mistral3ForConditionalGeneration.from_pretrained(
                    model_name, **load_kwargs
                )
                print("  Loaded as Mistral3ForConditionalGeneration (SDPA)")
        except (ImportError, AttributeError, Exception) as e:
            print(f"  Mistral3 class failed ({e}), falling back to AutoModelForCausalLM")
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

        if adapter_path is not None:
            print(f"Applying LoRA adapter: {adapter_path}")
            self.model = PeftModel.from_pretrained(
                self.model, adapter_path, is_trainable=False
            )

        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = next(self.model.parameters()).device

        print(f"Model ready on device: {self.device}")

    def generate_batch(
        self,
        messages: List[str],
        system_prompt: str = SYSTEM_PROMPT,
        max_tokens: int = None,
        temperature: float = None,
        top_p: float = 0.95,
        batch_size: int = 8,
    ) -> List[str]:
        if max_tokens is None:
            max_tokens = self.max_new_tokens
        if temperature is None:
            temperature = self.temperature

        all_responses = []

        for i in range(0, len(messages), batch_size):
            batch_msgs = messages[i:i+batch_size]

            # Use tokenize=True directly for correct special token handling
            # (MistralCommonTokenizer warns that tokenize=False is unsafe)
            all_input_ids = []
            for msg in batch_msgs:
                chat = build_messages(msg, system_prompt)
                try:
                    # tokenize=True returns a list of token IDs
                    ids = self.tokenizer.apply_chat_template(
                        chat, tokenize=True, return_tensors=None
                    )
                    if isinstance(ids, list) and isinstance(ids[0], list):
                        ids = ids[0]
                    all_input_ids.append(ids)
                except Exception as e:
                    # Fallback: encode the prompt manually
                    prompt = (
                        f"[SYSTEM_PROMPT]{system_prompt}[/SYSTEM_PROMPT]"
                        f"[INST]{msg}[/INST]"
                    )
                    ids = self.tokenizer.encode(prompt, add_special_tokens=True)
                    all_input_ids.append(ids)

            # Pad to same length (left-pad for left-padding tokenizers)
            max_len = min(max(len(ids) for ids in all_input_ids), 2048)
            pad_id = self.tokenizer.pad_token_id or 0
            padded = []
            attn_masks = []
            for ids in all_input_ids:
                ids = ids[-max_len:]  # truncate from left if needed
                pad_len = max_len - len(ids)
                padded.append([pad_id] * pad_len + ids)
                attn_masks.append([0] * pad_len + [1] * len(ids))

            import torch
            input_ids = torch.tensor(padded, dtype=torch.long).to(self.device)
            attention_mask = torch.tensor(attn_masks, dtype=torch.long).to(self.device)
            prompt_len = input_ids.shape[1]
            if i == 0:
                first_decoded = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
                print(f"  [debug] prompt tokens={prompt_len}, prompt: {first_decoded[:120]!r}")

            gen_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                do_sample=(temperature > 0),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p

            with torch.no_grad():
                out = self.model.generate(**gen_kwargs)

            # Decode only the newly generated tokens
            new_tokens = out[:, prompt_len:]
            for seq in new_tokens:
                text = self.tokenizer.decode(seq, skip_special_tokens=True)
                all_responses.append(text.strip())

            print(f"  [{min(i+batch_size, len(messages))}/{len(messages)}] done")

        return all_responses
