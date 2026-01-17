#!/usr/bin/env python3
"""Inspect tokenizer special tokens, chat templates, and formatting behavior."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer
from mistral_common.protocol.instruct.messages import (
    UserMessage,
    AssistantMessage,
    SystemMessage,
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

from src.utils.config_loader import load_config


def main():
    config_path = Path(__file__).parent.parent / "config.yaml"
    config = load_config(str(config_path))
    model_name = config["model"]["base_model"]

    print(f"Loading tokenizers for: {model_name}\n")

    # HuggingFace tokenizer
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Mistral tokenizer for chat formatting
    mistral_tokenizer = MistralTokenizer.from_hf_hub(model_name)

    # === Special Tokens ===
    print("=" * 60)
    print("SPECIAL TOKENS")
    print("=" * 60)

    print("\n[Key Tokens]")
    print(f"  BOS token: {repr(hf_tokenizer.bos_token)} (id: {hf_tokenizer.bos_token_id})")
    print(f"  EOS token: {repr(hf_tokenizer.eos_token)} (id: {hf_tokenizer.eos_token_id})")
    print(f"  PAD token: {repr(hf_tokenizer.pad_token)} (id: {hf_tokenizer.pad_token_id})")
    print(f"  UNK token: {repr(hf_tokenizer.unk_token)} (id: {hf_tokenizer.unk_token_id})")

    # Get vocab and find special tokens (tokens with < > or [ ])
    print("\n[Special Tokens in Vocabulary]")
    vocab = hf_tokenizer.get_vocab()
    special_tokens = {k: v for k, v in vocab.items()
                      if k.startswith('<') or k.startswith('[') or k.startswith('▁[')}
    for token, idx in sorted(special_tokens.items(), key=lambda x: x[1])[:1200]:
        print(f"  {repr(token):30} -> id: {idx}")
    if len(special_tokens) > 1200:
        print(f"  ... and {len(special_tokens) - 1200} more")

    # === Mistral Chat Formatting ===
    print("\n" + "=" * 60)
    print("MISTRAL CHAT FORMATTING (mistral_common)")
    print("=" * 60)

    # Test case 1: Single user message
    print("\n[Single User Message]")
    print("-" * 40)
    request1 = ChatCompletionRequest(
        messages=[UserMessage(content="What is 2+2?")]
    )
    tokens1 = mistral_tokenizer.encode_chat_completion(request1)
    print(f"Text: {tokens1.text}")
    print(f"Token count: {len(tokens1.tokens)}")
    print(f"Token IDs (first 20): {tokens1.tokens[:20]}")

    # Test case 2: With system prompt
    print("\n[With System Prompt]")
    print("-" * 40)
    request2 = ChatCompletionRequest(
        messages=[
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content="What is 2+2?"),
        ]
    )
    tokens2 = mistral_tokenizer.encode_chat_completion(request2)
    print(f"Text: {tokens2.text}")
    print(f"Token count: {len(tokens2.tokens)}")

    # Test case 3: Multi-turn conversation
    print("\n[Multi-turn Conversation]")
    print("-" * 40)
    request3 = ChatCompletionRequest(
        messages=[
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content="What is 2+2?"),
            AssistantMessage(content="2+2 equals 4."),
            UserMessage(content="And 3+3?"),
        ]
    )
    tokens3 = mistral_tokenizer.encode_chat_completion(request3)
    print(f"Text: {tokens3.text}")
    print(f"Token count: {len(tokens3.tokens)}")

    # === HuggingFace apply_chat_template (if available) ===
    print("\n" + "=" * 60)
    print("HUGGINGFACE apply_chat_template")
    print("=" * 60)

    hf_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
    ]

    try:
        formatted = hf_tokenizer.apply_chat_template(
            hf_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        print("\n[With System Prompt]")
        print("-" * 40)
        print(formatted)
    except Exception as e:
        print(f"Error: {e}")

    # === Token Count Comparison ===
    print("\n" + "=" * 60)
    print("TOKENIZATION EXAMPLE")
    print("=" * 60)

    test_text = "Hello, how are you doing today?"
    tokens = hf_tokenizer.encode(test_text)
    decoded = hf_tokenizer.convert_ids_to_tokens(tokens)

    print(f"\nInput: {repr(test_text)}")
    print(f"Token count: {len(tokens)}")
    print(f"Token IDs: {tokens}")
    print(f"Tokens: {decoded}")

    # === Vocabulary Info ===
    print("\n" + "=" * 60)
    print("VOCABULARY INFO")
    print("=" * 60)
    print(f"  Vocab size: {hf_tokenizer.vocab_size}")
    print(f"  Model max length: {hf_tokenizer.model_max_length}")


if __name__ == "__main__":
    main()
