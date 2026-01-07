# Model Switching Guide

This guide explains how to easily switch between different LLM models in the SyLLM pipeline.

## Quick Start

The easiest way to switch models is using the `.env` file:

```bash
# 1. Copy the example .env file
cp .env.example .env

# 2. Switch to your desired model
python scripts/switch_model.py deepseek-ai/deepseek-v3

# 3. Verify the change
python scripts/switch_model.py --current
```

That's it! All scripts will now use the model specified in `.env`.

## Available Models

List all recommended models:

```bash
python scripts/switch_model.py --list
```

Common models include:
- `deepseek-ai/deepseek-v3` (recommended)
- `deepseek-ai/deepseek-v2`
- `meta-llama/Llama-2-7b-chat-hf`
- `meta-llama/Llama-2-13b-chat-hf`
- `mistralai/Mistral-7B-Instruct-v0.2`
- `mistralai/Mixtral-8x7B-Instruct-v0.1`

## How It Works

### Priority Order

1. **Environment variables** (`.env` file or system env) - **Highest Priority**
2. **config.yaml** file - Fallback

The `MODEL_NAME` environment variable overrides `config.yaml`, allowing you to switch models without editing configuration files.

### Example .env File

```bash
# .env
MODEL_NAME=deepseek-ai/deepseek-v3
```

### Manual Editing

You can also manually edit `.env`:

```bash
# Edit .env file
nano .env

# Set MODEL_NAME to your desired model
MODEL_NAME=meta-llama/Llama-2-7b-chat-hf
```

## Integration

All components automatically use the model from `.env`:

- ✅ Data generation (`scripts/generate_data.py`)
- ✅ Fine-tuning (`scripts/train_finetune.py`)
- ✅ RLHF training (`scripts/train_rlhf.py`)
- ✅ Inference (`scripts/inference_demo.py`)
- ✅ Evaluation (`scripts/evaluate.py`)

## Additional Environment Variables

You can also override other settings:

```bash
# .env
MODEL_NAME=deepseek-ai/deepseek-v3
PREMISE_VERIFIER_MODEL=./models/verifier/premise
INFERENCE_VERIFIER_MODEL=./models/verifier/inference
RETRIEVAL_EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
DATA_DIR=./data
MODELS_DIR=./models
OUTPUT_DIR=./models/finetuned
```

## Programmatic Usage

Use the config loader in your code:

```python
from src.utils.config_loader import load_config, get_model_name, set_model_in_env

# Load config (automatically reads .env)
config = load_config()

# Get current model
model_name = get_model_name(config)
print(f"Using model: {model_name}")

# Switch model programmatically
set_model_in_env("meta-llama/Llama-2-7b-chat-hf")
```

## Troubleshooting

### Model Not Switching

1. Check if `.env` exists: `ls -la .env`
2. Verify MODEL_NAME is set: `cat .env | grep MODEL_NAME`
3. Check current model: `python scripts/switch_model.py --current`

### Model Not Found

Ensure you have:
- Access to the model (HuggingFace credentials, etc.)
- Correct model identifier
- Internet connection for downloading models

### Configuration Conflicts

If `.env` and `config.yaml` conflict, `.env` takes priority. To use `config.yaml`:
1. Remove or comment out `MODEL_NAME` in `.env`
2. Set `model.base_model` in `config.yaml`

## Best Practices

1. **Use .env for model switching** - Keeps config.yaml stable
2. **Version control .env.example** - Share template, not secrets
3. **Document custom models** - Add to `list_available_models()` if frequently used
4. **Test after switching** - Verify model loads correctly before training

## Example Workflow

```bash
# Switch to Llama 2
python scripts/switch_model.py meta-llama/Llama-2-7b-chat-hf

# Verify
python scripts/switch_model.py --current

# Generate data with new model
python scripts/generate_data.py

# Train
python scripts/train_finetune.py

# Switch back to DeepSeek
python scripts/switch_model.py deepseek-ai/deepseek-v3

# Continue with DeepSeek
python scripts/train_finetune.py
```


