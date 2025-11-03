# Quick Start Guide

This guide walks you through setting up and running SyLLM from scratch.

## Prerequisites

1. **Python 3.8+** installed
2. **CUDA-capable GPU** (recommended, but CPU inference is possible)
3. **Access to DeepSeek V3 model** (ensure you have proper licensing)
4. **~20GB disk space** for models and data

## Step 1: Installation

```bash
# Clone or navigate to the project directory
cd SyLLM

# Install dependencies
pip install -r requirements.txt

# (Optional) Install in development mode
pip install -e .
```

## Step 2: Configure

### Model Selection (Easy Way)

The easiest way to switch models is using the `.env` file:

```bash
# Copy the example .env file
cp .env.example .env

# Switch to your desired model
python scripts/switch_model.py deepseek-ai/deepseek-v3

# Or manually edit .env and set:
# MODEL_NAME=deepseek-ai/deepseek-v3
```

### Full Configuration

Edit `config.yaml` to customize other settings:

- `model.base_model`: Path to model (or set via MODEL_NAME in .env)
- `data.*`: Paths for data files (they'll be created)
- `training.output_dir`: Where to save fine-tuned models
- `verifier.*`: Verifier model paths (can use rule-based fallback initially)

## Step 3: Prepare Data

### Option A: Use Synthetic Generation

```bash
# Generate synthetic training data
python scripts/generate_data.py

# Prepare train/val/test splits
python scripts/prepare_data.py
```

### Option B: Use Your Own Data

1. Format your data as JSONL following the schema in `src/data/schema.py`
2. Place it in `data/raw.jsonl`
3. Run: `python scripts/prepare_data.py`

## Step 4: Fine-tune

```bash
# Fine-tune with LoRA
python scripts/train_finetune.py
```

This will:
- Load DeepSeek V3 base model
- Apply LoRA adapters
- Train on your structured data
- Save model to `models/finetuned/final/`

Monitor training logs. Training time depends on:
- Dataset size
- GPU capability
- Model size

## Step 5: (Optional) RLHF

After fine-tuning, further improve with RLHF:

```bash
python scripts/train_rlhf.py
```

This uses the verifier as a reward model to optimize for verified outputs.

## Step 6: Inference

Test your model:

```bash
python scripts/inference_demo.py
```

This will:
- Load your fine-tuned model
- Run example queries
- Show parsed outputs and verification results

## Step 7: Evaluate

Evaluate on test set:

```bash
# First, generate outputs on test set
# (You'll need to create a script that runs inference on test.jsonl)

# Then evaluate
python scripts/evaluate.py \
    --model_outputs outputs/test_outputs.jsonl \
    --ground_truth data/test.jsonl
```

## Advanced: Building Retrieval Index

If you want to use RAG for evidence retrieval:

```python
from src.retrieval.retriever import DocumentRetriever, RetrievalConfig

# Prepare your documents
documents = {
    "doc1": "Your document text here...",
    "doc2": "Another document...",
}

# Build index
config = RetrievalConfig()
retriever = DocumentRetriever(config)
retriever.index_documents(documents)
retriever.save_index()
```

Then update `config.yaml` with retrieval settings.

## Troubleshooting

### Out of Memory
- Reduce `per_device_train_batch_size` in `config.yaml`
- Increase `gradient_accumulation_steps`
- Use smaller LoRA rank

### Model Not Found
- Verify `model.base_model` path in `config.yaml`
- Ensure model files are accessible

### Parse Errors
- Check that training data follows correct JSON schema
- Review `src/data/schema.py` for expected format

### Verifier Not Working
- Verifier uses rule-based fallback if models aren't found
- Train verifier models on NLI data for better performance

## Next Steps

- Add domain-specific documents for retrieval
- Train verifier models on your domain
- Fine-tune hyperparameters for your use case
- Set up monitoring and logging for production

## Getting Help

- Check `README.md` for detailed documentation
- Review code comments in `src/` modules
- Open an issue on the repository

