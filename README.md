# SyLLM: Structured Reasoning LLM with Automated Verification

A comprehensive system for fine-tuning LLMs (DeepSeek V3) to output structured reasoning in premise-conclusion format with automated verification for commercial use.

## Overview

SyLLM enables LLMs to:
- Output structured reasoning chains with explicit premises and conclusions
- Automatically verify premise factuality and logical inference
- Ground claims in retrieved evidence
- Improve reasoning quality through RLHF with verifier-based rewards

## Features

- **Structured Output**: Enforces JSON format with premises and conclusions
- **Automated Verification**: Two-stage verifier (premise + inference checking)
- **RAG Integration**: Retrieval-augmented generation for evidence grounding
- **Data Generation**: Synthetic data generation pipeline with adversarial examples
- **Fine-tuning**: LoRA-based efficient fine-tuning for DeepSeek V3
- **RLHF**: Reinforcement learning using verifier as reward model
- **Evaluation**: Comprehensive metrics for premise accuracy, entailment, and verifier calibration

## Project Structure

```
SyLLM/
├── config.yaml              # Main configuration file
├── requirements.txt          # Python dependencies
├── src/
│   ├── data/                # Data generation and curation
│   │   ├── schema.py        # Data schema definitions
│   │   ├── generator.py    # Synthetic data generation
│   │   └── curation.py     # Data validation and splitting
│   ├── retrieval/          # RAG components
│   │   └── retriever.py    # Document retrieval with FAISS
│   ├── verification/        # Automated verification
│   │   └── verifier.py     # Premise and inference verifiers
│   ├── training/           # Training scripts
│   │   ├── finetune.py     # Fine-tuning with LoRA
│   │   └── rlhf.py         # RLHF training
│   ├── evaluation/         # Evaluation framework
│   │   └── evaluator.py   # Metrics and evaluation
│   └── inference/         # Inference pipeline
│       └── predictor.py   # Model prediction
├── scripts/               # Executable scripts
│   ├── generate_data.py  # Generate synthetic data
│   ├── prepare_data.py   # Prepare and split data
│   ├── train_finetune.py # Run fine-tuning
│   ├── train_rlhf.py     # Run RLHF training
│   ├── evaluate.py       # Evaluate model
│   └── inference_demo.py # Demo inference
└── data/                  # Data directory (created at runtime)
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd SyLLM
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure the system:
   - Edit `config.yaml` to set model paths, data paths, and hyperparameters
   - Ensure you have access to DeepSeek V3 model files

## Quick Start

### 1. Generate Training Data

```bash
python scripts/generate_data.py
```

This generates synthetic premise-conclusion pairs from documents.

### 2. Prepare Data Splits

```bash
python scripts/prepare_data.py
```

This validates, balances, and splits data into train/val/test sets.

### 3. Fine-tune the Model

```bash
python scripts/train_finetune.py
```

This fine-tunes DeepSeek V3 with LoRA to output structured JSON.

### 4. (Optional) RLHF Training

```bash
python scripts/train_rlhf.py
```

This further improves the model using RLHF with verifier rewards.

### 5. Run Inference

```bash
python scripts/inference_demo.py
```

This demonstrates inference with verification.

### 6. Evaluate

```bash
python scripts/evaluate.py --model_outputs outputs.jsonl --ground_truth data/test.jsonl
```

## Configuration

Edit `config.yaml` to customize:

- **Model**: Base model path, LoRA settings
- **Training**: Batch sizes, learning rates, epochs
- **RLHF**: PPO parameters, reward scaling
- **Verifier**: Confidence thresholds, model paths
- **Retrieval**: Embedding model, top-k retrieval
- **Data**: Paths for train/val/test splits

## Data Format

Training data uses JSONL format:

```json
{
  "id": "uuid",
  "context": "Question and retrieved context",
  "premises": [
    {
      "id": "p1",
      "text": "Factual premise statement",
      "evidence_spans": [
        {"doc_id": "D1", "start": 120, "end": 220, "text": "..."}
      ]
    }
  ],
  "conclusion": {
    "text": "Conclusion that follows from premises",
    "type": "entailment|contradiction|unsupported"
  },
  "confidence": 0.9,
  "timestamp": "YYYY-MM-DD"
}
```

## Verification System

The verifier has two stages:

1. **Premise Verifier**: Checks if each premise is supported by evidence
   - Labels: `supported`, `contradicted`, `unverifiable`
   - Uses NLI-style classifier (with rule-based fallback)

2. **Inference Verifier**: Checks if conclusion follows from premises
   - Labels: `entailed`, `non-entailed`, `weakly_supported`
   - Uses multi-premise entailment model

Final verdict: `accept`, `review`, or `reject`

## Evaluation Metrics

- **Premise Precision/Recall**: Accuracy of generated premises
- **Evidence Recall**: Fraction of premises with linked evidence
- **Entailment Accuracy**: Correctness of conclusions
- **Verifier Calibration**: Expected Calibration Error (ECE)
- **Parse Success Rate**: Percentage of valid JSON outputs

## Advanced Usage

### Custom Data Sources

Add your domain documents to the data generation pipeline:

```python
from src.data.generator import SyntheticDataGenerator, GenerationConfig

generator = SyntheticDataGenerator(GenerationConfig(num_examples=1000))
annotations = generator.generate_batch(your_documents, your_questions)
```

### Building Retrieval Index

```python
from src.retrieval.retriever import DocumentRetriever, RetrievalConfig

retriever = DocumentRetriever(RetrievalConfig())
retriever.index_documents(your_documents)
retriever.save_index()
```

### Custom Verifier Models

Train verifier models on domain-specific NLI data, then update paths in `config.yaml`.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- ~20GB disk space for models and data

## License

Ensure compliance with DeepSeek V3 model license for commercial use.

## Contributing

1. Follow the existing code structure
2. Add tests for new features
3. Update documentation
4. Ensure code passes linting

## Roadmap

- [ ] Human-in-the-loop review UI
- [ ] Active learning pipeline
- [ ] Production deployment templates
- [ ] Multi-domain adaptation
- [ ] Enhanced verifier training data

## Support

For issues and questions, please open an issue on the repository.

