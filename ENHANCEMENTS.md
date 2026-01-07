# SyLLM Enhancements: DAG-Based Reasoning with Advanced Verification

## Overview

Significant enhancements have been added to SyLLM based on cutting-edge research:
- **Graph of Verification (GoV)**: DAG-based reasoning with topological verification
- **LLM-TRes**: Repair mechanism with axiom injection
- **Logic Datasets**: Integration with LogiQA, LogicNLI, LogiBench
- **Staged Verification**: NLI → Semantic Parse → Z3/Datalog solvers
- **Enhanced RLHF**: Stepwise verification rewards

## Key Components Added

### 1. DAG-Based Reasoning Schema (`src/data/dag_schema.py`)

**Features:**
- `InferenceStep`: Reasoning steps with explicit dependencies
- `DAGReasoning`: Complete DAG structure with premises → inference steps → conclusion
- Topological ordering for verification
- Support for repair axioms

**Structure:**
```json
{
  "premises": [{"id": "p1", "text": "..."}],
  "inference_steps": [
    {
      "id": "inf1",
      "text": "From p1 and p2, we can infer...",
      "depends_on": ["p1", "p2"],
      "formal_expression": "...",
      "formal_type": "z3"
    }
  ],
  "conclusion": "...",
  "repair_axioms": []
}
```

### 2. Logic Dataset Loaders (`src/data/logic_datasets.py`)

**Supported Datasets:**
- **LogiQA**: Legal reasoning questions
- **LogicNLI**: Natural language inference with logical patterns
- **LogiBench**: General logical reasoning benchmark

**Usage:**
```bash
# Load and convert logic datasets
python scripts/load_logic_datasets.py --datasets all --split train

# Datasets are automatically converted to DAG format
```

### 3. Semantic Parser (`src/verification/semantic_parser.py`)

**Autoformalization:**
- Converts natural language → formal logic (Datalog, Z3, FOL)
- Rule-based patterns for common structures
- Optional LLM-based parsing for complex cases
- Multiple target format support

**Supported Formats:**
- **Datalog**: `is(X, Y)` - Fast rule-based reasoning
- **Z3 SMT**: `x > 5` - Arithmetic and constraint solving
- **FOL**: First-order logic for complex reasoning

### 4. Staged Verification (`src/verification/dag_verifier.py`)

**Three-Stage Pipeline:**

1. **Lightweight NLI Check** (fast, approximate)
   - Uses existing NLI models
   - Confidence > 0.9 → accept immediately

2. **Semantic Parsing** (moderate speed)
   - Attempts formalization
   - Creates verifiable expressions

3. **Symbolic Solver** (precise, slower)
   - **Z3**: For arithmetic, numeric constraints
   - **Datalog**: For rule-based reasoning
   - Returns proof or counterexample

**Benefits:**
- Fast cases handled quickly (NLI)
- Complex cases verified precisely (Z3/Datalog)
- Graceful degradation if formalization fails

### 5. LLM-TRes Repair Mechanism (`src/verification/repair.py`)

**Repair Pipeline:**
1. Detect verification failure
2. Analyze failure trace
3. Propose repair axiom (LLM or rule-based)
4. Formalize axiom if possible
5. Apply repair and re-verify
6. Audit trail of all repairs

**Repair Axiom Structure:**
```python
RepairAxiom(
    axiom_text="If X and Y, then Z",  # Natural language
    formal_expression="forall x, y. X(x) ∧ Y(y) → Z(x,y)",  # Formal
    fixes_node_id="inf1",
    failure_trace="Missing link between p1 and conclusion",
    human_approved=False  # Requires approval in production
)
```

### 6. Enhanced RLHF (`src/training/enhanced_rlhf.py`)

**Reward Structure:**
- `+1.0` per verified inference step
- `+5.0` bonus for complete DAG verification
- `-0.5` penalty per repair required
- `-1.0` penalty for missing dependencies
- Brevity penalty for excessive nodes

**Benefits:**
- Encourages verifiable reasoning chains
- Penalizes incomplete or broken dependencies
- Rewards end-to-end verification success

## Usage Workflow

### 1. Load Logic Datasets

```bash
# Convert logic datasets to DAG format
python scripts/load_logic_datasets.py --datasets logiqa,logibench --split train

# Output: data/logic_datasets_train.jsonl
```

### 2. Prepare Combined Data

```bash
# Prepare data (includes logic datasets automatically if configured)
python scripts/prepare_data.py

# Combines:
# - Synthetic data
# - Logic datasets (LogiQA, etc.)
# - Your custom annotations
```

### 3. Fine-tune with DAG Structure

```bash
# Fine-tuning automatically uses DAG format if configured
python scripts/train_finetune.py

# Model learns to output:
# - Explicit inference steps
# - Dependency relationships
# - Formalizable expressions
```

### 4. Enhanced RLHF Training

```python
from src.training.enhanced_rlhf import EnhancedRewardModel

# Use enhanced rewards in RLHF
reward_model = EnhancedRewardModel(
    verifier_config=verifier_config,
    step_reward=1.0,
    complete_bonus=5.0
)

# Integrate into PPO training loop
```

### 5. Inference with DAG Verification

```python
from src.verification.dag_verifier import DAGVerifier
from src.verification.repair import RepairPipeline, RepairAgent

# Verify with staged approach
verifier = DAGVerifier(config)
result = verifier.verify_dag(reasoning)

# Auto-repair if needed
repair_agent = RepairAgent()
repair_pipeline = RepairPipeline(verifier, repair_agent)

repaired_reasoning, repairs, final_result = \
    repair_pipeline.repair_and_verify(reasoning)
```

## Configuration Updates

### config.yaml Changes

```yaml
# Enable DAG format
schema:
  use_dag_format: true
  max_inference_steps: 10

# Logic datasets
data:
  use_logic_datasets: true
  logic_datasets_path: "./data/logic_datasets_train.jsonl"

# Staged verification
verifier:
  use_staged_verification: true
  use_z3: true
  use_datalog: true
  enable_repair: true
  max_repairs: 3
  auto_approve_repairs: false  # Safety: require human approval
```

## Performance Improvements

### Expected Metrics (from research)

- **Step Verification Rate (SVR)**: 70% → 90%+
- **End-to-End Verification Rate**: 40% → 80%+
- **Repair Success Rate**: 30% → 70%+
- **False Acceptance Rate**: < 1-5% (for regulated domains)

### Benchmark Integration

Supports evaluation on:
- LogiQA
- LogicNLI
- LogiBench
- Custom domain datasets

## Research Citations

The enhancements are based on:

1. **Graph of Verification (GoV)**
   - DAG-based verification structure
   - Topological ordering
   - Multi-granular verification

2. **LLM-TRes**
   - Repair axiom generation
   - Theory resolution framework
   - Formal guarantees

3. **SyLeR**
   - Syllogistic reasoning templates
   - Major/minor premise structure
   - Legal domain applications

4. **Autoformalization Research**
   - NL → Formal logic translation
   - FM-ALPACA / FM-BENCH datasets
   - Multi-format support (Datalog, Z3, Lean)

## Next Steps

1. **Train semantic parser** on autoformalization datasets
2. **Fine-tune verifier models** on domain-specific NLI data
3. **Expand Z3 integration** for more constraint types
4. **Human-in-the-loop UI** for repair approval
5. **Production deployment** with monitoring and audit logs

## Migration from Legacy Format

The system supports both formats:

**Legacy:**
```json
{
  "premises": [...],
  "conclusion": "..."
}
```

**DAG Format:**
```json
{
  "premises": [...],
  "inference_steps": [...],
  "conclusion": "..."
}
```

Conversion happens automatically during training and inference.

## Troubleshooting

### Z3 Not Available
```bash
pip install z3-solver
```

### Logic Datasets Not Found
```bash
# Datasets are downloaded automatically from HuggingFace
# Or download manually and place in data/ directory
python scripts/load_logic_datasets.py --datasets logiqa
```

### Repair Not Working
- Check `verifier.enable_repair` in config.yaml
- Ensure verifier models are loaded
- Review failure traces in repair history

## Safety and Governance

- **Repair Approval**: Set `auto_approve_repairs: false` in production
- **Audit Trail**: All repairs logged with timestamps and failure traces
- **Human Review**: Failed verifications route to human reviewers
- **Provenance**: All premises linked to source documents


