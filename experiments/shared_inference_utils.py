"""
Shared utilities for inference-based experiments (1, 2, 3).

Handles:
  - Loading and formatting stage0 data
  - Communicating with VLLMPredictor
  - Parsing model conclusions
  - Perturbation helpers
"""

import json
import random
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a logical reasoning assistant. "
    "Given the following premises, derive their valid conclusion."
)

BASE_MODEL = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
STAGE0_ADAPTER = str(
    PROJECT_ROOT / "mistralai_Mistral_Small_3.2_24B_Instruct_2506_20260302_175851" / "final"
)
STAGE1_ADAPTER = str(
    PROJECT_ROOT / "mistralai_Mistral_Small_3.2_24B_Instruct_2506_20260302_223714" / "final"
)
STAGE0_DATA = str(
    PROJECT_ROOT / "chain_stage0_n10000_len2-20_comp1_20260302_172126.jsonl"
)

HF_OVERRIDES = '{"architectures":["MistralForCausalLM"]}'

# Logical keywords NOT to perturb
LOGIC_KEYWORDS = {
    "if", "then", "and", "or", "not", "the", "it", "is", "case", "that",
    "for", "all", "there", "exist", "such", "a", "an", "only", "are",
    "does", "do", "has", "have", "with", "at", "by", "be", "been", "was",
    "were", "will", "would", "could", "should", "may", "might", "can",
    "no", "yes", "in", "on", "of", "to", "from"
}

# Replacement pool for perturbing entities/predicates
REPLACEMENT_POOL = [
    "the mountain", "the river", "the window", "the table", "the cloud",
    "the engine", "the pencil", "the mirror", "the bottle", "the lamp",
    "the garden", "the ceiling", "the pillow", "the socket", "the folder",
    "the ribbon", "the anchor", "the barrel", "the candle", "the bracket",
]
REPLACEMENT_ADJECTIVES = [
    "frozen", "hollow", "rusty", "golden", "silent", "broken", "ancient",
    "fragile", "polished", "twisted", "narrow", "dense", "smooth", "faded",
    "crisp", "humid", "rigid", "curved", "layered", "sharp",
]
REDUNDANT_PREMISES = [
    "the sky is blue",
    "water is wet",
    "the sun rises in the east",
    "two plus two equals four",
    "fire produces heat",
    "birds have feathers",
    "trees grow upward",
    "ice melts at high temperatures",
    "the moon orbits the earth",
    "plants require sunlight",
]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_stage0_data(n: int = 100, seed: int = 42, min_premises: int = 1,
                     data_path: str = STAGE0_DATA) -> List[Dict]:
    """Load n random stage0 examples."""
    random.seed(seed)
    data = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                ex = json.loads(line)
                if len(ex.get("premises", [])) >= min_premises:
                    data.append(ex)
    selected = random.sample(data, min(n, len(data)))
    print(f"Loaded {len(selected)} examples (pool={len(data)}, min_premises={min_premises})")
    return selected


def extract_ground_truth_conclusion(example: Dict) -> str:
    """Extract the ground-truth conclusion text from a stage0 example."""
    content = example.get("content", "")
    m = re.search(r"<CONCLUSION>\s*(.*?)\s*</CONCLUSION>", content, re.DOTALL)
    if m:
        return m.group(1).strip()
    return content.strip()


def format_user_message(premises: List[str]) -> str:
    """Format premise list into the training-compatible user message."""
    parts = [f"<PREMISE> {p.strip()} </PREMISE>" for p in premises if p.strip()]
    return " ".join(parts)


def parse_model_conclusion(response: str) -> str:
    """Extract conclusion text from model response."""
    m = re.search(r"<CONCLUSION>\s*(.*?)\s*</CONCLUSION>", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: return the whole response stripped
    return response.strip()


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    # Remove surrounding curly braces if present
    text = text.strip("{}")
    return text


def conclusions_match(pred: str, gold: str, threshold: float = 0.85) -> bool:
    """Check if predicted and gold conclusions match (exact or word-overlap)."""
    pred_n = normalize_text(pred)
    gold_n = normalize_text(gold)
    if pred_n == gold_n:
        return True
    # Word-overlap F1
    pred_words = set(pred_n.split())
    gold_words = set(gold_n.split())
    if not pred_words or not gold_words:
        return False
    overlap = len(pred_words & gold_words)
    precision = overlap / len(pred_words)
    recall = overlap / len(gold_words)
    if precision + recall == 0:
        return False
    f1 = 2 * precision * recall / (precision + recall)
    return f1 >= threshold


# ---------------------------------------------------------------------------
# Perturbation helpers
# ---------------------------------------------------------------------------

def _tokenize_for_perturbation(text: str) -> List[Tuple[int, int, str]]:
    """
    Return list of (start, end, word) for content words in text,
    excluding logical keywords, punctuation, and words inside braces.
    """
    tokens = []
    # Find word boundaries
    for m in re.finditer(r"\b([a-z][a-z0-9]*(?:\'[a-z]+)?)\b", text.lower()):
        word = m.group(1)
        if word not in LOGIC_KEYWORDS and len(word) > 2:
            tokens.append((m.start(), m.end(), m.group()))
    return tokens


def perturb_premise(premise: str, rng: random.Random) -> Optional[str]:
    """
    Perturb a single premise by substituting one content word.
    Returns the perturbed premise, or None if no suitable word found.
    """
    # Try to find entity names (multi-word capitalized phrases first)
    cap_matches = list(re.finditer(
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", premise
    ))
    if cap_matches:
        m = rng.choice(cap_matches)
        original = m.group(0)
        # Replace with a made-up name
        replacement_names = [
            "Samuel Chen", "Maria Santos", "Yuki Tanaka", "Omar Hassan",
            "Priya Patel", "Erik Larsen", "Fatima Al-Rashid", "Lucas Ferreira",
        ]
        replacement = rng.choice(replacement_names)
        return premise.replace(original, replacement, 1)

    # Fallback: find content nouns/predicates (after "is", "are", "not that")
    # Look for predicate patterns: "is <adj/noun>"
    pred_matches = list(re.finditer(
        r"\bis\s+((?:not\s+)?[a-z][a-z\s]+?)(?=\s*(?:\}|$|,))", premise
    ))
    if pred_matches:
        m = rng.choice(pred_matches)
        original = m.group(1).strip()
        if original.lower() in LOGIC_KEYWORDS or len(original.split()) > 4:
            pass  # too complex, fall through
        else:
            replacement = rng.choice(REPLACEMENT_ADJECTIVES)
            return premise.replace(original, replacement, 1)

    # Final fallback: replace a content word
    content_tokens = _tokenize_for_perturbation(premise)
    if not content_tokens:
        return None
    start, end, word = rng.choice(content_tokens)
    replacement = rng.choice(REPLACEMENT_ADJECTIVES)
    return premise[:start] + replacement + premise[end:]


def perturb_one_premise(premises: List[str], rng: random.Random) -> Tuple[List[str], int, str, str]:
    """
    Pick one premise at random and perturb it.
    Returns (perturbed_premises, perturbed_idx, original_text, new_text).
    """
    # Prefer longer premises for perturbation (more words to modify)
    indices = list(range(len(premises)))
    indices.sort(key=lambda i: len(premises[i].split()), reverse=True)
    # Pick among top half
    top_half = indices[:max(1, len(indices) // 2)]
    idx = rng.choice(top_half)
    original = premises[idx]
    perturbed = perturb_premise(original, rng)
    if perturbed is None or perturbed == original:
        # Last resort: append a modifier
        perturbed = original + " and " + rng.choice(REDUNDANT_PREMISES)
    new_premises = list(premises)
    new_premises[idx] = perturbed
    return new_premises, idx, original, perturbed


# ---------------------------------------------------------------------------
# VLLMPredictor factory
# ---------------------------------------------------------------------------

def make_predictor(lora_adapter: str = STAGE0_ADAPTER,
                   max_model_len: int = 4096,
                   gpu_memory_utilization: float = 0.75):
    """
    Create a predictor using the transformers backend (more reliable than vLLM
    for Mistral multimodal model variants).
    """
    from experiments.shared_transformers_inference import TransformersPredictor
    predictor = TransformersPredictor(
        model_name=BASE_MODEL,
        adapter_path=lora_adapter,
        use_4bit=True,
        max_new_tokens=256,
        temperature=0.0,
    )
    return predictor


def run_inference_batch(predictor, messages: List[str],
                        max_tokens: int = 256,
                        temperature: float = 0.0) -> List[str]:
    """Run batch inference on a list of user messages."""
    return predictor.generate_batch(
        messages=messages,
        system_prompt=SYSTEM_PROMPT,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
    )
