"""Answer parsing and extraction from model-generated text."""

import re
from typing import Optional, Tuple


def parse_multiple_choice_answer(
    text: str,
    num_choices: int = 4,
) -> Tuple[int, float]:
    """Parse a multiple-choice answer from generated text.

    Returns (answer_index, confidence) where confidence indicates parsing
    reliability (1.0 = clearly parsed, lower = ambiguous).

    Strategy (in priority order):
    1. "The answer is (X)" or "Answer: X" patterns
    2. JSON-style {"answer": "X"}
    3. Standalone letter at start or end of response
    4. First valid letter in the response
    5. Fallback: 0 with 0.0 confidence
    """
    text = text.strip()
    max_letter = chr(ord("A") + num_choices - 1)
    pattern_range = f"A-{max_letter}"

    # 1. Explicit answer patterns
    explicit_patterns = [
        rf"[Tt]he\s+(?:correct\s+)?answer\s+is\s*[:\s]*\(?([{pattern_range}])\)?",
        rf"[Aa]nswer\s*[:\s]+\(?([{pattern_range}])\)?",
        rf"[Cc]orrect\s+(?:answer|option)\s*[:\s]+\(?([{pattern_range}])\)?",
        rf"\*\*([{pattern_range}])\*\*",
    ]
    for pattern in explicit_patterns:
        match = re.search(pattern, text)
        if match:
            letter = match.group(1).upper()
            return ord(letter) - ord("A"), 1.0

    # 2. JSON-style answer
    json_match = re.search(r'"answer"\s*:\s*"([A-Z])"', text, re.IGNORECASE)
    if json_match:
        letter = json_match.group(1).upper()
        if letter <= max_letter:
            return ord(letter) - ord("A"), 0.9

    # 3. Parenthesized letter
    paren_match = re.search(rf"\(([{pattern_range}])\)", text)
    if paren_match:
        letter = paren_match.group(1).upper()
        return ord(letter) - ord("A"), 0.8

    # 4. Letter at start or end
    start_match = re.match(rf"^([{pattern_range}])[\.\s\)]", text)
    if start_match:
        return ord(start_match.group(1).upper()) - ord("A"), 0.7

    end_match = re.search(rf"([{pattern_range}])\.?\s*$", text)
    if end_match:
        return ord(end_match.group(1).upper()) - ord("A"), 0.6

    # 5. First valid letter anywhere
    any_match = re.search(rf"\b([{pattern_range}])\b", text)
    if any_match:
        return ord(any_match.group(1).upper()) - ord("A"), 0.3

    # 6. Fallback
    return 0, 0.0


def parse_gsm8k_answer(text: str) -> Optional[float]:
    """Extract numerical answer from GSM8K-style generation.

    Looks for "#### <number>" pattern first, then falls back to the
    last number in the response.

    Returns:
        Extracted number, or None if no number found.
    """
    text = text.strip()

    # Look for #### marker
    hash_match = re.search(r"####\s*([\-]?\d[\d,]*\.?\d*)", text)
    if hash_match:
        num_str = hash_match.group(1).replace(",", "")
        try:
            return float(num_str)
        except ValueError:
            pass

    # Fallback: last number in text
    numbers = re.findall(r"[\-]?\d[\d,]*\.?\d*", text)
    if numbers:
        num_str = numbers[-1].replace(",", "")
        try:
            return float(num_str)
        except ValueError:
            pass

    return None


def parse_true_false_unknown(text: str) -> int:
    """Parse True/False/Unknown for entailment tasks.

    Returns:
        0 for True, 1 for False, 2 for Unknown.
    """
    text_lower = text.strip().lower()

    # Check for explicit labels
    true_patterns = [r"\btrue\b", r"\bentailed\b", r"\byes\b", r"\bvalid\b"]
    false_patterns = [r"\bfalse\b", r"\bnot entailed\b", r"\bno\b", r"\binvalid\b"]
    unknown_patterns = [
        r"\bunknown\b",
        r"\buncertain\b",
        r"\bcannot be determined\b",
        r"\bundetermined\b",
    ]

    # Check "Answer: X" pattern first
    answer_match = re.search(r"answer\s*[:\s]+(\w+)", text_lower)
    if answer_match:
        word = answer_match.group(1)
        if word in ("true", "yes", "valid", "entailed"):
            return 0
        if word in ("false", "no", "invalid"):
            return 1
        if word in ("unknown", "uncertain", "undetermined"):
            return 2

    # Check patterns against full text (last occurrence wins for conflicts)
    for pat in unknown_patterns:
        if re.search(pat, text_lower):
            return 2
    for pat in false_patterns:
        if re.search(pat, text_lower):
            return 1
    for pat in true_patterns:
        if re.search(pat, text_lower):
            return 0

    return 2  # Default to unknown


def normalize_answer(text: str) -> str:
    """Normalize answer text for comparison."""
    text = text.lower().strip()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text
