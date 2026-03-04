#!/usr/bin/env python3
"""
Prepare GRPO training data from the FOLIO dataset.

Loads `tasksource/folio` (train + validation splits) from HuggingFace,
prepends a comprehensive preamble (instruction, format reference, inference
rules, and three worked demonstrations), and writes
`data/grpo_folio_prompted.jsonl` ready for `train_grpo.py`.

Usage
-----
    python scripts/prepare_grpo_data.py \\
        --output data/grpo_folio_prompted.jsonl

Optional overrides:
    --hf-dataset  tasksource/folio        (default)
    --splits      train validation        (default: both)
    --max-prompt-chars 8000               (warn threshold, default 8000)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("prepare_grpo_data")

# ---------------------------------------------------------------------------
# Demo examples (from data/stage2_train.jsonl, item_ids used for dedup)
# ---------------------------------------------------------------------------

DEMO_ITEM_IDS = {"folio_134", "folio_176", "folio_47"}

# Each demo is (question_text, reasoning_content, verdict)
# question_text must exactly match what format_question() produces so dedup works.
_DEMO_134_Q = (
    "Premises:\n"
    "Plungers suck.\n"
    "Vacuums suck.\n"
    "Vampires suck.\n"
    "Space is a vacuum.\n"
    "A duster is a household appliance that doesn't suck.\n"
    "\n"
    "Based on the premises, is the following conclusion true, false, or unknown?\n"
    "\n"
    "Conclusion: Space sucks."
)
_DEMO_134_C = (
    'We want to determine whether \u201cSpace sucks\u201d follows from the premises.\n'
    "\n"
    "First, use the premise that all vacuums suck, and the premise that space is a vacuum.\n"
    "\n"
    "<PREMISE> {for all x, {if x is a vacuum, then x sucks}} </PREMISE>\n"
    "<PREMISE> Space is a vacuum </PREMISE>\n"
    "<CONCLUSION> Space sucks </CONCLUSION>\n"
    "\n"
    "The statement about dusters (a household appliance that doesn't suck) does not apply "
    "to Space, and nothing else contradicts the derived conclusion."
)

_DEMO_176_Q = (
    "Premises:\n"
    "The summer Olympic games is a sporting event. \n"
    "The last summer Olympic games was in Tokyo.\n"
    "The United States won the most medals in Tokyo. \n"
    "\n"
    "Based on the premises, is the following conclusion true, false, or unknown?\n"
    "\n"
    "Conclusion: The last summer Olympic games were not in Tokyo."
)
_DEMO_176_C = (
    "We are told directly where the last summer Olympic games were held.\n"
    "\n"
    "<PREMISE> The last summer Olympic games was in Tokyo </PREMISE>\n"
    "<CONCLUSION> The last summer Olympic games were in Tokyo </CONCLUSION>\n"
    "\n"
    "The conclusion claims the negation of that statement.\n"
    "\n"
    "<PREMISE> The last summer Olympic games were in Tokyo </PREMISE>\n"
    "<CONCLUSION> {it is not the case that The last summer Olympic games were not in Tokyo} </CONCLUSION>\n"
    "\n"
    "Thus, the conclusion contradicts a premise and is false."
)

_DEMO_47_Q = (
    "Premises:\n"
    "There are four seasons in a year: Spring, Summer, Fall, and Winter.\n"
    "All students who want to have a long vacation have summer as their favorite season.\n"
    "Emma's favorite season is summer.\n"
    "Mia's favorite season is not the same as Emma's. \n"
    "James wants to have a long vacation.\n"
    "\n"
    "Based on the premises, is the following conclusion true, false, or unknown?\n"
    "\n"
    "Conclusion: Mia's favorite season is spring."
)
_DEMO_47_C = (
    "We formalize the relevant premises about favorites.\n"
    "\n"
    "<PREMISE> {for all x, {if x wants to have a long vacation, then x's favorite season is summer}} </PREMISE>\n"
    "<PREMISE> James wants to have a long vacation </PREMISE>\n"
    "<CONCLUSION> James's favorite season is summer </CONCLUSION>\n"
    "\n"
    "We are told Emma's favorite season, and that Mia's is different from Emma's.\n"
    "\n"
    "<PREMISE> Emma's favorite season is summer </PREMISE>\n"
    "<PREMISE> Mia's favorite season is not the same as Emma's </PREMISE>\n"
    "<CONCLUSION> {it is not the case that Mia's favorite season is summer} </CONCLUSION>\n"
    "\n"
    "We also know the only possible seasons are Spring, Summer, Fall, Winter, but knowing "
    "Mia is not Summer does not force Mia to be Spring; Mia could be Fall or Winter as well.\n"
    "\n"
    "Thus the conclusion cannot be derived, and it is not contradicted by the premises."
)

# Map question text → answer label (used to build the demos section of the preamble)
_DEMOS = [
    (_DEMO_134_Q, _DEMO_134_C, "True"),
    (_DEMO_176_Q, _DEMO_176_C, "False"),
    (_DEMO_47_Q,  _DEMO_47_C,  "Unknown"),
]

# Set of question texts to skip during dataset conversion (exact match).
_DEMO_QUESTION_TEXTS = {q.strip() for q, _, _ in _DEMOS}

# ---------------------------------------------------------------------------
# Preamble construction
# ---------------------------------------------------------------------------

_SECTION_1 = """\
You are a logical reasoning assistant. When given premises and a conclusion to evaluate,
reason step-by-step using the semi-formal proof format below. Every inference step must
be written using <PREMISE> and <CONCLUSION> tags. Compound formulas must be wrapped in
curly brackets {}. Work through the logic completely before stating your verdict
(True / False / Unknown).\
"""

_SECTION_2 = """\
## FORMAT

Tags:
  <PREMISE> ... </PREMISE>       — a fact used in this step
  <CONCLUSION> ... </CONCLUSION> — what is derived from the preceding premises
  <ASSUME> ... </ASSUME>         — open a temporary assumption block
  <DISCHARGE> ... </DISCHARGE>   — close the assumption block

Connective syntax (always wrap compound formulas in {}):
  {A and B}                          Conjunction
  {A or B}                           Disjunction
  {if A, then B}                     Implication
  {A if and only if B}               Biconditional
  {it is not the case that A}        Negation
  {for all x, P(x)}                  Universal quantification
  {there exist x such that P(x)}     Existential quantification\
"""

_SECTION_3 = """\
## INFERENCE RULES

AND_INTRO
  <PREMISE> A </PREMISE>
  <PREMISE> B </PREMISE>
  <CONCLUSION> {A and B} </CONCLUSION>

AND_ELIM
  <PREMISE> {A and B} </PREMISE>
  <CONCLUSION> A </CONCLUSION>          (or: conclude B)

OR_INTRO
  <PREMISE> A </PREMISE>
  <CONCLUSION> {A or B} </CONCLUSION>   (A alone suffices; B is arbitrary)

OR_ELIM  [discharge rule]
  <ASSUME> A </ASSUME>
    <PREMISE> ... </PREMISE>
    <CONCLUSION> G </CONCLUSION>
  <DISCHARGE> A </DISCHARGE>
  <ASSUME> B </ASSUME>
    <PREMISE> ... </PREMISE>
    <CONCLUSION> G </CONCLUSION>
  <DISCHARGE> B </DISCHARGE>
  <PREMISE> {A or B} </PREMISE>
  <CONCLUSION> G </CONCLUSION>

IMPLIES_ELIM  (Modus Ponens)
  <PREMISE> {if A, then G} </PREMISE>
  <PREMISE> A </PREMISE>
  <CONCLUSION> G </CONCLUSION>

IMPLIES_INTRO  [discharge rule]
  <ASSUME> A </ASSUME>
    <PREMISE> ... </PREMISE>
    <CONCLUSION> B </CONCLUSION>
  <DISCHARGE> A </DISCHARGE>
  <CONCLUSION> {if A, then B} </CONCLUSION>

IFF_INTRO
  <PREMISE> {if A, then B} </PREMISE>
  <PREMISE> {if B, then A} </PREMISE>
  <CONCLUSION> {A if and only if B} </CONCLUSION>

IFF_ELIM
  <PREMISE> {A if and only if B} </PREMISE>
  <PREMISE> A </PREMISE>
  <CONCLUSION> B </CONCLUSION>          (or: premise B -> conclude A)

NOT_INTRO  (reductio ad absurdum) [discharge rule]
  <ASSUME> A </ASSUME>
    <PREMISE> ... </PREMISE>
    <CONCLUSION> {B and {it is not the case that B}} </CONCLUSION>
  <DISCHARGE> A </DISCHARGE>
  <CONCLUSION> {it is not the case that A} </CONCLUSION>

NOT_ELIM  (double-negation elimination)
  <PREMISE> {it is not the case that {it is not the case that A}} </PREMISE>
  <CONCLUSION> A </CONCLUSION>

FORALL_INTRO  [FOL -- generalise from arbitrary constant c]
  <PREMISE> P(c) </PREMISE>
  <CONCLUSION> {for all x, P(x)} </CONCLUSION>

FORALL_ELIM  [FOL -- instantiate to specific constant]
  <PREMISE> {for all x, P(x)} </PREMISE>
  <CONCLUSION> P(c) </CONCLUSION>

EXISTS_INTRO  [FOL -- witness introduction]
  <PREMISE> P(c) </PREMISE>
  <CONCLUSION> {there exist x such that P(x)} </CONCLUSION>

EXISTS_ELIM  [FOL -- discharge rule]
  <ASSUME> P(c) </ASSUME>
    <PREMISE> ... </PREMISE>
    <CONCLUSION> G </CONCLUSION>
  <DISCHARGE> P(c) </DISCHARGE>
  <PREMISE> {there exist x such that P(x)} </PREMISE>
  <CONCLUSION> G </CONCLUSION>\
"""


def _build_examples_section() -> str:
    lines = ["## EXAMPLES"]
    for i, (question, content, verdict) in enumerate(_DEMOS, 1):
        lines.append(f"\n### Example {i} (Answer: {verdict})")
        lines.append(question)
        lines.append("\nReasoning:")
        lines.append(content)
        lines.append(f"Verdict: {verdict}")
        lines.append("\n---")
    return "\n".join(lines)


PREAMBLE = "\n\n".join([
    _SECTION_1,
    _SECTION_2,
    _SECTION_3,
    _build_examples_section(),
])

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

LABEL_NORM = {"Uncertain": "Unknown"}
VALID_LABELS = {"True", "False", "Unknown"}


def format_question(premises: str, conclusion: str) -> str:
    """Produce the standard FOLIO question block."""
    return (
        f"Premises:\n{premises}\n\n"
        "Based on the premises, is the following conclusion true, false, or unknown?\n\n"
        f"Conclusion: {conclusion}"
    )


def make_prompt(question: str) -> str:
    """Attach preamble + problem header to a question."""
    return (
        PREAMBLE
        + "\n\n## PROBLEM\n\n"
        + question
        + "\n\nReasoning:\n"
        + "(Use <PREMISE> and <CONCLUSION> tags for each inference step. "
        + "End your response with exactly one verdict line: "
        + "\"Verdict: True\", \"Verdict: False\", or \"Verdict: Unknown\".)\n\n"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output",
        default="data/grpo_folio_prompted.jsonl",
        help="Output JSONL path (default: data/grpo_folio_prompted.jsonl)",
    )
    p.add_argument(
        "--hf-dataset",
        default="tasksource/folio",
        help="HuggingFace dataset identifier (default: tasksource/folio)",
    )
    p.add_argument(
        "--fallback-dataset",
        default="yale-nlp/FOLIO",
        help="Fallback HF dataset if primary fails (default: yale-nlp/FOLIO)",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation"],
        help="Dataset splits to include (default: train validation)",
    )
    p.add_argument(
        "--max-prompt-chars",
        type=int,
        default=8000,
        help="Warn if any prompt exceeds this character count (default: 8000)",
    )
    return p.parse_args()


def load_folio(hf_dataset: str, fallback: str, splits: list[str]):
    """Load FOLIO splits from HuggingFace, trying fallback on failure."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error(
            "The `datasets` package is required. Install with: pip install datasets"
        )
        sys.exit(1)

    for dataset_id in [hf_dataset, fallback]:
        try:
            logger.info("Loading dataset '%s' splits: %s", dataset_id, splits)
            rows = []
            for split in splits:
                ds = load_dataset(dataset_id, split=split, trust_remote_code=False)
                rows.extend(ds)
                logger.info("  split '%s': %d rows", split, len(ds))
            return rows
        except Exception as exc:
            logger.warning("Failed to load '%s': %s", dataset_id, exc)

    logger.error("Could not load FOLIO from either '%s' or '%s'.", hf_dataset, fallback)
    sys.exit(1)


def main() -> None:
    args = parse_args()

    raw_rows = load_folio(args.hf_dataset, args.fallback_dataset, args.splits)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped_label = 0
    skipped_demo = 0
    long_prompts = 0
    label_counts: dict[str, int] = {}

    with output_path.open("w", encoding="utf-8") as fout:
        for row in raw_rows:
            # Normalise label
            raw_label = str(row.get("label", "")).strip()
            label = LABEL_NORM.get(raw_label, raw_label)
            if label not in VALID_LABELS:
                skipped_label += 1
                continue

            premises = str(row.get("premises", "")).strip()
            conclusion = str(row.get("conclusion", "")).strip()
            if not premises or not conclusion:
                skipped_label += 1
                continue

            question = format_question(premises, conclusion)

            # Skip demo examples (they appear only in the preamble)
            if question.strip() in _DEMO_QUESTION_TEXTS:
                skipped_demo += 1
                continue

            prompt = make_prompt(question)

            if len(prompt) > args.max_prompt_chars:
                long_prompts += 1
                logger.warning(
                    "Prompt exceeds %d chars (%d chars). premises=%r",
                    args.max_prompt_chars,
                    len(prompt),
                    premises[:80],
                )

            fout.write(json.dumps({"prompt": prompt, "target": label}) + "\n")
            written += 1
            label_counts[label] = label_counts.get(label, 0) + 1

    # Summary
    logger.info("")
    logger.info("=== Summary ===")
    logger.info("Total rows processed : %d", len(raw_rows))
    logger.info("Written              : %d", written)
    logger.info("Skipped (bad label)  : %d", skipped_label)
    logger.info("Skipped (demo)       : %d", skipped_demo)
    if long_prompts:
        logger.warning("Long prompts (>%d chars): %d", args.max_prompt_chars, long_prompts)
    logger.info("Label distribution   : %s", label_counts)
    logger.info("Output               : %s", output_path.resolve())


if __name__ == "__main__":
    main()
