#!/usr/bin/env python3
"""
Logic Training Data Generation Pipeline

Generates synthetic logic reasoning examples using:
- Propositional logic (0th order): P ^ Q -> R
- First-order logic (1st order): forall x. Human(x) -> Mortal(x)

Two generation modes:
1. Template-based (default): Uses fixed logic templates
2. Tree-based (--use-tree-generator): Random syntax tree generation for infinite variety

Atomic propositions are generated using Anthropic Claude API (pooled approach).
Only 5 API calls are made to generate a pool, then thousands of examples
are created by sampling from this pool.

Usage:
    python scripts/generate_logic_data.py -n 1000 -o ./data/logic_train.jsonl
    python scripts/generate_logic_data.py -n 500 -t propositional -o ./data/prop.jsonl
    python scripts/generate_logic_data.py --use-fallback -n 100  # No API calls
    python scripts/generate_logic_data.py --use-tree-generator --min-depth 2 --max-depth 4 -n 1000
"""

import sys
import os
import argparse
import json
import uuid
import random
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.schema import Annotation, Premise
from src.data.logic_templates import (
    LogicTemplate, LogicType,
    get_all_templates, get_templates_by_type,
    PROPOSITIONAL_TEMPLATES, FIRST_ORDER_TEMPLATES
)
from src.data.atomic_proposition_generator import (
    AtomicPropositionGenerator, GeneratorConfig, PropositionPool,
    create_fallback_pool, TOPIC_CATEGORIES
)
from src.data.syntax_tree import (
    LogicOrder, TreeGeneratorConfig, RandomTreeGenerator
)
from src.data.inference_generator import (
    InferenceGenerator, InferenceGeneratorConfig, InferencePattern, Inference,
    PROPOSITIONAL_PATTERNS, FOL_PATTERNS
)
from src.data.nl_renderer import InferenceRenderer, RenderConfig


class LogicDataGenerator:
    """Generate logic training data by combining templates with atomic propositions."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.9,
        use_fallback: bool = False
    ):
        self.use_fallback = use_fallback
        self.pool: Optional[PropositionPool] = None
        self.prop_generator: Optional[AtomicPropositionGenerator] = None

        if not use_fallback:
            config = GeneratorConfig(
                api_key=api_key,
                model=model,
                temperature=temperature
            )
            self.prop_generator = AtomicPropositionGenerator(config)

    def initialize_pool(
        self,
        topics: Optional[List[str]] = None,
        verbose: bool = True
    ):
        """Initialize the proposition pool (makes API calls or uses fallback)."""
        if self.use_fallback:
            if verbose:
                print("Using fallback pool (no API calls)...")
            self.pool = create_fallback_pool()
        else:
            self.pool = self.prop_generator.generate_pool(topics=topics, verbose=verbose)

    def generate_example(
        self,
        template: LogicTemplate
    ) -> Optional[Annotation]:
        """Generate a single logic example from a template by sampling from pool."""
        if self.pool is None:
            raise ValueError("Pool not initialized. Call initialize_pool() first.")

        # Get propositions by sampling from pool
        propositions, entities = self._sample_from_pool(template)

        # Fill in the template
        filled_text = self._fill_template(template, propositions, entities)

        # Parse into premises and conclusion
        premises, conclusion = self._parse_filled_template(filled_text)

        # Create Premise objects
        premise_objects = [
            Premise(
                id=f"p{i+1}",
                text=premise_text
            )
            for i, premise_text in enumerate(premises)
        ]

        return Annotation(
            id=str(uuid.uuid4()),
            premises=premise_objects,
            conclusion=conclusion,
            confidence=1.0,
            annotator_id=f"logic_generator_{template.logic_type.value}",
            verifier_notes=json.dumps({
                "template": template.name,
                "formal_notation": template.formal_notation,
                "logic_type": template.logic_type.value
            }),
            timestamp=datetime.now().isoformat()
        )

    def _sample_from_pool(
        self,
        template: LogicTemplate
    ) -> tuple:
        """Sample propositions and entities from pool for a template."""
        propositions = {}
        entities = {}

        if template.logic_type == LogicType.PROPOSITIONAL:
            # For propositional logic, sample full propositions
            sampled = self.pool.sample_propositions(len(template.premise_slots))
            for slot, prop in zip(template.premise_slots, sampled):
                propositions[slot] = prop.text
        else:
            # For first-order logic, handle based on template type
            if template.name == "universal_instantiation":
                # Need: X (category), P (predicate)
                cat = self.pool.sample_categories(1)[0] if self.pool.categories else None
                pred = self.pool.sample_predicates(1)[0] if self.pool.predicates else None
                propositions["X"] = cat.text if cat else "things"
                propositions["P"] = pred.text if pred else "special"
            elif template.name == "relational_inference":
                # Need: R (relation), S (predicate)
                rel = self.pool.sample_relations(1)[0] if self.pool.relations else None
                pred = self.pool.sample_predicates(1)[0] if self.pool.predicates else None
                propositions["R"] = rel.text if rel else "knows"
                propositions["S"] = pred.text if pred else "wise"
            else:
                # Default: sample predicates for P, Q, R, etc.
                sampled = self.pool.sample_predicates(len(template.premise_slots))
                for slot, prop in zip(template.premise_slots, sampled):
                    propositions[slot] = prop.text

            # Sample entities
            if template.entity_slots:
                entity_names = self.pool.entities.sample(len(template.entity_slots))
                for slot, name in zip(template.entity_slots, entity_names):
                    entities[slot] = name

        return propositions, entities

    def _fill_template(
        self,
        template: LogicTemplate,
        propositions: Dict[str, str],
        entities: Dict[str, str]
    ) -> str:
        """Fill template placeholders with propositions and entities."""
        filled = template.pattern

        # Replace proposition placeholders
        for slot, text in propositions.items():
            filled = filled.replace(f"{{{slot}}}", text)

        # Replace entity placeholders
        for slot, name in entities.items():
            filled = filled.replace(f"{{{slot}}}", name)

        return filled

    def _parse_filled_template(self, filled_text: str) -> tuple:
        """Parse filled template into premises and conclusion."""
        # Split by "Therefore" or similar conclusion markers
        conclusion_markers = ["Therefore,", "Hence,", "Thus,", "So,"]

        premises_text = filled_text
        conclusion = ""

        for marker in conclusion_markers:
            if marker in filled_text:
                parts = filled_text.split(marker)
                premises_text = parts[0].strip()
                conclusion = parts[1].strip().rstrip(".")
                break

        # Split premises by sentence
        premise_sentences = re.split(r'(?<=[.!?])\s+', premises_text)
        premises = [p.strip().rstrip(".") for p in premise_sentences if p.strip()]

        # Ensure conclusion exists
        if not conclusion:
            conclusion = premises[-1] if premises else "conclusion"
            premises = premises[:-1] if len(premises) > 1 else premises

        return premises, conclusion

    def generate_dataset(
        self,
        num_examples: int = 1000,
        logic_types: Optional[List[str]] = None,
        propositional_ratio: float = 0.5,
        topics: Optional[List[str]] = None,
        output_path: Optional[str] = None,
        verbose: bool = True
    ) -> List[Annotation]:
        """
        Generate a dataset of logic examples.

        Args:
            num_examples: Total number of examples to generate
            logic_types: List of logic types ["propositional", "first_order"] or None for both
            propositional_ratio: Ratio of propositional vs first-order (only if logic_types is None)
            topics: List of topics to use for pool generation
            output_path: Path to save JSONL output
            verbose: Print progress messages

        Returns:
            List of Annotation objects
        """
        # Initialize pool first (this is where API calls happen)
        if self.pool is None:
            self.initialize_pool(topics=topics, verbose=verbose)

        annotations = []

        # Determine templates to use
        if logic_types is None or set(logic_types) == {"propositional", "first_order"}:
            num_prop = int(num_examples * propositional_ratio)
            num_fol = num_examples - num_prop
            prop_templates = PROPOSITIONAL_TEMPLATES
            fol_templates = FIRST_ORDER_TEMPLATES
        elif "propositional" in logic_types and "first_order" not in logic_types:
            num_prop = num_examples
            num_fol = 0
            prop_templates = PROPOSITIONAL_TEMPLATES
            fol_templates = []
        else:
            num_prop = 0
            num_fol = num_examples
            prop_templates = []
            fol_templates = FIRST_ORDER_TEMPLATES

        if verbose:
            print(f"\nGenerating {num_prop} propositional and {num_fol} first-order examples...")
            print("(No additional API calls - sampling from pool)")

        # Generate propositional examples
        for i in range(num_prop):
            template = random.choice(prop_templates)

            try:
                annotation = self.generate_example(template)
                if annotation:
                    annotations.append(annotation)
                    if verbose and (i + 1) % 100 == 0:
                        print(f"  Generated {i + 1}/{num_prop} propositional examples")
            except Exception as e:
                if verbose:
                    print(f"  Error generating propositional example {i}: {e}")

        # Generate first-order examples
        for i in range(num_fol):
            template = random.choice(fol_templates)

            try:
                annotation = self.generate_example(template)
                if annotation:
                    annotations.append(annotation)
                    if verbose and (i + 1) % 100 == 0:
                        print(f"  Generated {i + 1}/{num_fol} first-order examples")
            except Exception as e:
                if verbose:
                    print(f"  Error generating first-order example {i}: {e}")

        # Shuffle
        random.shuffle(annotations)

        # Save to JSONL
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                for ann in annotations:
                    f.write(ann.to_jsonl() + "\n")
            if verbose:
                print(f"\nSaved {len(annotations)} examples to {output_path}")

        return annotations


class TreeBasedDataGenerator:
    """Generate logic training data using random syntax tree generation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.9,
        use_fallback: bool = False,
        min_depth: int = 2,
        max_depth: int = 5,
        logic_order: str = "propositional",
        inference_patterns: Optional[List[str]] = None
    ):
        self.use_fallback = use_fallback
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.pool: Optional[PropositionPool] = None
        self.prop_generator: Optional[AtomicPropositionGenerator] = None
        self.inference_generator: Optional[InferenceGenerator] = None
        self.renderer: Optional[InferenceRenderer] = None

        # Parse logic order
        self.logic_order_str = logic_order
        if logic_order == "first_order":
            self.logic_order = LogicOrder.FIRST_ORDER
        else:
            self.logic_order = LogicOrder.PROPOSITIONAL

        # Parse inference patterns
        self.patterns: Optional[List[InferencePattern]] = None
        if inference_patterns:
            self.patterns = []
            for p in inference_patterns:
                try:
                    self.patterns.append(InferencePattern(p))
                except ValueError:
                    print(f"Warning: Unknown inference pattern '{p}', skipping")
        elif logic_order == "first_order":
            self.patterns = FOL_PATTERNS
        elif logic_order == "propositional":
            self.patterns = PROPOSITIONAL_PATTERNS
        elif logic_order == "both":
            # Use all patterns for both
            self.patterns = PROPOSITIONAL_PATTERNS + FOL_PATTERNS

        if not use_fallback:
            config = GeneratorConfig(
                api_key=api_key,
                model=model,
                temperature=temperature
            )
            self.prop_generator = AtomicPropositionGenerator(config)

    def initialize_pool(
        self,
        topics: Optional[List[str]] = None,
        verbose: bool = True
    ):
        """Initialize the proposition pool (makes API calls or uses fallback)."""
        if self.use_fallback:
            if verbose:
                print("Using fallback pool (no API calls)...")
            self.pool = create_fallback_pool()
        else:
            self.pool = self.prop_generator.generate_pool(topics=topics, verbose=verbose)

        # Initialize inference generator with logic order
        inference_config = InferenceGeneratorConfig(
            min_subformula_depth=self.min_depth,
            max_subformula_depth=self.max_depth,
            logic_order=self.logic_order,
            patterns=self.patterns
        )
        self.inference_generator = InferenceGenerator(inference_config)

        # Initialize renderer
        self.renderer = InferenceRenderer(self.pool)

    def generate_example(self) -> Optional[Annotation]:
        """Generate a single logic example using tree generation."""
        if self.pool is None:
            raise ValueError("Pool not initialized. Call initialize_pool() first.")

        # Generate inference
        inference = self.inference_generator.generate()

        # Render to natural language
        rendered = self.renderer.render_structured(
            inference.premises,
            inference.conclusion
        )

        # Create Premise objects
        premise_objects = [
            Premise(
                id=f"p{i+1}",
                text=premise_text
            )
            for i, premise_text in enumerate(rendered["premises"])
        ]

        return Annotation(
            id=str(uuid.uuid4()),
            premises=premise_objects,
            conclusion=rendered["conclusion"],
            confidence=1.0,
            annotator_id=f"tree_generator_{inference.pattern.value}",
            verifier_notes=json.dumps({
                "pattern": inference.pattern.value,
                "formal_notation": inference.formal_notation,
                "full_formal": inference.to_formal(),
                "generator": "tree_based",
                "logic_order": inference.logic_order.value,
                "depth_range": [self.min_depth, self.max_depth]
            }),
            timestamp=datetime.now().isoformat()
        )

    def generate_dataset(
        self,
        num_examples: int = 1000,
        topics: Optional[List[str]] = None,
        output_path: Optional[str] = None,
        verbose: bool = True
    ) -> List[Annotation]:
        """
        Generate a dataset using random tree generation.

        Args:
            num_examples: Total number of examples to generate
            topics: List of topics to use for pool generation
            output_path: Path to save JSONL output
            verbose: Print progress messages

        Returns:
            List of Annotation objects
        """
        # Initialize pool first (this is where API calls happen)
        if self.pool is None:
            self.initialize_pool(topics=topics, verbose=verbose)

        annotations = []

        if verbose:
            print(f"\nGenerating {num_examples} examples with tree generator...")
            print(f"Depth range: {self.min_depth} - {self.max_depth}")
            if self.patterns:
                print(f"Patterns: {[p.value for p in self.patterns]}")
            else:
                print("Patterns: all")
            print("(No additional API calls - sampling from pool)")

        for i in range(num_examples):
            try:
                annotation = self.generate_example()
                if annotation:
                    annotations.append(annotation)
                    if verbose and (i + 1) % 100 == 0:
                        print(f"  Generated {i + 1}/{num_examples} examples")
            except Exception as e:
                if verbose:
                    print(f"  Error generating example {i}: {e}")

        # Shuffle
        random.shuffle(annotations)

        # Save to JSONL
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                for ann in annotations:
                    f.write(ann.to_jsonl() + "\n")
            if verbose:
                print(f"\nSaved {len(annotations)} examples to {output_path}")

        return annotations


def main():
    parser = argparse.ArgumentParser(
        description="Generate logic training data using propositional and first-order logic templates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 1000 examples using templates (makes ~5 API calls for pool, then samples)
  python scripts/generate_logic_data.py -n 1000 -o ./data/logic_train.jsonl

  # Only propositional logic
  python scripts/generate_logic_data.py -n 500 -t propositional -o ./data/prop.jsonl

  # Only first-order logic
  python scripts/generate_logic_data.py -n 500 -t first_order -o ./data/fol.jsonl

  # Use fallback pool (no API calls at all)
  python scripts/generate_logic_data.py --use-fallback -n 1000 -o ./data/logic_fallback.jsonl

  # Use tree generator for infinite variety
  python scripts/generate_logic_data.py --use-tree-generator --min-depth 2 --max-depth 4 -n 1000

  # Tree generator with specific inference patterns
  python scripts/generate_logic_data.py --use-tree-generator --inference-patterns modus_ponens hypothetical_syllogism -n 500

  # Custom API settings
  ANTHROPIC_API_KEY=sk-... python scripts/generate_logic_data.py -n 1000 --model claude-sonnet-4-20250514

Available inference patterns for --inference-patterns:
  modus_ponens, modus_tollens, hypothetical_syllogism, disjunctive_syllogism,
  conjunction_intro, conjunction_elim, disjunction_intro, double_negation_elim,
  constructive_dilemma, biconditional_intro, biconditional_elim, absorption
"""
    )
    parser.add_argument(
        "--num-examples", "-n",
        type=int,
        default=1000,
        help="Number of examples to generate (default: 1000)"
    )
    parser.add_argument(
        "--logic-types", "-t",
        nargs="+",
        choices=["propositional", "first_order", "both"],
        default=["both"],
        help="Logic types to include (default: both)"
    )
    parser.add_argument(
        "--propositional-ratio", "-r",
        type=float,
        default=0.5,
        help="Ratio of propositional to first-order examples when using both (default: 0.5)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./data/logic_training_data.jsonl",
        help="Output JSONL file path (default: ./data/logic_training_data.jsonl)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Claude model to use (default: claude-sonnet-4-20250514)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Generation temperature for pool creation (default: 0.9)"
    )
    parser.add_argument(
        "--topics",
        nargs="+",
        default=None,
        help="Specific topics to use for pool generation (default: all available topics)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--use-fallback",
        action="store_true",
        help="Use fallback pool (no API calls, limited variety)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress messages"
    )

    # Tree generator arguments
    parser.add_argument(
        "--use-tree-generator",
        action="store_true",
        help="Use random tree generator instead of fixed templates (infinite variety)"
    )
    parser.add_argument(
        "--logic-order",
        type=str,
        choices=["propositional", "first_order", "both"],
        default="propositional",
        help="Logic order to generate (default: propositional). Options: propositional (0th order), first_order (1st order), both"
    )
    parser.add_argument(
        "--min-depth",
        type=int,
        default=2,
        help="Minimum formula tree depth (default: 2, only with --use-tree-generator)"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=5,
        help="Maximum formula tree depth (default: 5, only with --use-tree-generator)"
    )
    parser.add_argument(
        "--inference-patterns",
        nargs="+",
        default=None,
        help="Specific inference patterns to use (default: based on --logic-order). Propositional: modus_ponens, modus_tollens, hypothetical_syllogism, disjunctive_syllogism, conjunction_intro, conjunction_elim, disjunction_intro, double_negation_elim, constructive_dilemma, biconditional_intro, biconditional_elim, absorption. FOL: universal_instantiation, universal_modus_ponens, existential_generalization, universal_syllogism, universal_contraposition, existential_syllogism"
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Determine if using tree generator or template-based
    if args.use_tree_generator:
        # Tree-based generation
        try:
            generator = TreeBasedDataGenerator(
                api_key=args.api_key,
                model=args.model,
                temperature=args.temperature,
                use_fallback=args.use_fallback,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
                logic_order=args.logic_order,
                inference_patterns=args.inference_patterns
            )
        except (ImportError, ValueError) as e:
            if args.use_fallback:
                generator = TreeBasedDataGenerator(
                    use_fallback=True,
                    min_depth=args.min_depth,
                    max_depth=args.max_depth,
                    logic_order=args.logic_order,
                    inference_patterns=args.inference_patterns
                )
            else:
                print(f"Error initializing generator: {e}")
                print("\nOptions:")
                print("  1. Install anthropic: pip install anthropic")
                print("  2. Set ANTHROPIC_API_KEY environment variable or use --api-key")
                print("  3. Use --use-fallback to generate without API calls")
                sys.exit(1)

        # Generate dataset
        if not args.quiet:
            print(f"{'='*60}")
            print("Logic Training Data Generator (Tree-Based)")
            print(f"{'='*60}")
            print(f"Mode: Random syntax tree generation")
            print(f"Logic order: {args.logic_order}")
            print(f"Seed: {args.seed}")
            print(f"Model: {args.model if not args.use_fallback else 'fallback (no API)'}")
            print(f"Examples to generate: {args.num_examples}")
            print(f"Depth range: {args.min_depth} - {args.max_depth}")
            if args.inference_patterns:
                print(f"Inference patterns: {args.inference_patterns}")
            if args.topics:
                print(f"Topics: {args.topics}")
            print()

        annotations = generator.generate_dataset(
            num_examples=args.num_examples,
            topics=args.topics,
            output_path=args.output,
            verbose=not args.quiet
        )

        # Print summary
        if not args.quiet:
            print(f"\n{'='*60}")
            print("Generation Complete")
            print(f"{'='*60}")
            print(f"Total examples: {len(annotations)}")

            # Count by pattern and logic order
            pattern_counts = {}
            order_counts = {"propositional": 0, "first_order": 0}
            for a in annotations:
                if a.verifier_notes:
                    notes = json.loads(a.verifier_notes)
                    pattern = notes.get("pattern", "unknown")
                    pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
                    logic_order = notes.get("logic_order", "propositional")
                    order_counts[logic_order] = order_counts.get(logic_order, 0) + 1

            print(f"\nBy logic order:")
            print(f"  Propositional (0th order): {order_counts.get('propositional', 0)}")
            print(f"  First-order (1st order): {order_counts.get('first_order', 0)}")

            print(f"\nBy inference pattern:")
            for pattern, count in sorted(pattern_counts.items()):
                print(f"  {pattern}: {count}")
            print(f"\nOutput: {args.output}")

            # Show sample
            if annotations:
                print(f"\nSample output:")
                sample = annotations[0]
                print(f"  Premises: {[p.text for p in sample.premises]}")
                print(f"  Conclusion: {sample.conclusion}")
                if sample.verifier_notes:
                    notes = json.loads(sample.verifier_notes)
                    print(f"  Formal: {notes.get('full_formal', 'N/A')}")
                    print(f"  Logic order: {notes.get('logic_order', 'N/A')}")

    else:
        # Template-based generation (original behavior)
        # Parse logic types
        if "both" in args.logic_types:
            logic_types = None  # Use both
        else:
            logic_types = args.logic_types

        # Create generator
        try:
            generator = LogicDataGenerator(
                api_key=args.api_key,
                model=args.model,
                temperature=args.temperature,
                use_fallback=args.use_fallback
            )
        except (ImportError, ValueError) as e:
            if args.use_fallback:
                generator = LogicDataGenerator(use_fallback=True)
            else:
                print(f"Error initializing generator: {e}")
                print("\nOptions:")
                print("  1. Install anthropic: pip install anthropic")
                print("  2. Set ANTHROPIC_API_KEY environment variable or use --api-key")
                print("  3. Use --use-fallback to generate without API calls")
                sys.exit(1)

        # Generate dataset
        if not args.quiet:
            print(f"{'='*60}")
            print("Logic Training Data Generator (Template-Based)")
            print(f"{'='*60}")
            print(f"Mode: Fixed logic templates")
            print(f"Seed: {args.seed}")
            print(f"Model: {args.model if not args.use_fallback else 'fallback (no API)'}")
            print(f"Examples to generate: {args.num_examples}")
            if args.topics:
                print(f"Topics: {args.topics}")
            print()

        annotations = generator.generate_dataset(
            num_examples=args.num_examples,
            logic_types=logic_types,
            propositional_ratio=args.propositional_ratio,
            topics=args.topics,
            output_path=args.output,
            verbose=not args.quiet
        )

        # Print summary
        if not args.quiet:
            print(f"\n{'='*60}")
            print("Generation Complete")
            print(f"{'='*60}")
            print(f"Total examples: {len(annotations)}")

            # Count by type
            prop_count = sum(
                1 for a in annotations
                if a.verifier_notes and "propositional" in a.verifier_notes
            )
            fol_count = len(annotations) - prop_count
            print(f"Propositional: {prop_count}")
            print(f"First-order: {fol_count}")
            print(f"Output: {args.output}")

            # Show sample
            if annotations:
                print(f"\nSample output:")
                sample = annotations[0]
                print(f"  Premises: {[p.text for p in sample.premises]}")
                print(f"  Conclusion: {sample.conclusion}")


if __name__ == "__main__":
    main()
