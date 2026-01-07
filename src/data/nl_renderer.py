"""
Natural language renderer for logic formula trees.

Converts syntax trees into natural language sentences using atomic propositions
from a proposition pool.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import random

from .syntax_tree import (
    FormulaNode, AtomNode, NegationNode, BinaryNode, QuantifiedNode,
    Connective, Quantifier, LogicOrder
)
from .atomic_proposition_generator import PropositionPool, AtomicProposition


@dataclass
class RenderConfig:
    """Configuration for natural language rendering."""
    # Connective templates (with {left} and {right} placeholders)
    and_templates: List[str] = field(default_factory=lambda: [
        "both {left} and {right}",
        "{left}, and {right}",
    ])
    or_templates: List[str] = field(default_factory=lambda: [
        "either {left} or {right}",
        "{left}, or {right}",
    ])
    implies_templates: List[str] = field(default_factory=lambda: [
        "if {left}, then {right}",
        "{right} whenever {left}",
    ])
    iff_templates: List[str] = field(default_factory=lambda: [
        "{left} if and only if {right}",
    ])
    not_templates: List[str] = field(default_factory=lambda: [
        "it is not the case that {child}",
        "{child} is false",
    ])

    # FOL-specific negation templates for predicates
    not_predicate_templates: List[str] = field(default_factory=lambda: [
        "{entity} is not {predicate}",
        "{entity} does not have the property of being {predicate}",
    ])

    # Quantifier templates (with {var}, {domain}, {body} placeholders)
    forall_templates: List[str] = field(default_factory=lambda: [
        "for all things, {body}",
        "all things are such that {body}",
    ])
    exists_templates: List[str] = field(default_factory=lambda: [
        "there exists something that {body}",
        "something {body}",
    ])

    # FOL predicate templates
    predicate_templates: List[str] = field(default_factory=lambda: [
        "{entity} is {predicate}",
    ])

    # FOL quantified predicate templates (for simple ∀x.P(x))
    forall_predicate_templates: List[str] = field(default_factory=lambda: [
        "all things are {predicate}",
        "everything is {predicate}",
    ])
    exists_predicate_templates: List[str] = field(default_factory=lambda: [
        "something is {predicate}",
        "there exists something that is {predicate}",
    ])

    # FOL quantified implication templates (for ∀x.(P(x)→Q(x)))
    forall_implies_templates: List[str] = field(default_factory=lambda: [
        "all {antecedent} things are {consequent}",
        "if something is {antecedent}, then it is {consequent}",
        "everything that is {antecedent} is also {consequent}",
    ])

    # Whether to vary templates or use consistent ones
    vary_templates: bool = True

    # Capitalize first letter of output
    capitalize_output: bool = True


class NaturalLanguageRenderer:
    """Convert formula trees to natural language using a proposition pool."""

    def __init__(
        self,
        pool: PropositionPool,
        config: Optional[RenderConfig] = None
    ):
        self.pool = pool
        self.config = config or RenderConfig()

        # Mapping from atom identifiers to natural language
        self._atom_mapping: Dict[str, str] = {}
        self._entity_mapping: Dict[str, str] = {}
        self._predicate_mapping: Dict[str, str] = {}

        # Track if we're rendering FOL
        self._is_fol_mode = False

    def render(self, formula: FormulaNode) -> str:
        """
        Render a formula tree to natural language.

        Args:
            formula: The formula tree to render

        Returns:
            Natural language string
        """
        # Reset mappings for fresh render
        self._atom_mapping = {}
        self._entity_mapping = {}
        self._predicate_mapping = {}

        # Detect FOL mode
        self._is_fol_mode = self._contains_fol(formula)

        # Pre-assign atoms/predicates to natural language
        if self._is_fol_mode:
            self._assign_fol_mappings(formula)
        else:
            atoms = formula.get_atoms()
            self._assign_atoms(atoms)

        # Render the formula
        result = self._render_node(formula)

        # Capitalize if configured
        if self.config.capitalize_output and result:
            result = result[0].upper() + result[1:]

        return result

    def render_inference(
        self,
        premises: List[FormulaNode],
        conclusion: FormulaNode
    ) -> Dict[str, Any]:
        """
        Render a complete inference to natural language.

        Returns dict with 'premises' list and 'conclusion' string.
        """
        # Reset mappings
        self._atom_mapping = {}
        self._entity_mapping = {}
        self._predicate_mapping = {}

        # Detect FOL mode from any formula
        self._is_fol_mode = any(self._contains_fol(p) for p in premises) or self._contains_fol(conclusion)

        if self._is_fol_mode:
            # Collect FOL info from all formulas
            for premise in premises:
                self._assign_fol_mappings(premise)
            self._assign_fol_mappings(conclusion)
        else:
            # Collect all atoms from premises and conclusion
            all_atoms = []
            for premise in premises:
                all_atoms.extend(premise.get_atoms())
            all_atoms.extend(conclusion.get_atoms())
            self._assign_atoms(list(set(all_atoms)))

        # Render premises
        rendered_premises = []
        for premise in premises:
            text = self._render_node(premise)
            if self.config.capitalize_output and text:
                text = text[0].upper() + text[1:]
            rendered_premises.append(text)

        # Render conclusion
        conclusion_text = self._render_node(conclusion)
        if self.config.capitalize_output and conclusion_text:
            conclusion_text = conclusion_text[0].upper() + conclusion_text[1:]

        return {
            "premises": rendered_premises,
            "conclusion": conclusion_text
        }

    def _contains_fol(self, formula: FormulaNode) -> bool:
        """Check if formula contains FOL elements (quantifiers or predicates)."""
        if isinstance(formula, QuantifiedNode):
            return True
        if isinstance(formula, AtomNode) and formula.is_predicate:
            return True
        if isinstance(formula, NegationNode):
            return self._contains_fol(formula.child)
        if isinstance(formula, BinaryNode):
            return self._contains_fol(formula.left) or self._contains_fol(formula.right)
        return False

    def _assign_fol_mappings(self, formula: FormulaNode) -> None:
        """Assign natural language to FOL predicates and entities."""
        self._collect_fol_elements(formula)

    def _collect_fol_elements(self, formula: FormulaNode) -> None:
        """Recursively collect and map FOL predicates and entities."""
        if isinstance(formula, AtomNode):
            # Map predicate identifier
            if formula.identifier not in self._predicate_mapping:
                if self.pool.predicates:
                    available = [p for p in self.pool.predicates
                                 if p.text not in self._predicate_mapping.values()]
                    if available:
                        pred = random.choice(available)
                        self._predicate_mapping[formula.identifier] = pred.text
                    else:
                        pred = random.choice(self.pool.predicates)
                        self._predicate_mapping[formula.identifier] = pred.text
                else:
                    self._predicate_mapping[formula.identifier] = formula.identifier.lower()

            # Map entity/variable names
            for var in formula.variables:
                if var not in self._entity_mapping:
                    # Check if it's a constant (a, b, c, d) or variable (x, y, z)
                    if var in ["a", "b", "c", "d"]:
                        # It's a constant - map to an entity name
                        if self.pool.entities and self.pool.entities.names:
                            available = [e for e in self.pool.entities.names
                                         if e not in self._entity_mapping.values()]
                            if available:
                                self._entity_mapping[var] = random.choice(available)
                            else:
                                self._entity_mapping[var] = random.choice(self.pool.entities.names)
                        else:
                            self._entity_mapping[var] = var.upper()
                    else:
                        # It's a variable - use "it" or "something"
                        self._entity_mapping[var] = "it"

        elif isinstance(formula, NegationNode):
            self._collect_fol_elements(formula.child)
        elif isinstance(formula, BinaryNode):
            self._collect_fol_elements(formula.left)
            self._collect_fol_elements(formula.right)
        elif isinstance(formula, QuantifiedNode):
            self._collect_fol_elements(formula.body)

    def _assign_atoms(self, atom_ids: List[str]) -> None:
        """Assign natural language to atom identifiers (propositional logic)."""
        unique_ids = list(set(atom_ids))

        # Sample propositions from pool
        if len(unique_ids) <= len(self.pool.propositions):
            sampled = self.pool.sample_propositions(len(unique_ids))
        else:
            # If we need more than available, reuse
            sampled = []
            pool_size = len(self.pool.propositions)
            for i in range(len(unique_ids)):
                idx = i % pool_size
                sampled.append(self.pool.propositions[idx])

        for atom_id, prop in zip(unique_ids, sampled):
            self._atom_mapping[atom_id] = prop.text

    def _render_node(self, node: FormulaNode) -> str:
        """Render a single node to natural language."""
        if isinstance(node, AtomNode):
            return self._render_atom(node)
        elif isinstance(node, NegationNode):
            return self._render_negation(node)
        elif isinstance(node, BinaryNode):
            return self._render_binary(node)
        elif isinstance(node, QuantifiedNode):
            return self._render_quantified(node)
        else:
            return str(node)

    def _render_atom(self, node: AtomNode) -> str:
        """Render an atomic proposition or FOL predicate."""
        # FOL predicate rendering
        if self._is_fol_mode and node.is_predicate:
            predicate = self._predicate_mapping.get(node.identifier, node.identifier.lower())

            if node.variables:
                # Get entity for the variable/constant
                var = node.variables[0]
                entity = self._entity_mapping.get(var, var)

                # For variables (x, y, z), use "it" in context
                if var in ["x", "y", "z", "w"]:
                    return f"it is {predicate}"
                else:
                    # For constants (a, b, c, d), use the entity name
                    template = random.choice(self.config.predicate_templates) if self.config.vary_templates else self.config.predicate_templates[0]
                    return template.format(entity=entity, predicate=predicate)
            else:
                return f"is {predicate}"

        # Propositional logic rendering
        if node.identifier in self._atom_mapping:
            return self._atom_mapping[node.identifier]

        # Fallback: use identifier directly
        if node.variables:
            return f"{node.identifier}({', '.join(node.variables)})"
        return node.identifier

    def _render_negation(self, node: NegationNode) -> str:
        """Render a negation."""
        # Special handling for FOL predicate negation
        if self._is_fol_mode and isinstance(node.child, AtomNode) and node.child.is_predicate:
            atom = node.child
            predicate = self._predicate_mapping.get(atom.identifier, atom.identifier.lower())

            if atom.variables:
                var = atom.variables[0]
                entity = self._entity_mapping.get(var, var)

                if var in ["x", "y", "z", "w"]:
                    return f"it is not {predicate}"
                else:
                    template = random.choice(self.config.not_predicate_templates) if self.config.vary_templates else self.config.not_predicate_templates[0]
                    return template.format(entity=entity, predicate=predicate)

        child_text = self._render_node(node.child)

        templates = self.config.not_templates
        if self.config.vary_templates:
            template = random.choice(templates)
        else:
            template = templates[0]

        return template.format(child=child_text)

    def _render_binary(self, node: BinaryNode) -> str:
        """Render a binary connective."""
        left_text = self._render_node(node.left)
        right_text = self._render_node(node.right)

        # Choose template based on connective
        if node.connective == Connective.AND:
            templates = self.config.and_templates
        elif node.connective == Connective.OR:
            templates = self.config.or_templates
        elif node.connective == Connective.IMPLIES:
            templates = self.config.implies_templates
        elif node.connective == Connective.IFF:
            templates = self.config.iff_templates
        else:
            templates = ["{left} {connective} {right}"]

        if self.config.vary_templates:
            template = random.choice(templates)
        else:
            template = templates[0]

        return template.format(
            left=left_text,
            right=right_text,
            connective=node.connective.value
        )

    def _render_quantified(self, node: QuantifiedNode) -> str:
        """Render a quantified formula."""
        # Check for special FOL patterns

        # Pattern: ∀x.P(x) or ∃x.P(x) - simple quantified predicate
        if isinstance(node.body, AtomNode) and node.body.is_predicate:
            predicate = self._predicate_mapping.get(node.body.identifier, node.body.identifier.lower())

            if node.quantifier == Quantifier.FORALL:
                templates = self.config.forall_predicate_templates
            else:
                templates = self.config.exists_predicate_templates

            template = random.choice(templates) if self.config.vary_templates else templates[0]
            return template.format(predicate=predicate)

        # Pattern: ∀x.(P(x) → Q(x)) - universal implication
        if node.quantifier == Quantifier.FORALL and isinstance(node.body, BinaryNode):
            if node.body.connective == Connective.IMPLIES:
                # Check if both sides are simple predicates
                if (isinstance(node.body.left, AtomNode) and node.body.left.is_predicate and
                    isinstance(node.body.right, AtomNode) and node.body.right.is_predicate):
                    antecedent = self._predicate_mapping.get(
                        node.body.left.identifier, node.body.left.identifier.lower()
                    )
                    consequent = self._predicate_mapping.get(
                        node.body.right.identifier, node.body.right.identifier.lower()
                    )

                    templates = self.config.forall_implies_templates
                    template = random.choice(templates) if self.config.vary_templates else templates[0]
                    return template.format(antecedent=antecedent, consequent=consequent)

        # Default: render body and wrap with quantifier
        body_text = self._render_node(node.body)

        if node.quantifier == Quantifier.FORALL:
            templates = self.config.forall_templates
        else:
            templates = self.config.exists_templates

        template = random.choice(templates) if self.config.vary_templates else templates[0]

        return template.format(
            var=node.variable,
            domain="things",
            body=body_text
        )


class InferenceRenderer:
    """Convenience class for rendering complete inferences."""

    def __init__(
        self,
        pool: PropositionPool,
        config: Optional[RenderConfig] = None
    ):
        self.renderer = NaturalLanguageRenderer(pool, config)

    def render(
        self,
        premises: List[FormulaNode],
        conclusion: FormulaNode,
        include_therefore: bool = True
    ) -> str:
        """
        Render an inference as a complete text.

        Args:
            premises: List of premise formula trees
            conclusion: Conclusion formula tree
            include_therefore: Whether to add "Therefore" before conclusion

        Returns:
            Complete inference as text
        """
        result = self.renderer.render_inference(premises, conclusion)

        # Build text
        premise_texts = result["premises"]
        conclusion_text = result["conclusion"]

        parts = []
        for i, premise in enumerate(premise_texts):
            parts.append(premise + ".")

        if include_therefore:
            parts.append(f"Therefore, {conclusion_text.lower()}.")
        else:
            parts.append(f"{conclusion_text}.")

        return " ".join(parts)

    def render_structured(
        self,
        premises: List[FormulaNode],
        conclusion: FormulaNode
    ) -> Dict[str, Any]:
        """
        Render to structured format for data generation.

        Returns:
            Dict with 'premises' list and 'conclusion' string
        """
        return self.renderer.render_inference(premises, conclusion)
