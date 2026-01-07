"""
Syntax tree nodes for random logic formula generation.

Supports both propositional (0th order) and first-order logic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple
import random


class LogicOrder(Enum):
    """Logic order type."""
    PROPOSITIONAL = "propositional"  # 0th order
    FIRST_ORDER = "first_order"      # 1st order


class Connective(Enum):
    """Logical connectives."""
    AND = "and"
    OR = "or"
    IMPLIES = "implies"
    IFF = "iff"  # biconditional


class Quantifier(Enum):
    """First-order logic quantifiers."""
    FORALL = "forall"
    EXISTS = "exists"


@dataclass
class FormulaNode(ABC):
    """Abstract base class for formula tree nodes."""

    @abstractmethod
    def depth(self) -> int:
        """Return the depth of the tree rooted at this node."""
        pass

    @abstractmethod
    def to_formal(self) -> str:
        """Convert to formal notation string."""
        pass

    @abstractmethod
    def get_atoms(self) -> List[str]:
        """Get all atom identifiers in this subtree."""
        pass

    @abstractmethod
    def copy(self) -> "FormulaNode":
        """Deep copy this node and its subtree."""
        pass

    def is_atomic(self) -> bool:
        """Check if this is an atomic formula."""
        return isinstance(self, AtomNode)


@dataclass
class AtomNode(FormulaNode):
    """
    Atomic proposition or predicate.

    For propositional logic: just an identifier like P, Q, R
    For first-order logic: predicate with variables like P(x), R(x,y)
    """
    identifier: str  # P, Q, R, etc.
    variables: List[str] = field(default_factory=list)  # [x], [x, y] for FOL
    is_predicate: bool = False

    def depth(self) -> int:
        return 1

    def to_formal(self) -> str:
        if self.variables:
            return f"{self.identifier}({','.join(self.variables)})"
        return self.identifier

    def get_atoms(self) -> List[str]:
        return [self.identifier]

    def copy(self) -> "AtomNode":
        return AtomNode(
            identifier=self.identifier,
            variables=self.variables.copy(),
            is_predicate=self.is_predicate
        )


@dataclass
class NegationNode(FormulaNode):
    """Negation: NOT child."""
    child: FormulaNode

    def depth(self) -> int:
        return 1 + self.child.depth()

    def to_formal(self) -> str:
        child_str = self.child.to_formal()
        if isinstance(self.child, (BinaryNode, QuantifiedNode)):
            return f"¬({child_str})"
        return f"¬{child_str}"

    def get_atoms(self) -> List[str]:
        return self.child.get_atoms()

    def copy(self) -> "NegationNode":
        return NegationNode(child=self.child.copy())


@dataclass
class BinaryNode(FormulaNode):
    """Binary connective: left OP right."""
    connective: Connective
    left: FormulaNode
    right: FormulaNode

    def depth(self) -> int:
        return 1 + max(self.left.depth(), self.right.depth())

    def to_formal(self) -> str:
        left_str = self.left.to_formal()
        right_str = self.right.to_formal()

        # Add parentheses for complex subformulas
        if isinstance(self.left, BinaryNode):
            left_str = f"({left_str})"
        if isinstance(self.right, BinaryNode):
            right_str = f"({right_str})"

        symbols = {
            Connective.AND: "∧",
            Connective.OR: "∨",
            Connective.IMPLIES: "→",
            Connective.IFF: "↔"
        }
        return f"{left_str} {symbols[self.connective]} {right_str}"

    def get_atoms(self) -> List[str]:
        return self.left.get_atoms() + self.right.get_atoms()

    def copy(self) -> "BinaryNode":
        return BinaryNode(
            connective=self.connective,
            left=self.left.copy(),
            right=self.right.copy()
        )


@dataclass
class QuantifiedNode(FormulaNode):
    """Quantified formula: QUANTIFIER variable. body"""
    quantifier: Quantifier
    variable: str  # The bound variable (x, y, etc.)
    body: FormulaNode

    def depth(self) -> int:
        return 1 + self.body.depth()

    def to_formal(self) -> str:
        q_symbol = "∀" if self.quantifier == Quantifier.FORALL else "∃"
        body_str = self.body.to_formal()
        return f"{q_symbol}{self.variable}.{body_str}"

    def get_atoms(self) -> List[str]:
        return self.body.get_atoms()

    def copy(self) -> "QuantifiedNode":
        return QuantifiedNode(
            quantifier=self.quantifier,
            variable=self.variable,
            body=self.body.copy()
        )


@dataclass
class TreeGeneratorConfig:
    """Configuration for random tree generation."""
    min_depth: int = 2
    max_depth: int = 5
    logic_order: LogicOrder = LogicOrder.PROPOSITIONAL

    # Probability weights for node types (will be normalized)
    atom_weight: float = 1.0
    negation_weight: float = 0.3
    binary_weight: float = 1.0
    quantifier_weight: float = 0.4  # Only used for FOL

    # Probability weights for binary connectives
    and_weight: float = 1.0
    or_weight: float = 1.0
    implies_weight: float = 1.2  # Slightly favor implications
    iff_weight: float = 0.5

    # Probability weights for quantifiers (FOL only)
    forall_weight: float = 1.0
    exists_weight: float = 0.8

    # Left-right balance for binary nodes (0.5 = balanced)
    left_depth_bias: float = 0.5

    # Available atom identifiers
    atom_pool: List[str] = field(default_factory=lambda: ["P", "Q", "R", "S", "T", "U"])
    variable_pool: List[str] = field(default_factory=lambda: ["x", "y", "z", "w"])


class RandomTreeGenerator:
    """Generate random formula trees with configurable depth and structure."""

    def __init__(self, config: Optional[TreeGeneratorConfig] = None):
        self.config = config or TreeGeneratorConfig()
        self._used_atoms: List[str] = []
        self._used_variables: List[str] = []

    def generate(self, target_depth: Optional[int] = None) -> FormulaNode:
        """
        Generate a random formula tree.

        Args:
            target_depth: Target depth (if None, random between min and max)

        Returns:
            Root node of generated formula tree
        """
        if target_depth is None:
            target_depth = random.randint(
                self.config.min_depth,
                self.config.max_depth
            )

        self._used_atoms = []
        self._used_variables = []

        return self._generate_node(target_depth)

    def _generate_node(self, remaining_depth: int) -> FormulaNode:
        """Generate a node with the given remaining depth budget."""
        # Base case: must generate atom
        if remaining_depth <= 1:
            return self._generate_atom()

        # Choose node type based on weights
        node_type = self._choose_node_type(remaining_depth)

        if node_type == "atom":
            return self._generate_atom()
        elif node_type == "negation":
            return self._generate_negation(remaining_depth)
        elif node_type == "binary":
            return self._generate_binary(remaining_depth)
        elif node_type == "quantifier":
            return self._generate_quantified(remaining_depth)
        else:
            return self._generate_atom()

    def _choose_node_type(self, remaining_depth: int) -> str:
        """Choose node type based on configured weights."""
        weights = {
            "atom": self.config.atom_weight if remaining_depth <= 2 else self.config.atom_weight * 0.3,
            "negation": self.config.negation_weight,
            "binary": self.config.binary_weight,
        }

        # Add quantifier only for FOL
        if self.config.logic_order == LogicOrder.FIRST_ORDER:
            weights["quantifier"] = self.config.quantifier_weight

        # Normalize weights
        total = sum(weights.values())
        r = random.random() * total
        cumulative = 0

        for node_type, weight in weights.items():
            cumulative += weight
            if r <= cumulative:
                return node_type

        return "binary"  # Default fallback

    def _generate_atom(self) -> AtomNode:
        """Generate an atomic formula."""
        # Get next available atom identifier
        available = [a for a in self.config.atom_pool if a not in self._used_atoms]
        if not available:
            # Reuse if pool exhausted
            available = self.config.atom_pool

        identifier = random.choice(available)
        self._used_atoms.append(identifier)

        # For FOL, potentially add variables
        variables = []
        is_predicate = False

        if self.config.logic_order == LogicOrder.FIRST_ORDER:
            is_predicate = True
            # Add 1-2 variables
            num_vars = random.choice([1, 1, 2])  # More likely 1 variable
            available_vars = [v for v in self.config.variable_pool if v not in self._used_variables]
            if not available_vars:
                available_vars = self.config.variable_pool

            for _ in range(min(num_vars, len(available_vars))):
                var = random.choice(available_vars)
                if var not in variables:
                    variables.append(var)

        return AtomNode(
            identifier=identifier,
            variables=variables,
            is_predicate=is_predicate
        )

    def _generate_negation(self, remaining_depth: int) -> NegationNode:
        """Generate a negation node."""
        child = self._generate_node(remaining_depth - 1)
        return NegationNode(child=child)

    def _generate_binary(self, remaining_depth: int) -> BinaryNode:
        """Generate a binary connective node."""
        # Choose connective based on weights
        connective = self._choose_connective()

        # Distribute depth between children based on bias
        left_depth, right_depth = self._distribute_depth(remaining_depth - 1)

        left = self._generate_node(left_depth)
        right = self._generate_node(right_depth)

        return BinaryNode(connective=connective, left=left, right=right)

    def _choose_connective(self) -> Connective:
        """Choose a binary connective based on weights."""
        weights = {
            Connective.AND: self.config.and_weight,
            Connective.OR: self.config.or_weight,
            Connective.IMPLIES: self.config.implies_weight,
            Connective.IFF: self.config.iff_weight,
        }

        total = sum(weights.values())
        r = random.random() * total
        cumulative = 0

        for connective, weight in weights.items():
            cumulative += weight
            if r <= cumulative:
                return connective

        return Connective.AND

    def _distribute_depth(self, total_depth: int) -> Tuple[int, int]:
        """Distribute depth between left and right children."""
        if total_depth <= 2:
            return (1, 1)

        # Apply bias for depth distribution
        bias = self.config.left_depth_bias
        if random.random() < bias:
            left_depth = random.randint(1, total_depth - 1)
            right_depth = max(1, total_depth - left_depth)
        else:
            right_depth = random.randint(1, total_depth - 1)
            left_depth = max(1, total_depth - right_depth)

        return (left_depth, right_depth)

    def _generate_quantified(self, remaining_depth: int) -> QuantifiedNode:
        """Generate a quantified formula (FOL only)."""
        # Choose quantifier
        weights = {
            Quantifier.FORALL: self.config.forall_weight,
            Quantifier.EXISTS: self.config.exists_weight,
        }
        total = sum(weights.values())
        r = random.random() * total

        quantifier = Quantifier.FORALL if r <= weights[Quantifier.FORALL] else Quantifier.EXISTS

        # Choose variable
        available_vars = [v for v in self.config.variable_pool if v not in self._used_variables]
        if not available_vars:
            available_vars = self.config.variable_pool
        variable = random.choice(available_vars)
        self._used_variables.append(variable)

        # Generate body
        body = self._generate_node(remaining_depth - 1)

        return QuantifiedNode(quantifier=quantifier, variable=variable, body=body)

    def generate_distinct_formulas(self, count: int) -> List[FormulaNode]:
        """Generate multiple distinct formulas."""
        formulas = []
        seen_formal = set()

        attempts = 0
        max_attempts = count * 3

        while len(formulas) < count and attempts < max_attempts:
            formula = self.generate()
            formal = formula.to_formal()

            if formal not in seen_formal:
                formulas.append(formula)
                seen_formal.add(formal)

            attempts += 1

        return formulas


def negate(formula: FormulaNode) -> FormulaNode:
    """Create the negation of a formula, simplifying double negation."""
    if isinstance(formula, NegationNode):
        # Double negation elimination: ¬¬P -> P
        return formula.child.copy()
    return NegationNode(child=formula.copy())


def make_implication(antecedent: FormulaNode, consequent: FormulaNode) -> BinaryNode:
    """Create P -> Q."""
    return BinaryNode(
        connective=Connective.IMPLIES,
        left=antecedent.copy(),
        right=consequent.copy()
    )


def make_conjunction(left: FormulaNode, right: FormulaNode) -> BinaryNode:
    """Create P ∧ Q."""
    return BinaryNode(
        connective=Connective.AND,
        left=left.copy(),
        right=right.copy()
    )


def make_disjunction(left: FormulaNode, right: FormulaNode) -> BinaryNode:
    """Create P ∨ Q."""
    return BinaryNode(
        connective=Connective.OR,
        left=left.copy(),
        right=right.copy()
    )


def make_biconditional(left: FormulaNode, right: FormulaNode) -> BinaryNode:
    """Create P ↔ Q."""
    return BinaryNode(
        connective=Connective.IFF,
        left=left.copy(),
        right=right.copy()
    )
