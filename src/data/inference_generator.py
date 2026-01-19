"""
Inference pattern generator for creating valid logical inferences.

Generates (premises, conclusion) pairs by instantiating inference patterns
with randomly generated subformulas.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable
from enum import Enum
import random

from .syntax_tree import (
    FormulaNode, AtomNode, NegationNode, BinaryNode, QuantifiedNode,
    Connective, Quantifier, LogicOrder, TreeGeneratorConfig, RandomTreeGenerator,
    negate, make_implication, make_conjunction, make_disjunction, make_biconditional
)


class InferencePattern(Enum):
    """Valid inference patterns."""
    # Basic propositional patterns (0th order)
    MODUS_PONENS = "modus_ponens"
    MODUS_TOLLENS = "modus_tollens"
    HYPOTHETICAL_SYLLOGISM = "hypothetical_syllogism"
    DISJUNCTIVE_SYLLOGISM = "disjunctive_syllogism"
    CONJUNCTION_INTRO = "conjunction_intro"
    CONJUNCTION_ELIM = "conjunction_elim"
    DISJUNCTION_INTRO = "disjunction_intro"
    DOUBLE_NEGATION_ELIM = "double_negation_elim"
    CONSTRUCTIVE_DILEMMA = "constructive_dilemma"
    BICONDITIONAL_INTRO = "biconditional_intro"
    BICONDITIONAL_ELIM = "biconditional_elim"
    ABSORPTION = "absorption"

    # First-order logic patterns (1st order)
    UNIVERSAL_INSTANTIATION = "universal_instantiation"
    UNIVERSAL_MODUS_PONENS = "universal_modus_ponens"
    EXISTENTIAL_GENERALIZATION = "existential_generalization"
    UNIVERSAL_SYLLOGISM = "universal_syllogism"
    UNIVERSAL_CONTRAPOSITION = "universal_contraposition"
    EXISTENTIAL_SYLLOGISM = "existential_syllogism"


# Separate lists for propositional and FOL patterns
PROPOSITIONAL_PATTERNS = [
    InferencePattern.MODUS_PONENS,
    InferencePattern.MODUS_TOLLENS,
    InferencePattern.HYPOTHETICAL_SYLLOGISM,
    InferencePattern.DISJUNCTIVE_SYLLOGISM,
    InferencePattern.CONJUNCTION_INTRO,
    InferencePattern.CONJUNCTION_ELIM,
    InferencePattern.DISJUNCTION_INTRO,
    InferencePattern.DOUBLE_NEGATION_ELIM,
    InferencePattern.CONSTRUCTIVE_DILEMMA,
    InferencePattern.BICONDITIONAL_INTRO,
    InferencePattern.BICONDITIONAL_ELIM,
    InferencePattern.ABSORPTION,
]

FOL_PATTERNS = [
    InferencePattern.UNIVERSAL_INSTANTIATION,
    InferencePattern.UNIVERSAL_MODUS_PONENS,
    InferencePattern.EXISTENTIAL_GENERALIZATION,
    InferencePattern.UNIVERSAL_SYLLOGISM,
    InferencePattern.UNIVERSAL_CONTRAPOSITION,
    InferencePattern.EXISTENTIAL_SYLLOGISM,
]


@dataclass
class Inference:
    """A valid logical inference with premises and conclusion."""
    premises: List[FormulaNode]
    conclusion: FormulaNode
    pattern: InferencePattern
    formal_notation: str  # e.g., "P→Q, P ⊢ Q"
    logic_order: LogicOrder = LogicOrder.PROPOSITIONAL

    def to_formal(self) -> str:
        """Get full formal representation."""
        premises_str = ", ".join(p.to_formal() for p in self.premises)
        return f"{premises_str} ⊢ {self.conclusion.to_formal()}"

    def is_fol(self) -> bool:
        """Check if this is a first-order logic inference."""
        return self.logic_order == LogicOrder.FIRST_ORDER


@dataclass
class InferenceGeneratorConfig:
    """Configuration for inference generation."""
    min_subformula_depth: int = 1
    max_subformula_depth: int = 3
    logic_order: LogicOrder = LogicOrder.PROPOSITIONAL
    patterns: Optional[List[InferencePattern]] = None  # None = all patterns

    # Tree generator config for subformulas
    tree_config: Optional[TreeGeneratorConfig] = None


class InferenceGenerator:
    """Generate valid logical inferences by instantiating patterns."""

    # Variable and predicate pools for FOL
    VARIABLES = ["x", "y", "z", "w"]
    PREDICATES = ["P", "Q", "R", "S", "T"]
    CONSTANTS = ["a", "b", "c", "d"]

    def __init__(self, config: Optional[InferenceGeneratorConfig] = None):
        self.config = config or InferenceGeneratorConfig()

        # Setup tree generator for subformulas
        tree_config = self.config.tree_config or TreeGeneratorConfig(
            min_depth=self.config.min_subformula_depth,
            max_depth=self.config.max_subformula_depth,
            logic_order=self.config.logic_order
        )
        self.tree_generator = RandomTreeGenerator(tree_config)

        # Available patterns based on logic order
        if self.config.patterns:
            self.available_patterns = self.config.patterns
        elif self.config.logic_order == LogicOrder.FIRST_ORDER:
            self.available_patterns = FOL_PATTERNS
        else:
            self.available_patterns = PROPOSITIONAL_PATTERNS

        # Pattern generators - propositional
        self._pattern_generators = {
            InferencePattern.MODUS_PONENS: self._gen_modus_ponens,
            InferencePattern.MODUS_TOLLENS: self._gen_modus_tollens,
            InferencePattern.HYPOTHETICAL_SYLLOGISM: self._gen_hypothetical_syllogism,
            InferencePattern.DISJUNCTIVE_SYLLOGISM: self._gen_disjunctive_syllogism,
            InferencePattern.CONJUNCTION_INTRO: self._gen_conjunction_intro,
            InferencePattern.CONJUNCTION_ELIM: self._gen_conjunction_elim,
            InferencePattern.DISJUNCTION_INTRO: self._gen_disjunction_intro,
            InferencePattern.DOUBLE_NEGATION_ELIM: self._gen_double_negation_elim,
            InferencePattern.CONSTRUCTIVE_DILEMMA: self._gen_constructive_dilemma,
            InferencePattern.BICONDITIONAL_INTRO: self._gen_biconditional_intro,
            InferencePattern.BICONDITIONAL_ELIM: self._gen_biconditional_elim,
            InferencePattern.ABSORPTION: self._gen_absorption,
            # FOL patterns
            InferencePattern.UNIVERSAL_INSTANTIATION: self._gen_universal_instantiation,
            InferencePattern.UNIVERSAL_MODUS_PONENS: self._gen_universal_modus_ponens,
            InferencePattern.EXISTENTIAL_GENERALIZATION: self._gen_existential_generalization,
            InferencePattern.UNIVERSAL_SYLLOGISM: self._gen_universal_syllogism,
            InferencePattern.UNIVERSAL_CONTRAPOSITION: self._gen_universal_contraposition,
            InferencePattern.EXISTENTIAL_SYLLOGISM: self._gen_existential_syllogism,
        }

    def generate(self, pattern: Optional[InferencePattern] = None) -> Inference:
        """
        Generate a valid inference.

        Args:
            pattern: Specific pattern to use, or None for random

        Returns:
            Inference object with premises and conclusion
        """
        if pattern is None:
            pattern = random.choice(self.available_patterns)

        generator = self._pattern_generators.get(pattern)
        if generator is None:
            raise ValueError(f"Unknown pattern: {pattern}")

        return generator()

    def generate_batch(self, count: int) -> List[Inference]:
        """Generate multiple inferences."""
        return [self.generate() for _ in range(count)]

    def _gen_subformula(self, depth: Optional[int] = None) -> FormulaNode:
        """Generate a random subformula."""
        if depth is None:
            depth = random.randint(
                self.config.min_subformula_depth,
                self.config.max_subformula_depth
            )
        return self.tree_generator.generate(target_depth=depth)

    # ============ Pattern Generators ============

    def _gen_modus_ponens(self) -> Inference:
        """
        Modus Ponens: P→Q, P ⊢ Q

        If P implies Q, and P is true, then Q is true.
        """
        p = self._gen_subformula()
        q = self._gen_subformula()

        implication = make_implication(p, q)

        return Inference(
            premises=[implication, p.copy()],
            conclusion=q.copy(),
            pattern=InferencePattern.MODUS_PONENS,
            formal_notation="P→Q, P ⊢ Q"
        )

    def _gen_modus_tollens(self) -> Inference:
        """
        Modus Tollens: P→Q, ¬Q ⊢ ¬P

        If P implies Q, and Q is false, then P is false.
        """
        p = self._gen_subformula()
        q = self._gen_subformula()

        implication = make_implication(p, q)
        not_q = negate(q)
        not_p = negate(p)

        return Inference(
            premises=[implication, not_q],
            conclusion=not_p,
            pattern=InferencePattern.MODUS_TOLLENS,
            formal_notation="P→Q, ¬Q ⊢ ¬P"
        )

    def _gen_hypothetical_syllogism(self) -> Inference:
        """
        Hypothetical Syllogism: P→Q, Q→R ⊢ P→R

        Chain of implications.
        """
        p = self._gen_subformula()
        q = self._gen_subformula()
        r = self._gen_subformula()

        p_implies_q = make_implication(p, q)
        q_implies_r = make_implication(q, r)
        p_implies_r = make_implication(p, r)

        return Inference(
            premises=[p_implies_q, q_implies_r],
            conclusion=p_implies_r,
            pattern=InferencePattern.HYPOTHETICAL_SYLLOGISM,
            formal_notation="P→Q, Q→R ⊢ P→R"
        )

    def _gen_disjunctive_syllogism(self) -> Inference:
        """
        Disjunctive Syllogism: P∨Q, ¬P ⊢ Q

        If P or Q, and not P, then Q.
        """
        p = self._gen_subformula()
        q = self._gen_subformula()

        disjunction = make_disjunction(p, q)
        not_p = negate(p)

        return Inference(
            premises=[disjunction, not_p],
            conclusion=q.copy(),
            pattern=InferencePattern.DISJUNCTIVE_SYLLOGISM,
            formal_notation="P∨Q, ¬P ⊢ Q"
        )

    def _gen_conjunction_intro(self) -> Inference:
        """
        Conjunction Introduction: P, Q ⊢ P∧Q

        If both P and Q are true, then P and Q.
        """
        p = self._gen_subformula()
        q = self._gen_subformula()

        conjunction = make_conjunction(p, q)

        return Inference(
            premises=[p.copy(), q.copy()],
            conclusion=conjunction,
            pattern=InferencePattern.CONJUNCTION_INTRO,
            formal_notation="P, Q ⊢ P∧Q"
        )

    def _gen_conjunction_elim(self) -> Inference:
        """
        Conjunction Elimination: P∧Q ⊢ P (or Q)

        If P and Q, then P.
        """
        p = self._gen_subformula()
        q = self._gen_subformula()

        conjunction = make_conjunction(p, q)

        # Randomly choose left or right elimination
        conclusion = p.copy() if random.random() < 0.5 else q.copy()

        return Inference(
            premises=[conjunction],
            conclusion=conclusion,
            pattern=InferencePattern.CONJUNCTION_ELIM,
            formal_notation="P∧Q ⊢ P"
        )

    def _gen_disjunction_intro(self) -> Inference:
        """
        Disjunction Introduction: P, Q ⊢ P∨Q

        If P and Q are both true, then P or Q.

        Note: Changed from standard P ⊢ P∨Q to ensure all atoms in conclusion
        appear in premises, avoiding ambiguity about which Q to use.
        """
        p = self._gen_subformula()
        q = self._gen_subformula()

        disjunction = make_disjunction(p, q)

        return Inference(
            premises=[p.copy(), q.copy()],
            conclusion=disjunction,
            pattern=InferencePattern.DISJUNCTION_INTRO,
            formal_notation="P, Q ⊢ P∨Q"
        )

    def _gen_double_negation_elim(self) -> Inference:
        """
        Double Negation Elimination: ¬¬P ⊢ P

        If not not P, then P.
        """
        p = self._gen_subformula()

        double_neg = NegationNode(child=NegationNode(child=p.copy()))

        return Inference(
            premises=[double_neg],
            conclusion=p.copy(),
            pattern=InferencePattern.DOUBLE_NEGATION_ELIM,
            formal_notation="¬¬P ⊢ P"
        )

    def _gen_constructive_dilemma(self) -> Inference:
        """
        Constructive Dilemma: (P→Q)∧(R→S), P∨R ⊢ Q∨S

        Complex dilemma with two implications.
        """
        p = self._gen_subformula()
        q = self._gen_subformula()
        r = self._gen_subformula()
        s = self._gen_subformula()

        p_implies_q = make_implication(p, q)
        r_implies_s = make_implication(r, s)
        implications = make_conjunction(p_implies_q, r_implies_s)
        p_or_r = make_disjunction(p, r)
        q_or_s = make_disjunction(q, s)

        return Inference(
            premises=[implications, p_or_r],
            conclusion=q_or_s,
            pattern=InferencePattern.CONSTRUCTIVE_DILEMMA,
            formal_notation="(P→Q)∧(R→S), P∨R ⊢ Q∨S"
        )

    def _gen_biconditional_intro(self) -> Inference:
        """
        Biconditional Introduction: P→Q, Q→P ⊢ P↔Q

        Bidirectional implication introduction.
        """
        p = self._gen_subformula()
        q = self._gen_subformula()

        p_implies_q = make_implication(p, q)
        q_implies_p = make_implication(q, p)
        biconditional = make_biconditional(p, q)

        return Inference(
            premises=[p_implies_q, q_implies_p],
            conclusion=biconditional,
            pattern=InferencePattern.BICONDITIONAL_INTRO,
            formal_notation="P→Q, Q→P ⊢ P↔Q"
        )

    def _gen_biconditional_elim(self) -> Inference:
        """
        Biconditional Elimination: P↔Q, P ⊢ Q (or P↔Q, Q ⊢ P)

        From biconditional and one side, derive the other.
        """
        p = self._gen_subformula()
        q = self._gen_subformula()

        biconditional = make_biconditional(p, q)

        # Randomly choose direction
        if random.random() < 0.5:
            premise2 = p.copy()
            conclusion = q.copy()
        else:
            premise2 = q.copy()
            conclusion = p.copy()

        return Inference(
            premises=[biconditional, premise2],
            conclusion=conclusion,
            pattern=InferencePattern.BICONDITIONAL_ELIM,
            formal_notation="P↔Q, P ⊢ Q"
        )

    def _gen_absorption(self) -> Inference:
        """
        Absorption: P→Q ⊢ P→(P∧Q)

        If P implies Q, then P implies (P and Q).
        """
        p = self._gen_subformula()
        q = self._gen_subformula()

        p_implies_q = make_implication(p, q)
        p_and_q = make_conjunction(p, q)
        p_implies_p_and_q = make_implication(p, p_and_q)

        return Inference(
            premises=[p_implies_q],
            conclusion=p_implies_p_and_q,
            pattern=InferencePattern.ABSORPTION,
            formal_notation="P→Q ⊢ P→(P∧Q)"
        )

    # ============ FOL Pattern Generators ============

    def _gen_predicate(self, name: str, variables: List[str]) -> AtomNode:
        """Create a predicate atom like P(x) or R(x,y)."""
        return AtomNode(
            identifier=name,
            variables=variables,
            is_predicate=True
        )

    def _gen_universal_instantiation(self) -> Inference:
        """
        Universal Instantiation: ∀x.P(x), Q(a) ⊢ P(a)

        All things have property P. Thing 'a' has property Q. Therefore, 'a' has property P.
        Example: All birds fly. Tweety is yellow. Therefore, Tweety flies.

        Note: Added Q(a) premise to ensure constant 'a' appears in premises,
        avoiding ambiguity about which entity to instantiate.
        """
        var = random.choice(self.VARIABLES)
        const = random.choice(self.CONSTANTS)
        pred_p, pred_q = random.sample(self.PREDICATES, 2)

        # ∀x.P(x)
        p_of_x = self._gen_predicate(pred_p, [var])
        forall_p = QuantifiedNode(
            quantifier=Quantifier.FORALL,
            variable=var,
            body=p_of_x
        )

        # Q(a) - introduces the constant 'a' in premises
        q_of_a = self._gen_predicate(pred_q, [const])

        # P(a)
        p_of_a = self._gen_predicate(pred_p, [const])

        return Inference(
            premises=[forall_p, q_of_a],
            conclusion=p_of_a,
            pattern=InferencePattern.UNIVERSAL_INSTANTIATION,
            formal_notation=f"∀{var}.{pred_p}({var}), {pred_q}({const}) ⊢ {pred_p}({const})",
            logic_order=LogicOrder.FIRST_ORDER
        )

    def _gen_universal_modus_ponens(self) -> Inference:
        """
        Universal Modus Ponens: ∀x.(P(x)→Q(x)), P(a) ⊢ Q(a)

        For all things, if P then Q. Thing 'a' has P. Therefore, 'a' has Q.
        Example: All humans are mortal. Socrates is human. Therefore, Socrates is mortal.
        """
        var = random.choice(self.VARIABLES)
        const = random.choice(self.CONSTANTS)
        pred_p, pred_q = random.sample(self.PREDICATES, 2)

        # P(x) → Q(x)
        p_of_x = self._gen_predicate(pred_p, [var])
        q_of_x = self._gen_predicate(pred_q, [var])
        implication = make_implication(p_of_x, q_of_x)

        # ∀x.(P(x) → Q(x))
        forall_impl = QuantifiedNode(
            quantifier=Quantifier.FORALL,
            variable=var,
            body=implication
        )

        # P(a)
        p_of_a = self._gen_predicate(pred_p, [const])

        # Q(a)
        q_of_a = self._gen_predicate(pred_q, [const])

        return Inference(
            premises=[forall_impl, p_of_a],
            conclusion=q_of_a,
            pattern=InferencePattern.UNIVERSAL_MODUS_PONENS,
            formal_notation=f"∀{var}.({pred_p}({var})→{pred_q}({var})), {pred_p}({const}) ⊢ {pred_q}({const})",
            logic_order=LogicOrder.FIRST_ORDER
        )

    def _gen_existential_generalization(self) -> Inference:
        """
        Existential Generalization: P(a) ⊢ ∃x.P(x)

        Thing 'a' has property P. Therefore, something has property P.
        Example: Tweety flies. Therefore, something flies.
        """
        var = random.choice(self.VARIABLES)
        const = random.choice(self.CONSTANTS)
        pred = random.choice(self.PREDICATES)

        # P(a)
        p_of_a = self._gen_predicate(pred, [const])

        # ∃x.P(x)
        p_of_x = self._gen_predicate(pred, [var])
        exists_p = QuantifiedNode(
            quantifier=Quantifier.EXISTS,
            variable=var,
            body=p_of_x
        )

        return Inference(
            premises=[p_of_a],
            conclusion=exists_p,
            pattern=InferencePattern.EXISTENTIAL_GENERALIZATION,
            formal_notation=f"{pred}({const}) ⊢ ∃{var}.{pred}({var})",
            logic_order=LogicOrder.FIRST_ORDER
        )

    def _gen_universal_syllogism(self) -> Inference:
        """
        Universal Syllogism: ∀x.(P(x)→Q(x)), ∀x.(Q(x)→R(x)) ⊢ ∀x.(P(x)→R(x))

        Chain of universal implications.
        Example: All humans are mortal. All mortals will die. Therefore, all humans will die.
        """
        var = random.choice(self.VARIABLES)
        pred_p, pred_q, pred_r = random.sample(self.PREDICATES, 3)

        # ∀x.(P(x) → Q(x))
        p_of_x = self._gen_predicate(pred_p, [var])
        q_of_x = self._gen_predicate(pred_q, [var])
        r_of_x = self._gen_predicate(pred_r, [var])

        p_implies_q = make_implication(p_of_x, q_of_x)
        forall_pq = QuantifiedNode(
            quantifier=Quantifier.FORALL,
            variable=var,
            body=p_implies_q
        )

        # ∀x.(Q(x) → R(x))
        q_of_x2 = self._gen_predicate(pred_q, [var])
        r_of_x2 = self._gen_predicate(pred_r, [var])
        q_implies_r = make_implication(q_of_x2, r_of_x2)
        forall_qr = QuantifiedNode(
            quantifier=Quantifier.FORALL,
            variable=var,
            body=q_implies_r
        )

        # ∀x.(P(x) → R(x))
        p_of_x3 = self._gen_predicate(pred_p, [var])
        r_of_x3 = self._gen_predicate(pred_r, [var])
        p_implies_r = make_implication(p_of_x3, r_of_x3)
        forall_pr = QuantifiedNode(
            quantifier=Quantifier.FORALL,
            variable=var,
            body=p_implies_r
        )

        return Inference(
            premises=[forall_pq, forall_qr],
            conclusion=forall_pr,
            pattern=InferencePattern.UNIVERSAL_SYLLOGISM,
            formal_notation=f"∀{var}.({pred_p}({var})→{pred_q}({var})), ∀{var}.({pred_q}({var})→{pred_r}({var})) ⊢ ∀{var}.({pred_p}({var})→{pred_r}({var}))",
            logic_order=LogicOrder.FIRST_ORDER
        )

    def _gen_universal_contraposition(self) -> Inference:
        """
        Universal Contraposition: ∀x.(P(x)→Q(x)), ¬Q(a) ⊢ ¬P(a)

        All P's are Q's. Thing 'a' is not Q. Therefore, 'a' is not P.
        Example: All birds fly. Tweety doesn't fly. Therefore, Tweety is not a bird.
        """
        var = random.choice(self.VARIABLES)
        const = random.choice(self.CONSTANTS)
        pred_p, pred_q = random.sample(self.PREDICATES, 2)

        # ∀x.(P(x) → Q(x))
        p_of_x = self._gen_predicate(pred_p, [var])
        q_of_x = self._gen_predicate(pred_q, [var])
        implication = make_implication(p_of_x, q_of_x)
        forall_impl = QuantifiedNode(
            quantifier=Quantifier.FORALL,
            variable=var,
            body=implication
        )

        # ¬Q(a)
        q_of_a = self._gen_predicate(pred_q, [const])
        not_q_of_a = negate(q_of_a)

        # ¬P(a)
        p_of_a = self._gen_predicate(pred_p, [const])
        not_p_of_a = negate(p_of_a)

        return Inference(
            premises=[forall_impl, not_q_of_a],
            conclusion=not_p_of_a,
            pattern=InferencePattern.UNIVERSAL_CONTRAPOSITION,
            formal_notation=f"∀{var}.({pred_p}({var})→{pred_q}({var})), ¬{pred_q}({const}) ⊢ ¬{pred_p}({const})",
            logic_order=LogicOrder.FIRST_ORDER
        )

    def _gen_existential_syllogism(self) -> Inference:
        """
        Existential Syllogism: ∃x.P(x), ∀x.(P(x)→Q(x)) ⊢ ∃x.Q(x)

        Something is P. All P's are Q's. Therefore, something is Q.
        Example: Something is red. All red things are visible. Therefore, something is visible.
        """
        var = random.choice(self.VARIABLES)
        pred_p, pred_q = random.sample(self.PREDICATES, 2)

        # ∃x.P(x)
        p_of_x = self._gen_predicate(pred_p, [var])
        exists_p = QuantifiedNode(
            quantifier=Quantifier.EXISTS,
            variable=var,
            body=p_of_x
        )

        # ∀x.(P(x) → Q(x))
        p_of_x2 = self._gen_predicate(pred_p, [var])
        q_of_x = self._gen_predicate(pred_q, [var])
        implication = make_implication(p_of_x2, q_of_x)
        forall_impl = QuantifiedNode(
            quantifier=Quantifier.FORALL,
            variable=var,
            body=implication
        )

        # ∃x.Q(x)
        q_of_x2 = self._gen_predicate(pred_q, [var])
        exists_q = QuantifiedNode(
            quantifier=Quantifier.EXISTS,
            variable=var,
            body=q_of_x2
        )

        return Inference(
            premises=[exists_p, forall_impl],
            conclusion=exists_q,
            pattern=InferencePattern.EXISTENTIAL_SYLLOGISM,
            formal_notation=f"∃{var}.{pred_p}({var}), ∀{var}.({pred_p}({var})→{pred_q}({var})) ⊢ ∃{var}.{pred_q}({var})",
            logic_order=LogicOrder.FIRST_ORDER
        )


# Convenience functions

def generate_inference(
    pattern: Optional[InferencePattern] = None,
    min_depth: int = 1,
    max_depth: int = 3,
    logic_order: LogicOrder = LogicOrder.PROPOSITIONAL
) -> Inference:
    """Generate a single inference with default settings."""
    config = InferenceGeneratorConfig(
        min_subformula_depth=min_depth,
        max_subformula_depth=max_depth,
        logic_order=logic_order
    )
    generator = InferenceGenerator(config)
    return generator.generate(pattern)


def generate_inferences(
    count: int,
    patterns: Optional[List[InferencePattern]] = None,
    min_depth: int = 1,
    max_depth: int = 3,
    logic_order: LogicOrder = LogicOrder.PROPOSITIONAL
) -> List[Inference]:
    """Generate multiple inferences."""
    config = InferenceGeneratorConfig(
        min_subformula_depth=min_depth,
        max_subformula_depth=max_depth,
        logic_order=logic_order,
        patterns=patterns
    )
    generator = InferenceGenerator(config)
    return generator.generate_batch(count)
