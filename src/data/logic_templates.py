"""Logic formula templates for propositional and first-order logic."""

from typing import List, Optional
from dataclasses import dataclass, field
from enum import Enum
import random


class LogicType(Enum):
    """Type of logic formula."""
    PROPOSITIONAL = "propositional"  # 0th order
    FIRST_ORDER = "first_order"      # 1st order


@dataclass
class LogicTemplate:
    """A logic template with placeholders for atomic propositions."""
    name: str
    logic_type: LogicType
    pattern: str  # e.g., "If {P}, then {Q}. {P}. Therefore, {Q}."
    formal_notation: str  # e.g., "(P -> Q) ^ P |- Q"
    premise_slots: List[str]  # ["P", "Q"]
    conclusion_slot: str  # "Q"
    num_entities: int = 0  # For first-order logic
    entity_slots: List[str] = field(default_factory=list)  # ["a", "b"]
    description: str = ""


# ============ PROPOSITIONAL LOGIC TEMPLATES (0th Order) ============

PROPOSITIONAL_TEMPLATES = [
    # Modus Ponens: P -> Q, P |- Q
    LogicTemplate(
        name="modus_ponens",
        logic_type=LogicType.PROPOSITIONAL,
        pattern="If {P}, then {Q}. {P}. Therefore, {Q}.",
        formal_notation="(P -> Q) ^ P |- Q",
        premise_slots=["P", "Q"],
        conclusion_slot="Q",
        description="If P implies Q, and P is true, then Q is true"
    ),

    # Modus Tollens: P -> Q, ~Q |- ~P
    LogicTemplate(
        name="modus_tollens",
        logic_type=LogicType.PROPOSITIONAL,
        pattern="If {P}, then {Q}. It is not the case that {Q}. Therefore, it is not the case that {P}.",
        formal_notation="(P -> Q) ^ ~Q |- ~P",
        premise_slots=["P", "Q"],
        conclusion_slot="not_P",
        description="If P implies Q, and Q is false, then P is false"
    ),

    # Hypothetical Syllogism: P -> Q, Q -> R |- P -> R
    LogicTemplate(
        name="hypothetical_syllogism",
        logic_type=LogicType.PROPOSITIONAL,
        pattern="If {P}, then {Q}. If {Q}, then {R}. Therefore, if {P}, then {R}.",
        formal_notation="(P -> Q) ^ (Q -> R) |- P -> R",
        premise_slots=["P", "Q", "R"],
        conclusion_slot="P_implies_R",
        description="Chain of implications"
    ),

    # Disjunctive Syllogism: P v Q, ~P |- Q
    LogicTemplate(
        name="disjunctive_syllogism",
        logic_type=LogicType.PROPOSITIONAL,
        pattern="Either {P} or {Q}. It is not the case that {P}. Therefore, {Q}.",
        formal_notation="(P v Q) ^ ~P |- Q",
        premise_slots=["P", "Q"],
        conclusion_slot="Q",
        description="If P or Q, and not P, then Q"
    ),

    # Conjunction: P, Q |- P ^ Q
    LogicTemplate(
        name="conjunction",
        logic_type=LogicType.PROPOSITIONAL,
        pattern="{P}. {Q}. Therefore, both {P} and {Q}.",
        formal_notation="P, Q |- P ^ Q",
        premise_slots=["P", "Q"],
        conclusion_slot="P_and_Q",
        description="If P and Q are both true, then P and Q"
    ),

    # Simplification: P ^ Q |- P
    LogicTemplate(
        name="simplification",
        logic_type=LogicType.PROPOSITIONAL,
        pattern="Both {P} and {Q}. Therefore, {P}.",
        formal_notation="P ^ Q |- P",
        premise_slots=["P", "Q"],
        conclusion_slot="P",
        description="If P and Q, then P"
    ),

    # Addition: P |- P v Q
    LogicTemplate(
        name="addition",
        logic_type=LogicType.PROPOSITIONAL,
        pattern="{P}. Therefore, either {P} or {Q}.",
        formal_notation="P |- P v Q",
        premise_slots=["P", "Q"],
        conclusion_slot="P_or_Q",
        description="If P, then P or Q (for any Q)"
    ),

    # Constructive Dilemma: (P -> Q) ^ (R -> S), P v R |- Q v S
    LogicTemplate(
        name="constructive_dilemma",
        logic_type=LogicType.PROPOSITIONAL,
        pattern="If {P}, then {Q}. If {R}, then {S}. Either {P} or {R}. Therefore, either {Q} or {S}.",
        formal_notation="(P -> Q) ^ (R -> S) ^ (P v R) |- Q v S",
        premise_slots=["P", "Q", "R", "S"],
        conclusion_slot="Q_or_S",
        description="Complex dilemma with two implications"
    ),

    # Biconditional Introduction: (P -> Q) ^ (Q -> P) |- P <-> Q
    LogicTemplate(
        name="biconditional_intro",
        logic_type=LogicType.PROPOSITIONAL,
        pattern="If {P}, then {Q}. If {Q}, then {P}. Therefore, {P} if and only if {Q}.",
        formal_notation="(P -> Q) ^ (Q -> P) |- P <-> Q",
        premise_slots=["P", "Q"],
        conclusion_slot="P_iff_Q",
        description="Bidirectional implication"
    ),
]


# ============ FIRST-ORDER LOGIC TEMPLATES (1st Order) ============

FIRST_ORDER_TEMPLATES = [
    # Universal Instantiation: forall x. P(x) |- P(a)
    LogicTemplate(
        name="universal_instantiation",
        logic_type=LogicType.FIRST_ORDER,
        pattern="All {X} are {P}. {a} is a {X}. Therefore, {a} is {P}.",
        formal_notation="forall x. P(x), a in X |- P(a)",
        premise_slots=["X", "P"],
        conclusion_slot="P_of_a",
        num_entities=1,
        entity_slots=["a"],
        description="Universal rule applied to specific entity"
    ),

    # Universal Modus Ponens: forall x. (P(x) -> Q(x)), P(a) |- Q(a)
    LogicTemplate(
        name="universal_modus_ponens",
        logic_type=LogicType.FIRST_ORDER,
        pattern="For all things, if something is {P}, then it is {Q}. {a} is {P}. Therefore, {a} is {Q}.",
        formal_notation="forall x. (P(x) -> Q(x)) ^ P(a) |- Q(a)",
        premise_slots=["P", "Q"],
        conclusion_slot="Q_of_a",
        num_entities=1,
        entity_slots=["a"],
        description="Universal implication with instantiation"
    ),

    # Existential Generalization: P(a) |- exists x. P(x)
    LogicTemplate(
        name="existential_generalization",
        logic_type=LogicType.FIRST_ORDER,
        pattern="{a} is {P}. Therefore, there exists something that is {P}.",
        formal_notation="P(a) |- exists x. P(x)",
        premise_slots=["P"],
        conclusion_slot="exists_P",
        num_entities=1,
        entity_slots=["a"],
        description="From specific to existential"
    ),

    # Syllogism with Universals: forall x.(P(x)->Q(x)), forall x.(Q(x)->R(x)) |- forall x.(P(x)->R(x))
    LogicTemplate(
        name="universal_chain",
        logic_type=LogicType.FIRST_ORDER,
        pattern="All things that are {P} are also {Q}. All things that are {Q} are also {R}. Therefore, all things that are {P} are also {R}.",
        formal_notation="forall x.(P(x)->Q(x)) ^ forall x.(Q(x)->R(x)) |- forall x.(P(x)->R(x))",
        premise_slots=["P", "Q", "R"],
        conclusion_slot="P_implies_R",
        num_entities=0,
        entity_slots=[],
        description="Chain of universal implications"
    ),

    # Contraposition with Universal: forall x.(P(x)->Q(x)), ~Q(a) |- ~P(a)
    LogicTemplate(
        name="universal_contraposition",
        logic_type=LogicType.FIRST_ORDER,
        pattern="All things that are {P} are also {Q}. {a} is not {Q}. Therefore, {a} is not {P}.",
        formal_notation="forall x.(P(x)->Q(x)) ^ ~Q(a) |- ~P(a)",
        premise_slots=["P", "Q"],
        conclusion_slot="not_P_of_a",
        num_entities=1,
        entity_slots=["a"],
        description="Contraposition with specific entity"
    ),

    # Disjunctive Syllogism with Predicates: (P(a) v Q(a)), ~P(a) |- Q(a)
    LogicTemplate(
        name="predicate_disjunctive_syllogism",
        logic_type=LogicType.FIRST_ORDER,
        pattern="{a} is either {P} or {Q}. {a} is not {P}. Therefore, {a} is {Q}.",
        formal_notation="(P(a) v Q(a)) ^ ~P(a) |- Q(a)",
        premise_slots=["P", "Q"],
        conclusion_slot="Q_of_a",
        num_entities=1,
        entity_slots=["a"],
        description="Disjunctive syllogism for predicates"
    ),

    # Conjunction of Predicates: P(a) ^ Q(a) |- exists x.(P(x) ^ Q(x))
    LogicTemplate(
        name="predicate_conjunction_exists",
        logic_type=LogicType.FIRST_ORDER,
        pattern="{a} is {P}. {a} is also {Q}. Therefore, there exists something that is both {P} and {Q}.",
        formal_notation="P(a) ^ Q(a) |- exists x.(P(x) ^ Q(x))",
        premise_slots=["P", "Q"],
        conclusion_slot="exists_P_and_Q",
        num_entities=1,
        entity_slots=["a"],
        description="From conjunction to existential"
    ),

    # Two-entity relation: R(a,b), forall x,y.(R(x,y)->S(x)) |- S(a)
    LogicTemplate(
        name="relational_inference",
        logic_type=LogicType.FIRST_ORDER,
        pattern="{a} {R} {b}. For all things, if something {R} another thing, then the first thing is {S}. Therefore, {a} is {S}.",
        formal_notation="R(a,b) ^ forall x,y.(R(x,y)->S(x)) |- S(a)",
        premise_slots=["R", "S"],
        conclusion_slot="S_of_a",
        num_entities=2,
        entity_slots=["a", "b"],
        description="Relational reasoning with two entities"
    ),
]


def get_all_templates() -> List[LogicTemplate]:
    """Get all logic templates."""
    return PROPOSITIONAL_TEMPLATES + FIRST_ORDER_TEMPLATES


def get_templates_by_type(logic_type: LogicType) -> List[LogicTemplate]:
    """Get templates filtered by logic type."""
    if logic_type == LogicType.PROPOSITIONAL:
        return PROPOSITIONAL_TEMPLATES.copy()
    else:
        return FIRST_ORDER_TEMPLATES.copy()


def sample_template(
    logic_type: Optional[LogicType] = None,
    exclude_names: Optional[List[str]] = None
) -> LogicTemplate:
    """Sample a random template, optionally filtered by type."""
    if logic_type:
        templates = get_templates_by_type(logic_type)
    else:
        templates = get_all_templates()

    if exclude_names:
        templates = [t for t in templates if t.name not in exclude_names]

    return random.choice(templates)


def get_template_by_name(name: str) -> Optional[LogicTemplate]:
    """Get a specific template by name."""
    for template in get_all_templates():
        if template.name == name:
            return template
    return None
