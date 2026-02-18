"""
Recursive descent parser for the semi-formal curly-bracket language.

Converts natural language text produced by the NL renderer back into
FormulaNode ASTs. The grammar uses curly brackets {} to delimit compound
formulas and fixed keyword patterns for logical connectives:

    {if P, then Q}          -> IMPLIES
    {it is not the case that P} -> NOT
    {P and Q}                -> AND
    {P or Q}                 -> OR
    {P if and only if Q}     -> IFF
    {for all x, BODY}        -> FORALL
    {there exist x such that BODY} -> EXISTS
    plain text               -> AtomNode
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set

from ..data.syntax_tree import (
    FormulaNode,
    AtomNode,
    NegationNode,
    BinaryNode,
    QuantifiedNode,
    Connective,
    Quantifier,
)


@dataclass
class ParseResult:
    """Result of parsing an inference text."""

    premises: List[FormulaNode] = field(default_factory=list)
    conclusion: Optional[FormulaNode] = None
    raw_premises: List[str] = field(default_factory=list)
    raw_conclusion: Optional[str] = None
    errors: List[str] = field(default_factory=list)


class SemiFormalParser:
    """
    Recursive descent parser that converts the project's semi-formal
    curly-bracket language into FormulaNode ASTs.

    The parser handles full inference texts of the form::

        {if P, then Q}. P. Therefore, Q.

    as well as individual formulas like ``{P and Q}`` or ``it is raining``.
    """

    def __init__(self) -> None:
        self._bound_vars: Set[str] = set()
        self._is_fol: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_text(self, text: str) -> ParseResult:
        """Parse a full inference text into premises and conclusion.

        The text is expected to follow the NL renderer format::

            Premise1. Premise2. Therefore, conclusion.

        Parameters
        ----------
        text:
            The full inference text to parse.

        Returns
        -------
        ParseResult
            Structured parse result with premises, conclusion, raw strings
            and any errors encountered.
        """
        text = text.strip()

        # Determine FOL mode from the entire text
        text_lower = text.lower()
        self._is_fol = "for all " in text_lower or "there exist " in text_lower

        # Split on "Therefore," (case-sensitive) into premise section and
        # conclusion section.
        parts = text.split("Therefore,", 1)
        premise_section = parts[0]
        conclusion_section = parts[1] if len(parts) > 1 else None

        # Split premises on "." and strip; filter empties.
        raw_premises: List[str] = []
        for segment in premise_section.split("."):
            segment = segment.strip()
            if segment:
                raw_premises.append(segment)

        # Prepare raw conclusion
        raw_conclusion: Optional[str] = None
        if conclusion_section is not None:
            raw_conclusion = conclusion_section.strip().rstrip(".")
            if raw_conclusion == "":
                raw_conclusion = None

        # Parse each premise
        result = ParseResult(
            raw_premises=raw_premises,
            raw_conclusion=raw_conclusion,
        )

        for raw in raw_premises:
            self._bound_vars = set()
            try:
                node = self._parse_formula_inner(raw)
                result.premises.append(node)
            except Exception as exc:  # noqa: BLE001
                result.errors.append(f"Error parsing premise '{raw}': {exc}")

        # Parse conclusion
        if raw_conclusion is not None:
            self._bound_vars = set()
            try:
                result.conclusion = self._parse_formula_inner(raw_conclusion)
            except Exception as exc:  # noqa: BLE001
                result.errors.append(
                    f"Error parsing conclusion '{raw_conclusion}': {exc}"
                )

        return result

    def parse_inference(
        self, premises: List[str], conclusion: str
    ) -> ParseResult:
        """Parse pre-split premises and conclusion strings.

        Parameters
        ----------
        premises:
            List of premise strings (already separated).
        conclusion:
            The conclusion string.

        Returns
        -------
        ParseResult
        """
        # Determine FOL mode
        all_texts = premises + [conclusion]
        all_lower = " ".join(all_texts).lower()
        self._is_fol = "for all " in all_lower or "there exist " in all_lower

        result = ParseResult(
            raw_premises=list(premises),
            raw_conclusion=conclusion,
        )

        for raw in premises:
            self._bound_vars = set()
            try:
                node = self._parse_formula_inner(raw)
                result.premises.append(node)
            except Exception as exc:  # noqa: BLE001
                result.errors.append(f"Error parsing premise '{raw}': {exc}")

        self._bound_vars = set()
        try:
            result.conclusion = self._parse_formula_inner(conclusion)
        except Exception as exc:  # noqa: BLE001
            result.errors.append(
                f"Error parsing conclusion '{conclusion}': {exc}"
            )

        return result

    def parse_formula(self, text: str) -> FormulaNode:
        """Parse a single formula string into a FormulaNode AST.

        Parameters
        ----------
        text:
            The formula text (may or may not be wrapped in curly brackets).

        Returns
        -------
        FormulaNode

        Raises
        ------
        ValueError
            If the text cannot be parsed.
        """
        text_lower = text.lower()
        self._is_fol = "for all " in text_lower or "there exist " in text_lower
        self._bound_vars = set()
        return self._parse_formula_inner(text)

    # ------------------------------------------------------------------
    # Core recursive descent
    # ------------------------------------------------------------------

    def _parse_formula_inner(self, text: str) -> FormulaNode:
        """Recursive descent entry point for a single formula."""
        text = text.strip()

        # Strip outer matching curly brackets if the opening bracket at
        # position 0 matches the closing bracket at position len-1.
        if (
            text.startswith("{")
            and text.endswith("}")
            and self._matching_bracket(text, 0) == len(text) - 1
        ):
            text = text[1:-1].strip()

        # The first character may have been uppercased by the renderer.
        # We work on a lowercase copy for keyword detection but preserve
        # the original for atom extraction.
        lower = text.lower()

        # --- IMPLIES: "if ..., then ..." ---
        if lower.startswith("if "):
            sep_idx = self._find_at_depth0(text, ", then ")
            if sep_idx != -1:
                left_text = text[3:sep_idx]  # after "if "
                right_text = text[sep_idx + 7:]  # after ", then "
                left = self._parse_formula_inner(left_text)
                right = self._parse_formula_inner(right_text)
                return BinaryNode(
                    connective=Connective.IMPLIES, left=left, right=right
                )

        # --- NEGATION: "it is not the case that ..." ---
        prefix_not = "it is not the case that "
        if lower.startswith(prefix_not):
            child_text = text[len(prefix_not):]
            child = self._parse_formula_inner(child_text)
            return NegationNode(child=child)

        # --- FORALL: "for all <var>, ..." ---
        if lower.startswith("for all "):
            sep_idx = self._find_at_depth0(text, ", ")
            if sep_idx != -1:
                var = text[8:sep_idx].strip()  # after "for all "
                body_text = text[sep_idx + 2:]  # after ", "
                self._bound_vars.add(var.lower())
                body = self._parse_formula_inner(body_text)
                return QuantifiedNode(
                    quantifier=Quantifier.FORALL,
                    variable=var.lower(),
                    body=body,
                )

        # --- EXISTS: "there exist <var> such that ..." ---
        if lower.startswith("there exist "):
            sep_idx = self._find_at_depth0(text, " such that ")
            if sep_idx != -1:
                var = text[12:sep_idx].strip()  # after "there exist "
                body_text = text[sep_idx + 11:]  # after " such that "
                self._bound_vars.add(var.lower())
                body = self._parse_formula_inner(body_text)
                return QuantifiedNode(
                    quantifier=Quantifier.EXISTS,
                    variable=var.lower(),
                    body=body,
                )

        # --- IFF: "... if and only if ..." (check BEFORE and/or) ---
        iff_idx = self._find_at_depth0(text, " if and only if ")
        if iff_idx != -1:
            left_text = text[:iff_idx]
            right_text = text[iff_idx + 16:]  # len(" if and only if ") == 16
            left = self._parse_formula_inner(left_text)
            right = self._parse_formula_inner(right_text)
            return BinaryNode(
                connective=Connective.IFF, left=left, right=right
            )

        # --- AND: "... and ..." ---
        and_idx = self._find_at_depth0(text, " and ")
        if and_idx != -1:
            left_text = text[:and_idx]
            right_text = text[and_idx + 5:]  # len(" and ") == 5
            left = self._parse_formula_inner(left_text)
            right = self._parse_formula_inner(right_text)
            return BinaryNode(
                connective=Connective.AND, left=left, right=right
            )

        # --- OR: "... or ..." ---
        or_idx = self._find_at_depth0(text, " or ")
        if or_idx != -1:
            left_text = text[:or_idx]
            right_text = text[or_idx + 4:]  # len(" or ") == 4
            left = self._parse_formula_inner(left_text)
            right = self._parse_formula_inner(right_text)
            return BinaryNode(
                connective=Connective.OR, left=left, right=right
            )

        # --- ATOM ---
        return self._parse_atom(text)

    # ------------------------------------------------------------------
    # Atom parsing
    # ------------------------------------------------------------------

    _ATOM_RE = re.compile(r"^(\S+)\s+is\s+(.+)$", re.IGNORECASE)

    def _parse_atom(self, text: str) -> AtomNode:
        """Parse an atomic formula from plain text.

        Rules:
        - Try ``<subject> is <predicate>`` pattern.
          - If subject (lowercased) is in ``_bound_vars`` -> predicate atom
            with the variable.
          - If ``_is_fol`` and subject is a single word -> predicate atom
            treating the subject as a constant/entity.
        - Otherwise -> propositional atom with the full text as identifier.
        """
        text = text.strip()
        m = self._ATOM_RE.match(text)
        if m:
            subject = m.group(1)
            predicate = m.group(2)
            subject_lower = subject.lower()
            predicate_lower = predicate.lower()

            if subject_lower in self._bound_vars:
                return AtomNode(
                    identifier=predicate_lower,
                    variables=[subject_lower],
                    is_predicate=True,
                )

            if self._is_fol:
                # Single-word subject treated as a constant in FOL context
                return AtomNode(
                    identifier=predicate_lower,
                    variables=[subject_lower],
                    is_predicate=True,
                )

        # Propositional atom: use the full text lowercased as the identifier.
        return AtomNode(
            identifier=text.lower(), variables=[], is_predicate=False
        )

    # ------------------------------------------------------------------
    # Bracket / depth helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _matching_bracket(text: str, start: int) -> int:
        """Find the index of the closing ``}`` matching the ``{`` at *start*.

        Parameters
        ----------
        text:
            The full text string.
        start:
            Index of the opening ``{``.

        Returns
        -------
        int
            Index of the matching ``}``, or ``-1`` if not found.
        """
        if start >= len(text) or text[start] != "{":
            return -1
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i
        return -1

    @staticmethod
    def _find_at_depth0(text: str, keyword: str) -> int:
        """Find the first occurrence of *keyword* at bracket depth 0.

        Scans *text* character by character, tracking ``{`` / ``}`` depth,
        and only reports a match when the depth is 0.

        Parameters
        ----------
        text:
            The text to search in.
        keyword:
            The keyword to find.

        Returns
        -------
        int
            The starting index of the keyword at depth 0, or ``-1``.
        """
        kw_len = len(keyword)
        depth = 0
        for i in range(len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1

            if depth == 0 and text[i : i + kw_len].lower() == keyword.lower():
                return i

        return -1
