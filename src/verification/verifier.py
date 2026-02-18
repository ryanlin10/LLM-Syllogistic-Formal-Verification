"""Z3-based symbolic verification pipeline.

Main entry point for verifying logical inferences using Z3 SMT solving.
Replaces the earlier NLI-based verifier with a symbolic approach that
parses natural-language premises/conclusions into formal logic and checks
entailment via Z3's satisfiability engine.
"""

from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass

try:
    import z3
except ImportError:
    pass

from .parser import SemiFormalParser, ParseResult
from .translator import Z3Translator, TranslationContext, Z3_AVAILABLE


@dataclass
class VerifierConfig:
    """Configuration for the symbolic verification pipeline."""

    timeout_ms: int = 5000
    # Legacy fields -- accepted but ignored, for backward compat with YAML configs
    premise_model_path: str = ""
    inference_model_path: str = ""
    confidence_threshold: float = 0.85
    batch_size: int = 32
    device: str = "cpu"


class VerifierPipeline:
    """End-to-end symbolic verification pipeline.

    Uses semi-formal parsing and Z3 SMT solving to determine whether a
    conclusion is logically entailed by a set of premises.  Degrades
    gracefully when Z3 is not installed.
    """

    def __init__(self, config: Optional[VerifierConfig] = None):
        self.config = config or VerifierConfig()
        self.parser = SemiFormalParser()
        self.translator = Z3Translator() if Z3_AVAILABLE else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(
        self,
        text: Optional[str] = None,
        *,
        premises: Optional[List[str]] = None,
        conclusion: Optional[str] = None,
        **kwargs,
    ) -> Union[bool, Dict]:
        """Verify a logical inference.

        Supports two call signatures:

        1. ``verify("free text with premises. Therefore, conclusion.")``
           Returns a ``bool`` indicating whether the conclusion follows.

        2. ``verify(premises=["p1", "p2"], conclusion="c")``
           Returns a ``Dict`` with keys ``verdict`` and ``confidence``.
        """
        # Signature 1: free-text input -> bool
        if text is not None and premises is None and conclusion is None:
            if not Z3_AVAILABLE:
                return False
            try:
                parsed: ParseResult = self.parser.parse_text(text)
                return self._check_entailment(
                    parsed.premises, parsed.conclusion
                )
            except Exception:
                return False

        # Signature 2: structured input -> dict
        if premises is not None and conclusion is not None:
            if not Z3_AVAILABLE:
                return {"verdict": "review", "confidence": 0.0}
            try:
                parsed = self.parser.parse_inference(premises, conclusion)
                result = self._check_entailment(
                    parsed.premises, parsed.conclusion
                )
                if result:
                    return {"verdict": "accept", "confidence": 1.0}
                else:
                    return {"verdict": "reject", "confidence": 1.0}
            except Exception:
                return {"verdict": "review", "confidence": 0.0}

        # Fallback: bad call signature
        return False

    def verify_inference(
        self, premises: List[str], conclusion: str
    ) -> bool:
        """Verify that *conclusion* follows from *premises*.

        Structured input, returns ``bool``.
        """
        if not Z3_AVAILABLE:
            return False
        try:
            parsed: ParseResult = self.parser.parse_inference(
                premises, conclusion
            )
            return self._check_entailment(parsed.premises, parsed.conclusion)
        except Exception:
            return False

    def verify_output(
        self,
        parsed_output: Dict[str, Any],
        evidence_spans: Optional[Any] = None,
    ) -> Dict:
        """Backward-compatible entry point used by rlhf.py and evaluator.py.

        Accepts a dict with ``premises`` (list of strings or dicts) and
        ``conclusion`` (string).  Returns a verdict dict.
        """
        try:
            raw_premises = parsed_output.get("premises", [])
            # Handle premises that may be dicts with a "text" key
            if raw_premises and isinstance(raw_premises[0], dict):
                premise_texts = [p.get("text", "") for p in raw_premises]
            else:
                premise_texts = list(raw_premises)

            conclusion = parsed_output.get("conclusion", "")

            if not premise_texts or not conclusion:
                return {
                    "verdict": "review",
                    "confidence": 0.0,
                    "reason": "symbolic_verification",
                    "details": {
                        "error": "empty premises or conclusion",
                    },
                }

            result = self.verify_inference(premise_texts, conclusion)

            return {
                "verdict": "accept" if result else "reject",
                "confidence": 1.0,
                "reason": "symbolic_verification",
                "details": {
                    "premises_count": len(premise_texts),
                    "conclusion": conclusion,
                    "z3_available": Z3_AVAILABLE,
                },
            }
        except Exception:
            return {
                "verdict": "review",
                "confidence": 0.0,
                "reason": "symbolic_verification",
                "details": {"error": "parse_failure"},
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_entailment(self, premises, conclusion) -> bool:
        """Core Z3 entailment check.

        Checks whether ``premises |= conclusion`` by asserting all
        premises together with ``Not(conclusion)`` and testing for
        unsatisfiability.

        Parameters
        ----------
        premises : List[FormulaNode]
            Parsed premise formula nodes.
        conclusion : FormulaNode
            Parsed conclusion formula node.

        Returns
        -------
        bool
            ``True`` if the inference is valid (the negated conclusion is
            unsatisfiable given the premises), ``False`` otherwise.
        """
        if self.translator is None:
            return False

        try:
            ctx = TranslationContext()

            premise_exprs = [
                self.translator.translate(p, ctx) for p in premises
            ]
            conclusion_expr = self.translator.translate(conclusion, ctx)

            solver = z3.Solver()
            solver.set("timeout", self.config.timeout_ms)

            for expr in premise_exprs:
                solver.add(expr)

            solver.add(z3.Not(conclusion_expr))

            check = solver.check()
            if check == z3.unsat:
                return True
            elif check == z3.sat:
                return False
            else:
                # z3.unknown
                return False
        except Exception:
            return False
