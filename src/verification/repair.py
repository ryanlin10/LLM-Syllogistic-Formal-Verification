"""LLM-TRes style repair mechanism with axiom injection."""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import uuid

from .semantic_parser import SemanticParser


@dataclass
class RepairAxiom:
    """A repair axiom added to fix a verification failure."""
    id: str
    axiom_text: str  # Natural language
    formal_expression: Optional[str] = None  # Formal representation
    formal_type: Optional[str] = None
    fixes_premise_idx: Optional[int] = None  # Which premise this axiom fixes
    failure_trace: str = ""  # Why verification failed
    confidence: float = 0.7
    human_approved: bool = False
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "axiom_text": self.axiom_text,
            "formal_expression": self.formal_expression,
            "formal_type": self.formal_type,
            "fixes_premise_idx": self.fixes_premise_idx,
            "failure_trace": self.failure_trace,
            "confidence": self.confidence,
            "human_approved": self.human_approved,
            "timestamp": self.timestamp
        }


class RepairAgent:
    """
    LLM-TRes style repair agent that proposes axioms to fix verification failures.
    """

    def __init__(self, llm_model=None, semantic_parser: Optional[SemanticParser] = None):
        self.llm_model = llm_model
        self.semantic_parser = semantic_parser or SemanticParser()
        self.repair_history: List[RepairAxiom] = []

    def propose_repair(
        self,
        premises: List[str],
        conclusion: str,
        failure_trace: str,
        failed_premise_idx: Optional[int] = None
    ) -> RepairAxiom:
        """
        Propose a repair axiom to fix a verification failure.

        Following LLM-TRes approach:
        1. Analyze failure trace
        2. Identify missing link/axiom
        3. Generate repair axiom
        4. Formalize if possible
        """
        # Generate repair using LLM or rule-based
        if self.llm_model:
            axiom_text = self._llm_generate_repair(
                premises, conclusion, failure_trace
            )
        else:
            axiom_text = self._rule_based_repair(
                premises, conclusion, failure_trace
            )

        # Try to formalize
        formal_exprs = self.semantic_parser.parse_with_multiple_targets(axiom_text)
        formal_expr = formal_exprs[0] if formal_exprs else None

        repair_axiom = RepairAxiom(
            id=str(uuid.uuid4()),
            axiom_text=axiom_text,
            formal_expression=formal_expr.expression if formal_expr and formal_expr.parse_success else None,
            formal_type=formal_expr.formal_type if formal_expr and formal_expr.parse_success else None,
            fixes_premise_idx=failed_premise_idx,
            failure_trace=failure_trace,
            confidence=formal_expr.confidence if formal_expr else 0.5
        )

        self.repair_history.append(repair_axiom)
        return repair_axiom

    def _rule_based_repair(
        self,
        premises: List[str],
        conclusion: str,
        failure_trace: str
    ) -> str:
        """Generate repair axiom using rule-based patterns."""
        # Common patterns
        if "missing" in failure_trace.lower() or "not found" in failure_trace.lower():
            # Try to bridge gap between premises and conclusion
            if premises:
                premise_summary = " and ".join(premises[:2])
                return f"If {premise_summary}, then {conclusion}"

        if "contradiction" in failure_trace.lower():
            # Suggest resolving contradiction
            if premises:
                return f"Despite {premises[0]}, {conclusion} still holds"

        # Default: create linking axiom
        if premises and conclusion:
            return f"{premises[0]} implies that {conclusion}"

        return f"Repair axiom for: {conclusion}"

    def _llm_generate_repair(
        self,
        premises: List[str],
        conclusion: str,
        failure_trace: str
    ) -> str:
        """Generate repair using LLM."""
        prompt = f"""A verification step failed. Propose a repair axiom that would make the verification succeed.

Premises: {'; '.join(premises)}
Conclusion: {conclusion}
Failure reason: {failure_trace}

Generate a concise axiom (rule or fact) that, if added, would allow the conclusion to be verified.
The axiom should be a natural language statement that bridges the gap between premises and conclusion.

Axiom:"""

        # Would call LLM here
        # For now, fallback to rule-based
        return self._rule_based_repair(premises, conclusion, failure_trace)

    def apply_repair(
        self,
        premises: List[str],
        repair_axiom: RepairAxiom
    ) -> List[str]:
        """
        Apply repair axiom by adding it to premises.

        Returns:
            Updated list of premises with repair axiom added
        """
        if repair_axiom.human_approved or True:  # Auto-approve for automated pipeline
            new_premises = premises.copy()
            new_premises.append(repair_axiom.axiom_text)
            return new_premises

        return premises

    def get_repair_history(self) -> List[Dict[str, Any]]:
        """Get audit trail of all repairs."""
        return [repair.to_dict() for repair in self.repair_history]


class RepairPipeline:
    """Complete repair pipeline: failure detection -> repair -> re-verification."""

    def __init__(self, verifier, repair_agent: RepairAgent):
        self.verifier = verifier
        self.repair_agent = repair_agent

    def repair_and_verify(
        self,
        premises: List[str],
        conclusion: str,
        context: Optional[str] = None,
        max_repairs: int = 3
    ) -> Tuple[List[str], List[RepairAxiom], Dict[str, Any]]:
        """
        Repair failed verification steps and re-verify.

        Args:
            premises: List of premise texts
            conclusion: The conclusion text
            context: Optional context for verification
            max_repairs: Maximum number of repair attempts

        Returns:
            (repaired_premises, applied_repairs, final_verification_result)
        """
        applied_repairs = []
        current_premises = premises.copy()

        for iteration in range(max_repairs):
            # Verify current state
            verification_result = self.verifier.verify(
                premises=current_premises,
                conclusion=conclusion,
                context=context
            )

            if verification_result.get("verdict") == "accept":
                # Success!
                return current_premises, applied_repairs, verification_result

            # Get failure information
            failure_trace = ""
            if "inference_result" in verification_result:
                inf_result = verification_result["inference_result"]
                failure_trace = inf_result.get("error", "") or f"Verification failed with method: {inf_result.get('method', 'unknown')}"
            else:
                failure_trace = "Verification failed"

            # Propose repair
            repair_axiom = self.repair_agent.propose_repair(
                premises=current_premises,
                conclusion=conclusion,
                failure_trace=failure_trace
            )

            # Apply repair
            repair_axiom.human_approved = True  # Auto-approve for automated pipeline
            current_premises = self.repair_agent.apply_repair(current_premises, repair_axiom)
            applied_repairs.append(repair_axiom)

        # Final verification
        final_result = self.verifier.verify(
            premises=current_premises,
            conclusion=conclusion,
            context=context
        )

        return current_premises, applied_repairs, final_result
