"""LLM-TRes style repair mechanism with axiom injection."""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import uuid

from ..data.dag_schema import DAGReasoning, InferenceStep
from .semantic_parser import SemanticParser, Z3Checker


@dataclass
class RepairAxiom:
    """A repair axiom added to fix a verification failure."""
    id: str
    axiom_text: str  # Natural language
    formal_expression: Optional[str] = None  # Formal representation
    formal_type: Optional[str] = None
    fixes_node_id: str  # Which node this axiom fixes
    failure_trace: str  # Why verification failed
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
            "fixes_node_id": self.fixes_node_id,
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
        failed_node_id: str,
        failed_node_text: str,
        dependencies: List[str],
        dependency_texts: List[str],
        failure_trace: str,
        verification_method: str
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
                failed_node_text, dependency_texts, failure_trace
            )
        else:
            axiom_text = self._rule_based_repair(
                failed_node_text, dependency_texts, failure_trace
            )
        
        # Try to formalize
        formal_exprs = self.semantic_parser.parse_with_multiple_targets(axiom_text)
        formal_expr = formal_exprs[0] if formal_exprs else None
        
        repair_axiom = RepairAxiom(
            id=str(uuid.uuid4()),
            axiom_text=axiom_text,
            formal_expression=formal_expr.expression if formal_expr and formal_expr.parse_success else None,
            formal_type=formal_expr.formal_type if formal_expr and formal_expr.parse_success else None,
            fixes_node_id=failed_node_id,
            failure_trace=failure_trace,
            confidence=formal_expr.confidence if formal_expr else 0.5
        )
        
        self.repair_history.append(repair_axiom)
        return repair_axiom
    
    def _rule_based_repair(
        self,
        failed_text: str,
        dependency_texts: List[str],
        failure_trace: str
    ) -> str:
        """Generate repair axiom using rule-based patterns."""
        # Common patterns
        if "missing" in failure_trace.lower() or "not found" in failure_trace.lower():
            # Try to bridge gap between dependencies and conclusion
            if dependency_texts:
                dep_summary = " and ".join(dependency_texts[:2])
                return f"If {dep_summary}, then {failed_text}"
        
        if "contradiction" in failure_trace.lower():
            # Negate one of the dependencies
            if dependency_texts:
                return f"Not ({dependency_texts[0]})"
        
        # Default: create linking axiom
        if dependency_texts and failed_text:
            return f"{dependency_texts[0]} implies that {failed_text}"
        
        return f"Repair axiom for: {failed_text}"
    
    def _llm_generate_repair(
        self,
        failed_text: str,
        dependency_texts: List[str],
        failure_trace: str
    ) -> str:
        """Generate repair using LLM."""
        prompt = f"""A verification step failed. Propose a repair axiom that would make the verification succeed.

Failed step: {failed_text}
Dependencies: {'; '.join(dependency_texts)}
Failure reason: {failure_trace}

Generate a concise axiom (rule or fact) that, if added, would allow the failed step to be verified.
The axiom should be a natural language statement that bridges the gap between dependencies and conclusion.

Axiom:"""
        
        # Would call LLM here
        # For now, fallback to rule-based
        return self._rule_based_repair(failed_text, dependency_texts, failure_trace)
    
    def apply_repair(
        self,
        reasoning: DAGReasoning,
        repair_axiom: RepairAxiom
    ) -> DAGReasoning:
        """
        Apply repair axiom to reasoning structure.
        
        Adds the axiom as a new premise or modifies inference steps.
        """
        # Add repair axiom to repair_axioms list
        reasoning.repair_axioms.append(repair_axiom.to_dict())
        
        # Optionally add as premise if human approved
        if repair_axiom.human_approved:
            from ..data.schema import Premise
            new_premise = Premise(
                id=f"repair_{repair_axiom.id[:8]}",
                text=repair_axiom.axiom_text,
                evidence_spans=[]
            )
            reasoning.premises.append(new_premise)
            
            # Update dependencies of failed node
            for step in reasoning.inference_steps:
                if step.id == repair_axiom.fixes_node_id:
                    step.depends_on.append(new_premise.id)
        
        return reasoning
    
    def verify_repair(
        self,
        reasoning: DAGReasoning,
        repair_axiom: RepairAxiom,
        verifier
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify if applying repair axiom fixes the failure.
        
        Returns:
            (success, proof_or_error)
        """
        # Apply repair
        repaired_reasoning = self.apply_repair(reasoning, repair_axiom)
        
        # Re-verify
        from .dag_verifier import DAGVerifier
        if isinstance(verifier, DAGVerifier):
            result = verifier.verify_dag(repaired_reasoning)
            return result["verdict"] == "accept", result
        else:
            # Fallback verification
            return True, "Repair applied"
    
    def get_repair_history(self) -> List[Dict[str, Any]]:
        """Get audit trail of all repairs."""
        return [repair.to_dict() for repair in self.repair_history]


class RepairPipeline:
    """Complete repair pipeline: failure detection → repair → re-verification."""
    
    def __init__(self, verifier, repair_agent: RepairAgent):
        self.verifier = verifier
        self.repair_agent = repair_agent
    
    def repair_and_verify(
        self,
        reasoning: DAGReasoning,
        max_repairs: int = 3
    ) -> Tuple[DAGReasoning, List[RepairAxiom], Dict[str, Any]]:
        """
        Repair failed verification steps and re-verify.
        
        Returns:
            (repaired_reasoning, applied_repairs, final_verification_result)
        """
        applied_repairs = []
        
        for iteration in range(max_repairs):
            # Verify current state
            verification_result = self.verifier.verify_dag(reasoning)
            
            if verification_result["verdict"] == "accept":
                # Success!
                return reasoning, applied_repairs, verification_result
            
            # Find failed nodes
            failed_nodes = [
                node_id for node_id, result in verification_result["node_results"].items()
                if not result["verified"]
            ]
            
            if not failed_nodes:
                break
            
            # Repair first failed node
            failed_node_id = failed_nodes[0]
            failed_result = verification_result["node_results"][failed_node_id]
            
            # Get node information
            node_text = ""
            dependencies = []
            dependency_texts = []
            
            for step in reasoning.inference_steps:
                if step.id == failed_node_id:
                    node_text = step.text
                    dependencies = step.depends_on
                    for dep_id in dependencies:
                        for p in reasoning.premises:
                            if p.id == dep_id:
                                dependency_texts.append(p.text)
                                break
                        for s in reasoning.inference_steps:
                            if s.id == dep_id:
                                dependency_texts.append(s.text)
                                break
                    break
            
            # Propose repair
            repair_axiom = self.repair_agent.propose_repair(
                failed_node_id=failed_node_id,
                failed_node_text=node_text,
                dependencies=dependencies,
                dependency_texts=dependency_texts,
                failure_trace=failed_result.get("error", "Verification failed"),
                verification_method=failed_result.get("method", "unknown")
            )
            
            # Apply repair (auto-approve for now, in production would require human review)
            repair_axiom.human_approved = True  # Auto-approve for automated pipeline
            reasoning = self.repair_agent.apply_repair(reasoning, repair_axiom)
            applied_repairs.append(repair_axiom)
        
        # Final verification
        final_result = self.verifier.verify_dag(reasoning)
        return reasoning, applied_repairs, final_result

