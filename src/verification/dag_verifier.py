"""DAG-based verification following GoV (Graph of Verification) with staged verification."""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from ..data.dag_schema import DAGReasoning, InferenceStep, Premise
from .semantic_parser import SemanticParser, Z3Checker, DatalogChecker, FormalExpression
from .verifier import PremiseVerifier, InferenceVerifier, VerifierConfig


@dataclass
class VerificationResult:
    """Result of verifying a single node."""
    node_id: str
    verified: bool
    method: str  # "nli" | "z3" | "datalog" | "human" | "failed"
    confidence: float
    error: Optional[str] = None
    counterexample: Optional[str] = None
    proof: Optional[str] = None


class StagedVerifier:
    """
    Staged verification following GoV:
    1. Lightweight NLI check (fast, approximate)
    2. Semantic parse to formal logic
    3. Symbolic solver (Z3/Datalog) for precise verification
    """
    
    def __init__(self, config: VerifierConfig):
        self.config = config
        self.premise_verifier = PremiseVerifier(config)
        self.inference_verifier = InferenceVerifier(config)
        self.semantic_parser = SemanticParser()
        
        # Initialize solvers
        self.z3_checker = None
        self.datalog_checker = None
        try:
            self.z3_checker = Z3Checker()
        except:
            pass
        
        self.datalog_checker = DatalogChecker()
    
    def verify_node(
        self,
        node_text: str,
        dependencies: List[str],
        dependency_texts: List[str],
        node_type: str = "inference"  # "premise" | "inference"
    ) -> VerificationResult:
        """
        Verify a single node using staged approach.
        
        Args:
            node_text: Text of the node to verify
            dependencies: IDs of dependency nodes
            dependency_texts: Texts of dependency nodes
            node_type: Type of node ("premise" or "inference")
        """
        # Stage 1: Lightweight NLI check
        nli_result = self._nli_check(node_text, dependency_texts)
        if nli_result["confidence"] > 0.9:
            return VerificationResult(
                node_id="",
                verified=True,
                method="nli",
                confidence=nli_result["confidence"]
            )
        
        # Stage 2: Try semantic parsing
        formal_exprs = self.semantic_parser.parse_with_multiple_targets(node_text)
        
        for formal_expr in formal_exprs:
            if not formal_expr.parse_success:
                continue
            
            # Stage 3: Symbolic verification
            if formal_expr.formal_type == "z3" and self.z3_checker:
                dep_formals = [
                    self.semantic_parser.parse(dep_text, "z3") 
                    for dep_text in dependency_texts
                ]
                dep_z3 = [df.expression for df in dep_formals if df.parse_success]
                
                if dep_z3:
                    is_valid, proof_or_counter = self.z3_checker.check_entailment(
                        dep_z3, formal_expr.expression
                    )
                    if is_valid:
                        return VerificationResult(
                            node_id="",
                            verified=True,
                            method="z3",
                            confidence=0.95,
                            proof=proof_or_counter
                        )
                    else:
                        return VerificationResult(
                            node_id="",
                            verified=False,
                            method="z3",
                            confidence=0.8,
                            counterexample=proof_or_counter
                        )
            
            elif formal_expr.formal_type == "datalog":
                # Add dependencies as facts
                for dep_text in dependency_texts:
                    dep_formal = self.semantic_parser.parse(dep_text, "datalog")
                    if dep_formal.parse_success:
                        self.datalog_checker.add_fact(dep_formal.expression)
                
                is_valid, proof = self.datalog_checker.check_entailment(formal_expr.expression)
                if is_valid:
                    return VerificationResult(
                        node_id="",
                        verified=True,
                        method="datalog",
                        confidence=0.9,
                        proof=proof
                    )
        
        # Fallback: NLI with lower threshold
        if nli_result["confidence"] > 0.7:
            return VerificationResult(
                node_id="",
                verified=True,
                method="nli_weak",
                confidence=nli_result["confidence"]
            )
        
        return VerificationResult(
            node_id="",
            verified=False,
            method="failed",
            confidence=0.5,
            error="All verification methods failed"
        )
    
    def _nli_check(self, hypothesis: str, premises: List[str]) -> Dict[str, Any]:
        """Lightweight NLI check using inference verifier."""
        if not premises:
            return {"confidence": 0.5, "label": "neutral"}
        
        # Use existing inference verifier
        result = self.inference_verifier.verify(premises, hypothesis)
        
        label_map = {
            "entailed": "entailment",
            "non-entailed": "contradiction",
            "weakly_supported": "neutral"
        }
        
        return {
            "confidence": result["confidence"],
            "label": label_map.get(result["label"], "neutral")
        }


class DAGVerifier:
    """
    Verify DAG reasoning structure following GoV (Graph of Verification).
    Performs topological ordering and verifies nodes in dependency order.
    """
    
    def __init__(self, config: VerifierConfig):
        self.config = config
        self.staged_verifier = StagedVerifier(config)
    
    def verify_dag(self, reasoning: DAGReasoning) -> Dict[str, Any]:
        """
        Verify entire DAG reasoning structure.
        
        Returns:
            Dictionary with verification results for each node and overall verdict
        """
        # Get topological order
        topological_order = reasoning.topological_order()
        
        node_results = {}
        verified_nodes = set()
        
        # Verify nodes in topological order
        for node_id in topological_order:
            # Find node
            node = None
            node_text = ""
            node_type = "premise"
            
            # Check if it's a premise
            for premise in reasoning.premises:
                if premise.id == node_id:
                    node = premise
                    node_text = premise.text
                    node_type = "premise"
                    break
            
            # Check if it's an inference step
            if not node:
                for step in reasoning.inference_steps:
                    if step.id == node_id:
                        node = step
                        node_text = step.text
                        node_type = "inference"
                        break
            
            if not node:
                continue
            
            # Get dependencies
            deps = reasoning.get_dependencies().get(node_id, [])
            dep_texts = []
            for dep_id in deps:
                if dep_id in verified_nodes:
                    # Get text from verified nodes
                    for p in reasoning.premises:
                        if p.id == dep_id:
                            dep_texts.append(p.text)
                    for s in reasoning.inference_steps:
                        if s.id == dep_id:
                            dep_texts.append(s.text)
            
            # Verify node
            result = self.staged_verifier.verify_node(
                node_text=node_text,
                dependencies=deps,
                dependency_texts=dep_texts,
                node_type=node_type
            )
            result.node_id = node_id
            
            node_results[node_id] = result
            
            # Update node verification status
            if result.verified:
                verified_nodes.add(node_id)
                if isinstance(node, InferenceStep):
                    node.verified = True
                    node.verification_method = result.method
                elif isinstance(node, Premise):
                    # Premises are verified separately
                    pass
        
        # Overall verdict
        all_verified = all(r.verified for r in node_results.values())
        conclusion_verified = all(
            r.verified for node_id, r in node_results.items()
            if any(node_id in step.depends_on for step in reasoning.inference_steps)
            or node_id in [p.id for p in reasoning.premises]
        )
        
        # Check conclusion
        conclusion_step = None
        for step in reasoning.inference_steps:
            if reasoning.conclusion.lower() in step.text.lower():
                conclusion_step = step
                break
        
        if conclusion_step and conclusion_step.id in node_results:
            conclusion_verified = node_results[conclusion_step.id].verified
        
        verdict = "accept" if (all_verified and conclusion_verified) else \
                 "review" if any(r.verified for r in node_results.values()) else \
                 "reject"
        
        return {
            "verdict": verdict,
            "node_results": {k: {
                "verified": v.verified,
                "method": v.method,
                "confidence": v.confidence,
                "error": v.error,
                "counterexample": v.counterexample
            } for k, v in node_results.items()},
            "all_verified": all_verified,
            "conclusion_verified": conclusion_verified,
            "verified_count": len(verified_nodes),
            "total_count": len(node_results)
        }

