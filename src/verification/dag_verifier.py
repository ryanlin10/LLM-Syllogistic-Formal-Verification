"""Staged verification with NLI, semantic parsing, and symbolic solvers."""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .semantic_parser import SemanticParser, Z3Checker, DatalogChecker, FormalExpression
from .verifier import PremiseVerifier, InferenceVerifier, VerifierConfig


@dataclass
class VerificationResult:
    """Result of a verification step."""
    verified: bool
    method: str  # "nli" | "z3" | "datalog" | "nli_weak" | "failed"
    confidence: float
    error: Optional[str] = None
    counterexample: Optional[str] = None
    proof: Optional[str] = None


class StagedVerifier:
    """
    Staged verification following GoV (Graph of Verification):
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
        except Exception:
            pass

        self.datalog_checker = DatalogChecker()

    def verify_inference(
        self,
        premises: List[str],
        conclusion: str
    ) -> VerificationResult:
        """
        Verify that conclusion follows from premises using staged approach.

        Args:
            premises: List of premise texts
            conclusion: The conclusion text to verify

        Returns:
            VerificationResult with verdict and metadata
        """
        # Stage 1: Lightweight NLI check
        nli_result = self._nli_check(conclusion, premises)
        if nli_result["confidence"] > 0.9:
            return VerificationResult(
                verified=True,
                method="nli",
                confidence=nli_result["confidence"]
            )

        # Stage 2: Try semantic parsing
        formal_exprs = self.semantic_parser.parse_with_multiple_targets(conclusion)

        for formal_expr in formal_exprs:
            if not formal_expr.parse_success:
                continue

            # Stage 3: Symbolic verification
            if formal_expr.formal_type == "z3" and self.z3_checker:
                dep_formals = [
                    self.semantic_parser.parse(premise, "z3")
                    for premise in premises
                ]
                dep_z3 = [df.expression for df in dep_formals if df.parse_success]

                if dep_z3:
                    is_valid, proof_or_counter = self.z3_checker.check_entailment(
                        dep_z3, formal_expr.expression
                    )
                    if is_valid:
                        return VerificationResult(
                            verified=True,
                            method="z3",
                            confidence=0.95,
                            proof=proof_or_counter
                        )
                    else:
                        return VerificationResult(
                            verified=False,
                            method="z3",
                            confidence=0.8,
                            counterexample=proof_or_counter
                        )

            elif formal_expr.formal_type == "datalog":
                # Add premises as facts
                for premise in premises:
                    premise_formal = self.semantic_parser.parse(premise, "datalog")
                    if premise_formal.parse_success:
                        self.datalog_checker.add_fact(premise_formal.expression)

                is_valid, proof = self.datalog_checker.check_entailment(formal_expr.expression)
                if is_valid:
                    return VerificationResult(
                        verified=True,
                        method="datalog",
                        confidence=0.9,
                        proof=proof
                    )

        # Fallback: NLI with lower threshold
        if nli_result["confidence"] > 0.7:
            return VerificationResult(
                verified=True,
                method="nli_weak",
                confidence=nli_result["confidence"]
            )

        return VerificationResult(
            verified=False,
            method="failed",
            confidence=0.5,
            error="All verification methods failed"
        )

    def verify_premises(
        self,
        premises: List[str],
        context: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Verify each premise individually.

        Args:
            premises: List of premise texts
            context: Optional context for factual verification

        Returns:
            List of verification results for each premise
        """
        results = []

        for premise in premises:
            result = self.premise_verifier.verify(premise, context)
            results.append({
                "premise": premise,
                "verified": result["label"] == "supported",
                "label": result["label"],
                "confidence": result["confidence"]
            })

        return results

    def verify_full(
        self,
        premises: List[str],
        conclusion: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Full verification: verify premises and inference.

        Args:
            premises: List of premise texts
            conclusion: The conclusion text
            context: Optional context for premise verification

        Returns:
            Complete verification result with verdict
        """
        # Verify premises
        premise_results = self.verify_premises(premises, context)
        all_premises_verified = all(r["verified"] for r in premise_results)

        # Verify inference
        inference_result = self.verify_inference(premises, conclusion)

        # Determine overall verdict
        if all_premises_verified and inference_result.verified:
            verdict = "accept"
        elif inference_result.verified or any(r["verified"] for r in premise_results):
            verdict = "review"
        else:
            verdict = "reject"

        # Calculate overall confidence
        premise_conf = sum(r["confidence"] for r in premise_results) / len(premise_results) if premise_results else 0.5
        overall_confidence = (premise_conf + inference_result.confidence) / 2

        return {
            "verdict": verdict,
            "confidence": overall_confidence,
            "premise_results": premise_results,
            "inference_result": {
                "verified": inference_result.verified,
                "method": inference_result.method,
                "confidence": inference_result.confidence,
                "error": inference_result.error,
                "proof": inference_result.proof,
                "counterexample": inference_result.counterexample
            },
            "all_premises_verified": all_premises_verified
        }

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
