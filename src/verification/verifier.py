"""Automated premise and inference verification modules."""

import torch
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ..data.schema import Annotation, Premise


@dataclass
class VerifierConfig:
    """Configuration for verifier models."""
    premise_model_path: str = "./models/verifier/premise"
    inference_model_path: str = "./models/verifier/inference"
    confidence_threshold: float = 0.85
    batch_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class PremiseVerifier:
    """Verifies if a premise is supported by evidence."""
    
    def __init__(self, config: VerifierConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load premise verification model."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.premise_model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.premise_model_path,
                num_labels=3  # supported, contradicted, unverifiable
            )
            self.model.to(self.config.device)
            self.model.eval()
        except Exception as e:
            print(f"Warning: Could not load premise verifier from {self.config.premise_model_path}")
            print(f"Error: {e}. Using lightweight rule-based fallback.")
            self.model = None
    
    def verify(
        self,
        premise: str,
        evidence_spans: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Verify a single premise."""
        # Rule-based fallback if model not loaded
        if self.model is None:
            return self._rule_based_verify(premise, evidence_spans)
        
        # Model-based verification
        evidence_text = ""
        if evidence_spans:
            evidence_text = " ".join([span.get("text", "") for span in evidence_spans])
        
        # Format input
        text = f"Premise: {premise}\nEvidence: {evidence_text}"
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        
        label_idx = np.argmax(probs)
        confidence = float(probs[label_idx])
        labels = ["supported", "contradicted", "unverifiable"]
        label = labels[label_idx]
        
        return {
            "label": label,
            "confidence": confidence,
            "probs": {l: float(p) for l, p in zip(labels, probs)},
            "premise": premise,
            "evidence_used": len(evidence_spans) if evidence_spans else 0
        }
    
    def _rule_based_verify(
        self,
        premise: str,
        evidence_spans: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Lightweight rule-based verification fallback."""
        if not evidence_spans:
            return {
                "label": "unverifiable",
                "confidence": 0.5,
                "probs": {"supported": 0.2, "contradicted": 0.1, "unverifiable": 0.7},
                "premise": premise,
                "evidence_used": 0
            }
        
        # Simple keyword overlap check
        premise_lower = premise.lower()
        evidence_text = " ".join([span.get("text", "").lower() for span in evidence_spans])
        
        # Count word overlap
        premise_words = set(premise_lower.split())
        evidence_words = set(evidence_text.split())
        overlap = len(premise_words & evidence_words) / max(len(premise_words), 1)
        
        if overlap > 0.3:
            label = "supported"
            confidence = min(0.7, overlap)
        else:
            label = "unverifiable"
            confidence = 0.5
        
        return {
            "label": label,
            "confidence": confidence,
            "probs": {
                "supported": confidence if label == "supported" else 0.2,
                "contradicted": 0.1,
                "unverifiable": 1 - confidence if label == "supported" else confidence
            },
            "premise": premise,
            "evidence_used": len(evidence_spans)
        }


class InferenceVerifier:
    """Verifies if a conclusion follows from premises."""
    
    def __init__(self, config: VerifierConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load inference verification model."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.inference_model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.inference_model_path,
                num_labels=3  # entailed, non-entailed, weakly_supported
            )
            self.model.to(self.config.device)
            self.model.eval()
        except Exception as e:
            print(f"Warning: Could not load inference verifier from {self.config.inference_model_path}")
            print(f"Error: {e}. Using lightweight rule-based fallback.")
            self.model = None
    
    def verify(
        self,
        premises: List[str],
        conclusion: str
    ) -> Dict[str, Any]:
        """Verify if conclusion follows from premises."""
        # Rule-based fallback
        if self.model is None:
            return self._rule_based_verify(premises, conclusion)
        
        # Format input
        premises_text = " ".join([f"Premise {i+1}: {p}" for i, p in enumerate(premises)])
        text = f"{premises_text}\nConclusion: {conclusion}"
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True
        )
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        
        label_idx = np.argmax(probs)
        confidence = float(probs[label_idx])
        labels = ["entailed", "non-entailed", "weakly_supported"]
        label = labels[label_idx]
        
        return {
            "label": label,
            "confidence": confidence,
            "probs": {l: float(p) for l, p in zip(labels, probs)},
            "premises_count": len(premises),
            "conclusion": conclusion
        }
    
    def _rule_based_verify(self, premises: List[str], conclusion: str) -> Dict[str, Any]:
        """Lightweight rule-based verification fallback."""
        if not premises:
            return {
                "label": "non-entailed",
                "confidence": 0.6,
                "probs": {"entailed": 0.1, "non-entailed": 0.7, "weakly_supported": 0.2},
                "premises_count": 0,
                "conclusion": conclusion
            }
        
        # Simple keyword and semantic overlap
        conclusion_lower = conclusion.lower()
        premises_text = " ".join(premises).lower()
        
        # Word overlap
        conclusion_words = set(conclusion_lower.split())
        premises_words = set(premises_text.split())
        overlap = len(conclusion_words & premises_words) / max(len(conclusion_words), 1)
        
        # Simple heuristics
        if overlap > 0.4 and len(premises) >= 2:
            label = "entailed"
            confidence = min(0.7, overlap + 0.2)
        elif overlap > 0.2:
            label = "weakly_supported"
            confidence = 0.5
        else:
            label = "non-entailed"
            confidence = 0.6
        
        return {
            "label": label,
            "confidence": confidence,
            "probs": {
                "entailed": confidence if label == "entailed" else 0.2,
                "non-entailed": confidence if label == "non-entailed" else 0.3,
                "weakly_supported": confidence if label == "weakly_supported" else 0.2
            },
            "premises_count": len(premises),
            "conclusion": conclusion
        }


class VerifierPipeline:
    """End-to-end verification pipeline."""
    
    def __init__(self, config: VerifierConfig):
        self.config = config
        self.premise_verifier = PremiseVerifier(config)
        self.inference_verifier = InferenceVerifier(config)
    
    def verify_output(
        self,
        parsed_output: Dict[str, Any],
        evidence_spans: Optional[List[List[Dict[str, Any]]]] = None
    ) -> Dict[str, Any]:
        """Verify complete model output."""
        premises = parsed_output.get("premises", [])
        conclusion = parsed_output.get("conclusion", "")
        
        # Handle different premise formats
        if isinstance(premises[0], dict) if premises else False:
            premise_texts = [p.get("text", "") for p in premises]
        else:
            premise_texts = premises
        
        # Verify each premise
        premise_results = []
        for i, premise_text in enumerate(premise_texts):
            evidence = evidence_spans[i] if evidence_spans and i < len(evidence_spans) else None
            result = self.premise_verifier.verify(premise_text, evidence)
            premise_results.append(result)
        
        # Check if any premise is contradicted
        contradicted_premises = [
            r for r in premise_results 
            if r["label"] == "contradicted" and r["confidence"] > 0.7
        ]
        
        if contradicted_premises:
            return {
                "verdict": "reject",
                "reason": "contradicted_premise",
                "details": {
                    "premises": premise_results,
                    "inference": None
                },
                "confidence": 1.0 - max(r["confidence"] for r in contradicted_premises)
            }
        
        # Check for unverifiable premises (below threshold)
        unverifiable_premises = [
            r for r in premise_results
            if r["label"] == "unverifiable" and r["confidence"] < self.config.confidence_threshold
        ]
        
        # Verify inference
        inference_result = self.inference_verifier.verify(premise_texts, conclusion)
        
        # Determine final verdict
        if inference_result["label"] == "entailed" and \
           inference_result["confidence"] > self.config.confidence_threshold and \
           len(unverifiable_premises) == 0:
            verdict = "accept"
        elif inference_result["label"] == "weakly_supported" and \
             len(unverifiable_premises) == 0:
            verdict = "review"
        else:
            verdict = "review" if len(contradicted_premises) == 0 else "reject"
        
        return {
            "verdict": verdict,
            "reason": "inference_check" if verdict != "reject" else "premise_or_inference_failure",
            "details": {
                "premises": premise_results,
                "inference": inference_result
            },
            "confidence": min(
                inference_result["confidence"],
                min((r["confidence"] for r in premise_results), default=1.0)
            )
        }

