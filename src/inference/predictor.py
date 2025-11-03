"""Inference pipeline for generating structured outputs."""

import json
import yaml
import torch
from typing import Dict, Any, Optional, List
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from ..data.schema import safe_parse_model_output, format_prompt
from ..verification.verifier import VerifierPipeline, VerifierConfig
from ..retrieval.retriever import DocumentRetriever, RetrievalConfig


class StructuredLLMPredictor:
    """Predictor for structured premise-conclusion outputs."""
    
    def __init__(
        self,
        model_path: str,
        verifier_config: Optional[VerifierConfig] = None,
        retriever: Optional[DocumentRetriever] = None,
        config_path: str = "./config.yaml"
    ):
        self.config_path = config_path
        
        # Load config with environment variable support
        from ..utils.config_loader import load_config
        self.config = load_config(config_path)
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Check if LoRA weights exist
        lora_path = Path(model_path) / "adapter_model.bin"
        if lora_path.exists():
            # Load base model first
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config["model"]["base_model"],
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            self.model = PeftModel.from_pretrained(base_model, model_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Setup verifier
        if verifier_config is None:
            verifier_config = VerifierConfig(**self.config["verifier"])
        self.verifier = VerifierPipeline(verifier_config)
        
        # Setup retriever
        self.retriever = retriever
        if retriever is None and self.config.get("retrieval"):
            try:
                retrieval_config = RetrievalConfig(**self.config["retrieval"])
                self.retriever = DocumentRetriever(retrieval_config)
                self.retriever.load_index()
            except Exception as e:
                print(f"Warning: Could not load retriever: {e}")
                self.retriever = None
        
        # System prompt
        self.system_prompt = self.config.get("prompts", {}).get("system_prompt")
    
    def generate(
        self,
        question: str,
        context: Optional[str] = None,
        use_retrieval: bool = True,
        verify: bool = True,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Dict[str, Any]:
        """Generate structured output."""
        # Retrieve context if needed
        if use_retrieval and self.retriever:
            retrieved_context = self.retriever.retrieve_for_context(question)
            if context:
                context = f"{context}\n\n{retrieved_context}"
            else:
                context = retrieved_context
        
        # Format prompt
        prompt = format_prompt(context or "", question, self.system_prompt)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        # Parse
        parsed, parse_error = safe_parse_model_output(generated_text)
        
        result = {
            "question": question,
            "context": context,
            "raw_output": generated_text,
            "parsed": parsed,
            "parse_error": parse_error
        }
        
        # Verify if requested
        if verify and parsed:
            # Retrieve evidence for premises if retriever available
            evidence_spans = None
            if self.retriever and parsed.get("premises"):
                premises = parsed.get("premises", [])
                evidence_spans = []
                for premise in premises:
                    premise_text = premise if isinstance(premise, str) else premise.get("text", "")
                    evidence = self.retriever.link_premise_to_evidence(premise_text)
                    evidence_spans.append(evidence)
            
            verdict = self.verifier.verify_output(parsed, evidence_spans)
            result["verification"] = verdict
        else:
            result["verification"] = None
        
        return result
    
    def generate_batch(
        self,
        questions: List[str],
        contexts: Optional[List[str]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Generate outputs for a batch of questions."""
        results = []
        for i, question in enumerate(questions):
            context = contexts[i] if contexts and i < len(contexts) else None
            result = self.generate(question, context, **kwargs)
            results.append(result)
        return results

