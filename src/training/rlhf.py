"""Reinforcement Learning from Human Feedback (RLHF) training script."""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification
)
from peft import PeftModel
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from datasets import Dataset
import numpy as np

from ..data.schema import safe_parse_model_output
from ..verification.verifier import VerifierPipeline, VerifierConfig
from ..data.curation import DataCurator


class RewardModel:
    """Reward model using verifier."""
    
    def __init__(self, verifier_config: VerifierConfig):
        self.verifier = VerifierPipeline(verifier_config)
        self.config = verifier_config
    
    def compute_reward(
        self,
        parsed_outputs: List[Dict[str, Any]],
        evidence_spans: Optional[List[List[List[Dict[str, Any]]]]] = None
    ) -> List[float]:
        """Compute rewards for a batch of outputs."""
        rewards = []
        
        for i, parsed in enumerate(parsed_outputs):
            evidence = evidence_spans[i] if evidence_spans and i < len(evidence_spans) else None
            verdict = self.verifier.verify_output(parsed, evidence)
            
            # Map verdict to reward
            if verdict["verdict"] == "accept":
                reward = 1.0 * verdict["confidence"]
            elif verdict["verdict"] == "review":
                reward = 0.5 * verdict["confidence"]
            else:  # reject
                reward = -0.5 * (1 - verdict["confidence"])
            
            # Add brevity penalty (encourage concise outputs)
            premises_count = len(parsed.get("premises", []))
            conclusion_length = len(parsed.get("conclusion", ""))
            brevity_penalty = -0.01 * (premises_count - 3) ** 2 - 0.001 * max(0, conclusion_length - 200)
            reward += brevity_penalty
            
            rewards.append(reward)
        
        return rewards


def format_prompt_for_rlhf(context: str, question: Optional[str] = None, system_prompt: Optional[str] = None) -> str:
    """Format prompt for RLHF generation."""
    if system_prompt is None:
        system_prompt = """You must respond only in valid JSON format. Your response must contain exactly two fields:
1. "premises": an array of concise factual statements (premises) that support your conclusion
2. "conclusion": a single sentence that follows logically from the premises"""
    
    user_part = f"Question: {question}\n" if question else ""
    user_part += f"Context: {context}"
    
    return f"System: {system_prompt}\n\nUser: {user_part}\n\nAssistant:"


def train_rlhf(config_path: str = "./config.yaml"):
    """RLHF training using PPO with verifier as reward model."""
    config = load_config(config_path)
    rlhf_config = config["rlhf"]
    
    # Load base model (from fine-tuning)
    print("Loading fine-tuned model...")
    base_model_path = config["training"]["output_dir"] + "/final"
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Wrap model with value head for PPO
    model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)
    model_ref = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)  # Reference for KL penalty
    
    # Load data
    print("Loading training data...")
    curator = DataCurator()
    train_data = curator.load_jsonl(config["data"]["train_path"])
    
    # Setup reward model
    verifier_config = VerifierConfig(**config["verifier"])
    reward_model = RewardModel(verifier_config)
    
    # Create prompt dataset
    prompts = []
    for ann in train_data[:1000]:  # Use subset for RLHF
        context = ann.get("context", "")
        prompt = format_prompt_for_rlhf(context)
        prompts.append({"query": prompt})
    
    prompt_dataset = Dataset.from_list(prompts)
    
    # PPO Config
    ppo_config = PPOConfig(
        model_name=base_model_path,
        learning_rate=rlhf_config["learning_rate"],
        batch_size=rlhf_config["batch_size"],
        mini_batch_size=rlhf_config["mini_batch_size"],
        gradient_accumulation_steps=rlhf_config["gradient_accumulation_steps"],
        ppo_epochs=rlhf_config["ppo_epochs"],
        cliprange=rlhf_config["cliprange"],
        cliprange_value=rlhf_config["cliprange_value"],
        gamma=rlhf_config["gamma"],
        lam=rlhf_config["lam"],
        kl_penalty=rlhf_config["kl_penalty"],
        log_with="wandb" if config["training"].get("report_to") == "wandb" else None,
        output_dir=rlhf_config["output_dir"]
    )
    
    # PPO Trainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=model_ref,
        tokenizer=tokenizer
    )
    
    # Generation kwargs
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 512,
        "temperature": 0.7,
    }
    
    output_length_sampler = LengthSampler(
        output_min_length=100,
        output_max_length=500
    )
    
    # Training loop
    print("Starting RLHF training...")
    max_ppo_steps = 100
    
    for epoch in range(max_ppo_steps):
        for batch in prompt_dataset:
            query_tensors = [tokenizer.encode(q, return_tensors="pt").squeeze() for q in batch["query"]]
            
            # Generate responses
            response_tensors = []
            for query_tensor in query_tensors:
                response = ppo_trainer.generate(
                    query_tensor.unsqueeze(dim=0),
                    return_prompt=False,
                    length_sampler=output_length_sampler,
                    **generation_kwargs
                )
                response_tensors.append(response.squeeze())
            
            # Decode responses
            responses = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]
            
            # Parse outputs
            parsed_outputs = []
            for resp in responses:
                parsed, error = safe_parse_model_output(resp)
                if parsed is None:
                    # Create dummy output for failed parsing
                    parsed = {"premises": [], "conclusion": ""}
                parsed_outputs.append(parsed)
            
            # Compute rewards
            rewards = reward_model.compute_reward(parsed_outputs)
            rewards_tensor = [torch.tensor(r) for r in rewards]
            
            # PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards_tensor)
            ppo_trainer.log_stats(stats, batch, rewards)
            
            if (epoch + 1) % 10 == 0:
                print(f"Step {epoch + 1}: Mean reward = {np.mean(rewards):.4f}")
    
    # Save final model
    final_output_dir = Path(rlhf_config["output_dir"]) / "final"
    final_output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    print(f"Saved RLHF model to {final_output_dir}")


def load_config(config_path: str = "./config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "./config.yaml"
    train_rlhf(config_path)

