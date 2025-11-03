"""Enhanced RLHF with stepwise verification rewards following GoV and LLM-TRes."""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from datasets import Dataset
import numpy as np

from ..data.schema import safe_parse_model_output
from ..data.dag_schema import DAGReasoning, DAGAnnotation
from ..verification.dag_verifier import DAGVerifier
from ..verification.repair import RepairPipeline, RepairAgent
from ..verification.verifier import VerifierConfig
from ..data.curation import DataCurator


class EnhancedRewardModel:
    """
    Enhanced reward model with stepwise verification rewards.
    
    Rewards:
    - +1 for each verified inference step
    - +k bonus for complete DAG verification
    - Penalty for repairs required
    - Penalty for missing dependencies
    """
    
    def __init__(
        self,
        verifier_config: VerifierConfig,
        step_reward: float = 1.0,
        complete_bonus: float = 5.0,
        repair_penalty: float = -0.5,
        missing_dep_penalty: float = -1.0
    ):
        self.verifier = DAGVerifier(verifier_config)
        self.repair_agent = RepairAgent()
        self.repair_pipeline = RepairPipeline(self.verifier, self.repair_agent)
        
        self.step_reward = step_reward
        self.complete_bonus = complete_bonus
        self.repair_penalty = repair_penalty
        self.missing_dep_penalty = missing_dep_penalty
    
    def compute_reward(
        self,
        parsed_outputs: List[Dict[str, Any]],
        apply_repair: bool = True
    ) -> List[float]:
        """Compute rewards for a batch of outputs."""
        rewards = []
        
        for parsed in parsed_outputs:
            # Convert to DAG reasoning if needed
            reasoning = self._parse_to_dag(parsed)
            
            if not reasoning:
                rewards.append(-2.0)  # Severe penalty for unparseable
                continue
            
            # Verify
            verification_result = self.verifier.verify_dag(reasoning)
            
            # Base reward from verification
            verified_steps = sum(
                1 for r in verification_result["node_results"].values()
                if r["verified"]
            )
            total_steps = len(verification_result["node_results"])
            
            # Stepwise reward
            step_reward = (verified_steps / max(total_steps, 1)) * self.step_reward
            
            # Complete verification bonus
            complete_bonus = 0.0
            if verification_result["verdict"] == "accept":
                complete_bonus = self.complete_bonus
            
            # Repair penalty
            repair_penalty = 0.0
            if apply_repair and verification_result["verdict"] != "accept":
                # Try repair
                repaired_reasoning, applied_repairs, repair_result = \
                    self.repair_pipeline.repair_and_verify(reasoning, max_repairs=1)
                
                if applied_repairs:
                    repair_penalty = self.repair_penalty * len(applied_repairs)
                    
                    # If repair succeeded, reduce penalty
                    if repair_result["verdict"] == "accept":
                        repair_penalty *= 0.5
            
            # Missing dependency penalty
            missing_dep_penalty = 0.0
            all_node_ids = reasoning.get_all_node_ids()
            for step in reasoning.inference_steps:
                for dep_id in step.depends_on:
                    if dep_id not in all_node_ids:
                        missing_dep_penalty += self.missing_dep_penalty
                        break
            
            # Brevity penalty (encourage concise but complete reasoning)
            total_nodes = len(reasoning.premises) + len(reasoning.inference_steps)
            brevity_penalty = -0.01 * max(0, total_nodes - 5) ** 2
            
            total_reward = (
                step_reward +
                complete_bonus +
                repair_penalty +
                missing_dep_penalty +
                brevity_penalty
            )
            
            rewards.append(total_reward)
        
        return rewards
    
    def _parse_to_dag(self, parsed: Dict[str, Any]) -> Optional[DAGReasoning]:
        """Convert parsed output to DAG reasoning structure."""
        try:
            from ..data.dag_schema import Premise, InferenceStep
            
            # Check if already in DAG format
            if "reasoning" in parsed:
                reasoning_dict = parsed["reasoning"]
            else:
                # Convert from legacy format
                premises_list = parsed.get("premises", [])
                conclusion = parsed.get("conclusion", "")
                
                if isinstance(conclusion, dict):
                    conclusion = conclusion.get("text", "")
                
                # Create premises
                premises = []
                for i, p in enumerate(premises_list):
                    if isinstance(p, dict):
                        text = p.get("text", p.get("premise", ""))
                        p_id = p.get("id", f"p{i+1}")
                    else:
                        text = str(p)
                        p_id = f"p{i+1}"
                    
                    premises.append(Premise(id=p_id, text=text, evidence_spans=[]))
                
                # Create simple inference steps
                inference_steps = []
                if len(premises) >= 2:
                    # Chain premises
                    for i in range(len(premises) - 1):
                        step = InferenceStep(
                            id=f"inf{i+1}",
                            text=f"Combining premises {premises[i].id} and {premises[i+1].id}",
                            depends_on=[premises[i].id, premises[i+1].id]
                        )
                        inference_steps.append(step)
                    
                    # Final step to conclusion
                    final_step = InferenceStep(
                        id="inf_final",
                        text=f"Concluding from previous steps",
                        depends_on=[step.id for step in inference_steps] if inference_steps else [p.id for p in premises]
                    )
                    inference_steps.append(final_step)
                elif len(premises) == 1:
                    step = InferenceStep(
                        id="inf1",
                        text=f"From premise {premises[0].id}",
                        depends_on=[premises[0].id]
                    )
                    inference_steps.append(step)
                
                reasoning_dict = {
                    "premises": [p.to_dict() for p in premises],
                    "inference_steps": [s.to_dict() for s in inference_steps],
                    "conclusion": conclusion,
                    "conclusion_type": "entailment"
                }
            
            return DAGReasoning.from_dict(reasoning_dict)
        
        except Exception as e:
            print(f"Error parsing to DAG: {e}")
            return None


def train_enhanced_rlhf(config_path: str = "./config.yaml"):
    """Enhanced RLHF training with stepwise verification rewards."""
    from .rlhf import load_config, train_rlhf
    
    # For now, use the enhanced reward model with existing RLHF infrastructure
    # In production, would integrate more tightly
    
    config = load_config(config_path)
    verifier_config = VerifierConfig(**config["verifier"])
    
    # Create enhanced reward model
    reward_model = EnhancedRewardModel(
        verifier_config=verifier_config,
        step_reward=1.0,
        complete_bonus=5.0,
        repair_penalty=-0.5
    )
    
    print("Enhanced RLHF with stepwise verification rewards")
    print("Using DAG verification and repair mechanisms")
    
    # Continue with standard RLHF flow but with enhanced rewards
    # (Would integrate this into the main RLHF training loop)
    
    print("Note: Enhanced RLHF requires integration into main RLHF script")
    print("Use train_rlhf() with EnhancedRewardModel for full functionality")


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "./config.yaml"
    train_enhanced_rlhf(config_path)

