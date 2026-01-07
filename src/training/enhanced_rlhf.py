"""Enhanced RLHF with verification-based rewards."""

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

from ..data.schema import safe_parse_model_output, Annotation, Premise
from ..verification.verifier import VerifierPipeline, VerifierConfig
from ..data.curation import DataCurator


class EnhancedRewardModel:
    """
    Enhanced reward model with verification-based rewards.

    Rewards:
    - +1.0 for verified (accepted) outputs
    - +0.5 for outputs requiring review
    - -0.5 for rejected outputs
    - -2.0 for unparseable outputs
    """

    def __init__(
        self,
        verifier_config: VerifierConfig,
        accept_reward: float = 1.0,
        review_reward: float = 0.5,
        reject_penalty: float = -0.5,
        parse_fail_penalty: float = -2.0
    ):
        self.verifier = VerifierPipeline(verifier_config)

        self.accept_reward = accept_reward
        self.review_reward = review_reward
        self.reject_penalty = reject_penalty
        self.parse_fail_penalty = parse_fail_penalty

    def compute_reward(
        self,
        parsed_outputs: List[Dict[str, Any]],
        contexts: Optional[List[str]] = None
    ) -> List[float]:
        """Compute rewards for a batch of outputs."""
        rewards = []

        for i, parsed in enumerate(parsed_outputs):
            # Check if parsing failed
            if parsed is None:
                rewards.append(self.parse_fail_penalty)
                continue

            # Extract premises and conclusion
            premises = self._extract_premises(parsed)
            conclusion = self._extract_conclusion(parsed)

            if not premises or not conclusion:
                rewards.append(self.parse_fail_penalty)
                continue

            # Get context for verification
            context = contexts[i] if contexts and i < len(contexts) else None

            # Verify the output
            try:
                verification_result = self.verifier.verify(
                    premises=premises,
                    conclusion=conclusion,
                    context=context
                )

                verdict = verification_result.get("verdict", "reject")

                if verdict == "accept":
                    reward = self.accept_reward
                elif verdict == "review":
                    reward = self.review_reward
                else:
                    reward = self.reject_penalty

                # Add confidence bonus/penalty
                confidence = verification_result.get("confidence", 0.5)
                reward += (confidence - 0.5) * 0.2  # Small adjustment based on confidence

                rewards.append(reward)

            except Exception as e:
                print(f"Verification error: {e}")
                rewards.append(self.reject_penalty)

        return rewards

    def _extract_premises(self, parsed: Dict[str, Any]) -> List[str]:
        """Extract premise texts from parsed output."""
        premises_data = parsed.get("premises", [])
        premises = []

        for p in premises_data:
            if isinstance(p, dict):
                text = p.get("text", p.get("premise", ""))
            else:
                text = str(p)

            if text:
                premises.append(text)

        return premises

    def _extract_conclusion(self, parsed: Dict[str, Any]) -> Optional[str]:
        """Extract conclusion text from parsed output."""
        conclusion = parsed.get("conclusion", "")

        if isinstance(conclusion, dict):
            return conclusion.get("text", "")

        return str(conclusion) if conclusion else None


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_enhanced_rlhf(config_path: str = "./config.yaml"):
    """Enhanced RLHF training with verification-based rewards."""
    config = load_config(config_path)
    verifier_config = VerifierConfig(**config.get("verifier", {}))

    # Create enhanced reward model
    reward_model = EnhancedRewardModel(
        verifier_config=verifier_config,
        accept_reward=1.0,
        review_reward=0.5,
        reject_penalty=-0.5
    )

    print("Enhanced RLHF with verification-based rewards")
    print("Reward structure:")
    print(f"  Accept: +{reward_model.accept_reward}")
    print(f"  Review: +{reward_model.review_reward}")
    print(f"  Reject: {reward_model.reject_penalty}")
    print(f"  Parse fail: {reward_model.parse_fail_penalty}")

    # Load training data
    curator = DataCurator()
    train_data = curator.load_jsonl(config.get("data", {}).get("train_path", "./data/train.jsonl"))

    if not train_data:
        print("No training data found")
        return

    print(f"Loaded {len(train_data)} training examples")

    # Note: Full RLHF training loop would be integrated here
    # This provides the reward model infrastructure
    print("Note: Use train_rlhf() with EnhancedRewardModel for full RLHF training")


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "./config.yaml"
    train_enhanced_rlhf(config_path)
