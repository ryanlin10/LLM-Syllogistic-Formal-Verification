"""Configuration loader with environment variable support for model switching."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv


def load_config(config_path: str = "./config.yaml", env_path: str = "./.env") -> Dict[str, Any]:
    """
    Load configuration from YAML file and override with environment variables.
    
    Priority order:
    1. Environment variables (.env file)
    2. System environment variables
    3. config.yaml file
    
    Args:
        config_path: Path to config.yaml file
        env_path: Path to .env file
        
    Returns:
        Dictionary with merged configuration
    """
    # Load .env file if it exists
    env_file = Path(env_path)
    if env_file.exists():
        load_dotenv(env_file)
        # Only print if MODEL_NAME is set to avoid clutter
        if "MODEL_NAME" in os.environ:
            pass  # Will print later
    
    # Load base config from YAML
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, "r") as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}
        print(f"Warning: Config file {config_path} not found, using defaults and environment variables")
    
    # Override with environment variables
    # Model configuration
    if "MODEL_NAME" in os.environ:
        if "model" not in config:
            config["model"] = {}
        config["model"]["base_model"] = os.environ["MODEL_NAME"]
        print(f"Model set to: {os.environ['MODEL_NAME']} (from environment)")
    elif "model" in config and "base_model" in config["model"]:
        print(f"Model set to: {config['model']['base_model']} (from config.yaml)")
    else:
        # Default fallback
        if "model" not in config:
            config["model"] = {}
        config["model"]["base_model"] = "deepseek-ai/deepseek-v3"
        print(f"Model set to default: {config['model']['base_model']}")
    
    # Verifier model paths (optional overrides)
    if "PREMISE_VERIFIER_MODEL" in os.environ:
        if "verifier" not in config:
            config["verifier"] = {}
        config["verifier"]["premise_model_path"] = os.environ["PREMISE_VERIFIER_MODEL"]
    
    if "INFERENCE_VERIFIER_MODEL" in os.environ:
        if "verifier" not in config:
            config["verifier"] = {}
        config["verifier"]["inference_model_path"] = os.environ["INFERENCE_VERIFIER_MODEL"]
    
    # Retrieval embedding model (optional override)
    if "RETRIEVAL_EMBEDDING_MODEL" in os.environ:
        if "retrieval" not in config:
            config["retrieval"] = {}
        config["retrieval"]["embedding_model"] = os.environ["RETRIEVAL_EMBEDDING_MODEL"]
    
    # Data paths (optional overrides)
    if "DATA_DIR" in os.environ:
        if "paths" not in config:
            config["paths"] = {}
        config["paths"]["data_dir"] = os.environ["DATA_DIR"]
    
    if "MODELS_DIR" in os.environ:
        if "paths" not in config:
            config["paths"] = {}
        config["paths"]["models_dir"] = os.environ["MODELS_DIR"]
    
    # Training output directory (optional override)
    if "OUTPUT_DIR" in os.environ:
        if "training" not in config:
            config["training"] = {}
        config["training"]["output_dir"] = os.environ["OUTPUT_DIR"]
    
    # Ensure default structure if missing
    if "model" not in config:
        config["model"] = {}
    if "training" not in config:
        config["training"] = {}
    if "data" not in config:
        config["data"] = {}
    if "verifier" not in config:
        config["verifier"] = {}
    if "retrieval" not in config:
        config["retrieval"] = {}
    
    return config


def get_model_name(config: Optional[Dict[str, Any]] = None) -> str:
    """Get the current model name from config."""
    if config is None:
        config = load_config()
    
    return config.get("model", {}).get("base_model", "deepseek-ai/deepseek-v3")


def set_model_in_env(model_name: str, env_path: str = "./.env"):
    """Helper function to set model name in .env file."""
    env_file = Path(env_path)
    
    # Read existing .env if it exists
    env_vars = {}
    if env_file.exists():
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    env_vars[key.strip()] = value.strip()
    
    # Update MODEL_NAME
    env_vars["MODEL_NAME"] = model_name
    
    # Write back
    with open(env_file, "w") as f:
        f.write("# Model Configuration\n")
        f.write("# Change MODEL_NAME to switch between different models easily\n")
        f.write(f"MODEL_NAME={model_name}\n")
        
        # Write other vars if they exist
        for key, value in env_vars.items():
            if key != "MODEL_NAME":
                f.write(f"{key}={value}\n")
    
    print(f"Updated {env_path} with MODEL_NAME={model_name}")


def list_available_models() -> Dict[str, str]:
    """Return a dictionary of recommended model names and descriptions."""
    return {
        "deepseek-ai/deepseek-v3": "DeepSeek V3 (recommended for this project)",
        "deepseek-ai/deepseek-v2": "DeepSeek V2 (alternative)",
        "meta-llama/Llama-2-7b-chat-hf": "Llama 2 7B Chat",
        "meta-llama/Llama-2-13b-chat-hf": "Llama 2 13B Chat",
        "mistralai/Mistral-7B-Instruct-v0.2": "Mistral 7B Instruct",
        "mistralai/Mixtral-8x7B-Instruct-v0.1": "Mixtral 8x7B Instruct",
    }

