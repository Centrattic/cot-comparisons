from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """
    Configuration for a model used in CoT generation or analysis.

    This is a minimal config for now - expand as needed for activation
    extraction, API configuration, etc.
    """

    name: str  # Model identifier (e.g., "claude-3-5-sonnet", "gpt-4o")
    provider: str  # API provider: "anthropic", "openai", "together", etc.
    thought_tokens: Optional[int] = None  # Max tokens for CoT
    response_tokens: Optional[int] = None  # Max tokens for response


# Pre-configured models (expand as needed)
MODELS = {
    "claude-3-5-sonnet": ModelConfig(
        name="claude-3-5-sonnet-20241022",
        provider="anthropic",
        thought_tokens=8000,
        response_tokens=1000,
    ),
    "claude-3-opus": ModelConfig(
        name="claude-3-opus-20240229",
        provider="anthropic",
        thought_tokens=8000,
        response_tokens=1000,
    ),
    "gpt-4o": ModelConfig(
        name="gpt-4o",
        provider="openai",
        thought_tokens=8000,
        response_tokens=1000,
    ),
    "deepseek-r1": ModelConfig(
        name="deepseek-reasoner",
        provider="deepseek",
        thought_tokens=16000,
        response_tokens=1000,
    ),
}


def get_model_config(model_name: str) -> ModelConfig:
    """
    Get a model configuration by name.

    Args:
        model_name: Short name (e.g., "claude-3-5-sonnet") or full name.

    Returns:
        ModelConfig for the requested model.

    Raises:
        KeyError: If model not found in registry.
    """
    if model_name in MODELS:
        return MODELS[model_name]

    # Try to find by full name
    for config in MODELS.values():
        if config.name == model_name:
            return config

    raise KeyError(f"Model '{model_name}' not found in registry. Available: {list(MODELS.keys())}")
