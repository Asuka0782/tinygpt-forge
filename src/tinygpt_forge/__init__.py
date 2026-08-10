"""TinyGPT Forge public package."""

from tinygpt_forge.config import ModelConfig
from tinygpt_forge.model.gpt import CausalLMOutput, TinyGPT

__all__ = ["CausalLMOutput", "ModelConfig", "TinyGPT"]
__version__ = "0.0.1"
