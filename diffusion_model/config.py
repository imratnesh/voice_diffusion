# Model configuration and utilities
from dataclasses import dataclass

@dataclass
class DiffusionConfig:
    pretrained_model: str = "facebook/musicgen-medium"
    output_dir: str = "outputs/"
    sampling_rate: int = 32000

