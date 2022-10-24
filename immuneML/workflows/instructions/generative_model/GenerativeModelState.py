from dataclasses import dataclass
from pathlib import Path

@dataclass
class GenerativeModelState:
    generative_model_units: dict
    result_path: Path = None
    name: str = None