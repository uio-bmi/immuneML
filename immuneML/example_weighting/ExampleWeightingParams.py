from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExampleWeightingParams:
    result_path: Path
    pool_size: int = 4
    learn_model: bool = True
