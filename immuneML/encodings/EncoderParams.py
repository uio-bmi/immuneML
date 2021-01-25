from dataclasses import dataclass
from pathlib import Path

from immuneML.environment.LabelConfiguration import LabelConfiguration


@dataclass
class EncoderParams:
    result_path: Path
    label_config: LabelConfiguration
    filename: str = ""
    pool_size: int = 4
    model: dict = None
    learn_model: bool = True
    encode_labels: bool = True
