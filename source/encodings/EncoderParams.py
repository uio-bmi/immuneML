from dataclasses import dataclass

from source.environment.LabelConfiguration import LabelConfiguration


@dataclass
class EncoderParams:
    result_path: str
    label_config: LabelConfiguration
    filename: str = ""
    pool_size: int = 4
    model: dict = None
    learn_model: bool = True
    encode_labels: bool = True
