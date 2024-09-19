from dataclasses import dataclass
from pathlib import Path

from immuneML.data_model.SequenceParams import RegionType
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.environment.SequenceType import SequenceType


@dataclass
class EncoderParams:
    result_path: Path
    label_config: LabelConfiguration
    pool_size: int = 4
    model: dict = None
    learn_model: bool = True
    encode_labels: bool = True
    sequence_type: SequenceType = SequenceType.AMINO_ACID
    region_type: RegionType = RegionType.IMGT_CDR3
