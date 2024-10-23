from dataclasses import dataclass
from pathlib import Path

from immuneML.data_model.SequenceParams import RegionType
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.environment.SequenceType import SequenceType


@dataclass
class EncoderParams:
    result_path: Path = None
    label_config: LabelConfiguration = None
    pool_size: int = 4
    model: dict = None
    learn_model: bool = True
    encode_labels: bool = True
    sequence_type: SequenceType = SequenceType.AMINO_ACID
    region_type: RegionType = RegionType.IMGT_CDR3

    def get_sequence_field_name(self):
        return self.region_type.value + "_aa" if self.sequence_type == SequenceType.AMINO_ACID else ""

    def get_seq_name_for_seq_object(self):
        return 'sequence' if self.sequence_type == SequenceType.NUCLEOTIDE else "sequence_aa"
