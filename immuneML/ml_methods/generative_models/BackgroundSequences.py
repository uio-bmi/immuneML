from bionumpy import AminoAcidEncoding
from bionumpy.bnpdataclass import bnpdataclass

from immuneML.data_model.AIRRSequenceSet import DNANEncoding
from immuneML.environment.SequenceType import SequenceType


@bnpdataclass
class BackgroundSequences:
    sequence_aa: AminoAcidEncoding
    sequence: DNANEncoding
    v_call: str
    j_call: str
    region_type: str
    frame_type: str
    p_gen: float
    from_default_model: int
    duplicate_count: int
    locus: str

    def get_sequence(self, sequence_type: SequenceType = SequenceType.AMINO_ACID):
        if sequence_type == SequenceType.AMINO_ACID:
            return self.sequence_aa
        else:
            return self.sequence
