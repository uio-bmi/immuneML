from bionumpy import AminoAcidEncoding, DNAEncoding
from bionumpy.bnpdataclass import bnpdataclass

from immuneML.environment.SequenceType import SequenceType


@bnpdataclass
class GenModelAsTSV:
    sequence_aa: AminoAcidEncoding
    sequence: DNAEncoding
    v_call: str
    j_call: str
    region_type: str
    frame_type: str

    def get_sequence(self, sequence_type: SequenceType = SequenceType.AMINO_ACID):
        if sequence_type == SequenceType.AMINO_ACID:
            return self.sequence_aa
        else:
            return self.sequence