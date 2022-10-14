from bionumpy import AminoAcidArray, DNAArray
from bionumpy.bnpdataclass import bnpdataclass


@bnpdataclass
class GenModelAsTSV:
    sequence_aa: AminoAcidArray
    sequence: DNAArray
    v_call: str
    j_call: str
    region_type: str
    frame_type: str
