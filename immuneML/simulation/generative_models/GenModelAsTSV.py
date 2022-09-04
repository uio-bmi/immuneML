from npstructures import npdataclass


@npdataclass
class GenModelAsTSV:
    sequence_aa: str
    sequence: str
    v_call: str
    j_call: str
    region_type: str
    frame_type: str
