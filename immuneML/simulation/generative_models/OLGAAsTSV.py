from npstructures import npdataclass


@npdataclass
class OLGAAsTSV:
    sequence: str
    sequence_aa: str
    v_call: str
    j_call: str
    region_type: str
    frame_type: str
