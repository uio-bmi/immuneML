# quality: gold
from typing import Union, List

from immuneML.data_model.receptor.RegionType import RegionType
from immuneML.data_model.receptor.receptor_sequence.Chain import Chain
from immuneML.data_model.receptor.receptor_sequence.SequenceFrameType import SequenceFrameType


class SequenceMetadata:
    """
    class modeling the existing knowledge about a receptor_sequence, should be stored according to
    AIRR nomenclature
        - v call
        - j call
        - chain
        - duplicate_count
        - region_type (e.g. IMGT_CDR3, IMGT_CDR1, FULL_SEQUENCE)
        - frame_type (e.g. IN, OUT, STOP)
        - sample
        - custom params (dictionary with custom sequence information)

    """

    def __init__(self, v_call: str = None, j_call: str = None, chain=None, duplicate_count: int = None,
                 frame_type: Union[SequenceFrameType, str] = '',
                 region_type: str = None, cell_id: str = None, custom_params: dict = None):
        self.v_call = v_call
        self.j_call = j_call
        self.chain = Chain.get_chain(chain) if chain and isinstance(chain, str) else chain if isinstance(chain,
                                                                                                         Chain) else None
        self.duplicate_count = int(float(duplicate_count)) if isinstance(duplicate_count, str) else duplicate_count
        self.frame_type = SequenceFrameType(frame_type.upper()) \
            if frame_type and isinstance(frame_type, str) and frame_type != 'nan' else frame_type if isinstance(
            frame_type, SequenceFrameType) else SequenceFrameType.UNDEFINED
        self.region_type = RegionType(region_type.upper()) \
            if region_type and isinstance(region_type, str) and region_type != 'nan' else region_type if isinstance(
            region_type, RegionType) else None
        self.cell_id = cell_id
        self.custom_params = custom_params if custom_params is not None else {}

    @property
    def v_gene(self):
        return self.v_call.split("*")[0]

    @property
    def j_gene(self):
        return self.j_call.split("*")[0]

    def get_attribute(self, name: str):
        """Returns the attribute value if attribute is present either directly or in custom_params, otherwise returns
        None"""
        if hasattr(self, name):
            return getattr(self, name)
        elif name in self.custom_params:
            return self.custom_params[name]
        else:
            return None

    def get_all_attribute_names(self) -> List[str]:
        return [el for el in vars(self) if el != 'custom_params'] + list(self.custom_params.keys())
