# quality: gold
from source.data_model.metadata.Sample import Sample


class SequenceMetadata:
    """
    class modeling the existing knowledge about a receptor_sequence:
        - v gene
        - j gene
        - chain
        - count
        - region_type (e.g. CDR3, CDR1, whole sequence)
        - frame_type (e.g. In, Out, Stop)
        - sample
    """
    def __init__(self, v_gene: str = None, j_gene: str = None, chain: str = None,
                 count: int = None, frame_type: str = None, region_type: str = None,
                 sample: Sample = None):
        self.v_gene = v_gene
        self.j_gene = j_gene
        self.chain = chain
        self.count = count
        self.frame_type = frame_type
        self.region_type = region_type  # should be e.g. CDR, CDR3, whole sequence etc
        # TODO: note that frame_type is optional parameter, specific for adaptive?
        self.sample = sample
        self.custom_params = {}
