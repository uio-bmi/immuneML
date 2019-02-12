# quality: gold


class SequenceMetadata:
    """
    class modeling the existing knowledge about a sequence:
        - v gene
        - j gene
        - chain
        - count
        - frame_type (e.g. In, Out, Stop)
    """
    def __init__(self, v_gene: str = None, j_gene: str = None, chain: str = None, count: int = None, frame_type: str = None):
        self.v_gene = v_gene
        self.j_gene = j_gene
        self.chain = chain
        self.count = count
        self.frame_type = frame_type
        # TODO: note that frame_type is optional parameter, specific for adaptive?
