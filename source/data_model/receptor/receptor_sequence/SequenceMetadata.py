# quality: gold
from source.data_model.receptor.receptor_sequence.Chain import Chain
from source.data_model.receptor.receptor_sequence.SequenceFrameType import SequenceFrameType

class SequenceMetadata:
    """
    class modeling the existing knowledge about a receptor_sequence, should be stored according to
    IMGT gene nomenclature (human can be found here:
    http://www.imgt.org/IMGTrepertoire/index.php?section=LocusGenes&repertoire=genetable&species=human&group=TRBV):
        - v subgroup
        - v gene
        - v allele
        - j subgroup
        - j gene
        - j allele
        - chain
        - count
        - region_type (e.g. CDR3, CDR1, whole sequence)
        - frame_type (e.g. In, Out, Stop)
        - sample
        - custom params (dictionary with custom sequence information)
    """
    def __init__(self,
                 v_subgroup: str = None, v_gene: str = None, v_allele: str = None,
                 j_subgroup: str = None, j_gene: str = None, j_allele: str = None,
                 chain=None,
                 count: int = None,
                 frame_type: str = SequenceFrameType.IN.name,
                 region_type: str = None,
                 cell_id: str = None,
                 custom_params: dict = None):
        self.v_subgroup = v_subgroup
        self.v_gene = v_gene
        self.v_allele = v_allele
        self.j_subgroup = j_subgroup
        self.j_gene = j_gene
        self.j_allele = j_allele
        self.chain = Chain[chain] if chain and isinstance(chain, str) else chain if isinstance(chain, Chain) else None
        self.count = int(float(count)) if isinstance(count, str) else count
        self.frame_type = frame_type
        self.region_type = region_type  # should be e.g. CDR, CDR3, whole sequence etc
        self.cell_id = cell_id
        self.custom_params = custom_params if custom_params is not None else {}
