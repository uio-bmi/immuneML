import dataclasses
from typing import Union

from olga.load_model import GenerativeModelVJ, GenerativeModelVDJ, GenomicData
from olga.sequence_generation import SequenceGenerationVDJ, SequenceGenerationVJ


@dataclasses.dataclass
class InternalOlgaModel:
    sequence_gen_model: Union[SequenceGenerationVDJ, SequenceGenerationVJ] = None
    v_gene_mapping: list = None
    j_gene_mapping: list = None
    genomic_data: GenomicData = None
    olga_gen_model: Union[GenerativeModelVDJ, GenerativeModelVJ] = None
