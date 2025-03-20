import numpy as np

from immuneML.data_model.AIRRSequenceSet import AIRRSequenceSet
from immuneML.data_model.SequenceParams import RegionType
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.protein_embedding.ProteinEmbeddingEncoder import ProteinEmbeddingEncoder
from immuneML.util.ParameterValidator import ParameterValidator


class ESMCEncoder(ProteinEmbeddingEncoder):

    def __init__(self, name: str = None, region_type: RegionType = RegionType.IMGT_CDR3, device: str = 'cpu',
                 num_processes: int = 1):
        super().__init__(region_type, name, num_processes, device)
        self.transformer_link = 'esmc_300m'

    @staticmethod
    def build_object(dataset: Dataset, **params):
        ParameterValidator.assert_region_type(params, ESMCEncoder.__name__)
        return ESMCEncoder(**{**params, 'region_type': RegionType[params['region_type'].upper()]})

    def _embed_sequence_set(self, sequence_set: AIRRSequenceSet, seq_field: str) -> np.ndarray:
        pass

    def _get_encoding_name(self) -> str:
        return "ESMC"

    def _get_model_link(self) -> str:
        return self.transformer_link

    def _get_caching_params(self, dataset, params: EncoderParams):
        pass
