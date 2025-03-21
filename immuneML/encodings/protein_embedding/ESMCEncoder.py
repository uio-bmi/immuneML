import numpy as np

from immuneML.data_model.AIRRSequenceSet import AIRRSequenceSet
from immuneML.data_model.SequenceParams import RegionType
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.protein_embedding.ProteinEmbeddingEncoder import ProteinEmbeddingEncoder
from immuneML.util.ParameterValidator import ParameterValidator


class ESMCEncoder(ProteinEmbeddingEncoder):
    """
    Encoder based on a pretrained protein language model by Hayes et al. 2025. The used transformer model is
    "esmc_300m".

    Original publication:
    Hayes, T., Rao, R., Akin, H., Sofroniew, N. J., Oktay, D., Lin, Z., Verkuil, R., Tran, V. Q., Deaton, J.,
    Wiggert, M., Badkundri, R., Shafkat, I., Gong, J., Derry, A., Molina, R. S., Thomas, N., Khan, Y. A.,
    Mishra, C., Kim, C., … Rives, A. (2025). Simulating 500 million years of evolution with a language model.
    Science, 387(6736), 850–858. https://doi.org/10.1126/science.ads0018

    Original GitHub repository with license information: https://github.com/evolutionaryscale/esm

    **Dataset type:**

    - SequenceDatasets

    - ReceptorDatasets

    - RepertoireDatasets

    **Specification arguments:**

    - region_type (RegionType): Which part of the receptor sequence to encode. Defaults to IMGT_CDR3.

    - device (str): Which device to use for model inference - 'cpu', 'cuda', 'mps' - as defined by pytorch.
      Defaults to 'cpu'.

    - num_processes (int): Number of processes to use for parallel processing. Defaults to 1.

    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            encodings:
                my_emsc_encoder:
                    ESMC::
                        region_type: IMGT_CDR3
                        device: cpu
                        num_processes: 4

    """

    def __init__(self, name: str = None, region_type: RegionType = RegionType.IMGT_CDR3, device: str = 'cpu',
                 num_processes: int = 1):
        super().__init__(region_type, name, num_processes, device)
        self.transformer_link = 'esmc_300m'

    @staticmethod
    def build_object(dataset: Dataset, **params):
        ParameterValidator.assert_region_type(params, ESMCEncoder.__name__)
        return ESMCEncoder(**{**params, 'region_type': RegionType[params['region_type'].upper()]})

    def _embed_sequence_set(self, sequence_set: AIRRSequenceSet, seq_field: str) -> np.ndarray:
        from esm.models.esmc import ESMC
        import torch

        model = ESMC.from_pretrained(self.transformer_link).to(self.device)
        sequences = getattr(sequence_set, seq_field)
        sequences = sequences.tolist()

        tokens = model._tokenize(sequences)
        sequence_id = tokens != model.tokenizer.pad_token_id

        with torch.no_grad():
            output = model.forward(
                sequence_tokens=tokens,
                sequence_id=sequence_id
            )

        return output.embeddings.cpu().numpy().mean(axis=1)

    def _get_encoding_name(self) -> str:
        return f"ESMC({self.transformer_link})"

    def _get_model_link(self) -> str:
        return self.transformer_link

    def _get_caching_params(self, dataset, params: EncoderParams):
        cache_params = (dataset.identifier, self.__class__.__name__, self.region_type.name, self._get_model_link())
        return cache_params
