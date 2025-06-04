import numpy as np

from immuneML.caching.CacheHandler import CacheHandler
from immuneML.data_model.EncodedData import EncodedData
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams


class ShannonDiversityEncoder(DatasetEncoder):
    """
    ShannonDiversity encoder calculates the Shannon diversity index for each repertoire in a dataset. The diversity is
    computed as:

    .. math::

        diversity = - \\sum_{i=1}^{n} p_i \\log(p_i)

    where :math:`p_i` is the clonal count for each unique sequence in the repertoire (from duplicate_count field)
    divided by the total clonal counts, and :math:`n` is the total number of clonotypes (sequences) in the repertoire.


    **Dataset type:**

    - RepertoireDataset

    **Specification arguments:**

    No arguments are needed for this encoder.

    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            encodings:
                shannon_div_enc: ShannonDiversity

    """

    def __init__(self, name: str = None):
        super().__init__(name=name)

    @staticmethod
    def build_object(dataset: Dataset, **params):
        assert isinstance(dataset, RepertoireDataset), \
            f"{ShannonDiversityEncoder.__name__}: Dataset must be of type RepertoireDataset, but got {type(dataset)}."
        return ShannonDiversityEncoder(**params)

    def encode(self, dataset, params: EncoderParams) -> Dataset:
        assert isinstance(dataset, RepertoireDataset), \
            f"{ShannonDiversityEncoder.__name__}: Dataset must be of type RepertoireDataset, but got {type(dataset)}."

        examples = CacheHandler.memo_by_params((dataset.identifier, ShannonDiversityEncoder.__name__,
                                                params.label_config.get_labels_by_name() if params.encode_labels else ''),
                                               lambda: self._encode(dataset, params))

        encoded_dataset = dataset.clone()
        encoded_dataset.encoded_data = EncodedData(examples=examples,
                                                   labels=dataset.get_metadata(params.label_config.get_labels_by_name()) if params.encode_labels else {},
                                                   example_ids=dataset.get_example_ids(),
                                                   encoding=ShannonDiversityEncoder.__name__)
        return encoded_dataset

    def _encode(self, dataset: RepertoireDataset, params: EncoderParams) -> np.ndarray:
        entropies = []
        for repertoire in dataset.repertoires:
            data = repertoire.data
            probabilities = data.duplicate_count / np.sum(data.duplicate_count)
            entropy = -1. * np.sum(probabilities * np.log(probabilities))
            entropies.append(entropy)
        return np.array(entropies).reshape(-1, 1)
