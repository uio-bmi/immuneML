from abc import ABC, abstractmethod
import numpy as np

from immuneML.caching.CacheHandler import CacheHandler
from immuneML.data_model.AIRRSequenceSet import AIRRSequenceSet
from immuneML.data_model.EncodedData import EncodedData
from immuneML.data_model.SequenceParams import RegionType
from immuneML.data_model.bnp_util import get_sequence_field_name
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.data_model.datasets.ElementDataset import SequenceDataset, ReceptorDataset
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.environment.SequenceType import SequenceType
from immuneML.util.ParameterValidator import ParameterValidator


class ProteinEmbeddingEncoder(DatasetEncoder, ABC):
    """
    Abstract base class for protein embedding encoders that handles dataset-type specific logic.
    Subclasses must implement the _embed_sequence_set method.
    """

    def __init__(self, region_type: RegionType, name: str = None):
        super().__init__(name)
        self.region_type = region_type

    @staticmethod
    @abstractmethod
    def build_object(dataset: Dataset, **params):
        pass

    def encode(self, dataset: Dataset, params: EncoderParams) -> Dataset:
        cache_params = self._get_caching_params(dataset, params)
        if isinstance(dataset, SequenceDataset):
            return CacheHandler.memo_by_params(cache_params, lambda: self._encode_sequence_dataset(dataset, params))
        elif isinstance(dataset, ReceptorDataset):
            return CacheHandler.memo_by_params(cache_params, lambda: self._encode_receptor_dataset(dataset, params))
        elif isinstance(dataset, RepertoireDataset):
            return CacheHandler.memo_by_params(cache_params, lambda: self._encode_repertoire_dataset(dataset, params))
        else:
            raise RuntimeError(f"{self.__class__.__name__}: {self.name}: invalid dataset type: {type(dataset)}.")

    def _encode_sequence_dataset(self, dataset: SequenceDataset, params: EncoderParams):
        seq_field = get_sequence_field_name(self.region_type, SequenceType.AMINO_ACID)
        embeddings = self._embed_sequence_set(dataset.data, seq_field)

        encoded_dataset = dataset.clone()
        encoded_dataset.encoded_data = EncodedData(examples=np.array(embeddings),
                                                   labels={label.name: getattr(dataset.data, label.name).tolist()
                                                           for label in params.label_config.get_label_objects()},
                                                   example_ids=dataset.data.sequence_id.tolist(),
                                                   encoding=self._get_encoding_name())
        return encoded_dataset

    def _encode_receptor_dataset(self, dataset: ReceptorDataset, params: EncoderParams):
        seq_field = get_sequence_field_name(region_type=self.region_type, sequence_type=SequenceType.AMINO_ACID)

        data = dataset.data

        loci = sorted(list(set(data.locus.tolist())))
        assert len(loci) == 2, (
            f"{self.__class__.__name__}: {self.name}: to encode receptor dataset, it has to include "
            f"two different chains, but got: {loci} instead.")

        embeddings = self._embed_sequence_set(data, seq_field)

        cell_ids = data.cell_id.tolist()
        chain_types = data.locus.tolist()

        chain1_embeddings = {}
        chain2_embeddings = {}

        for i, (cell_id, chain_type) in enumerate(zip(cell_ids, chain_types)):
            if chain_type == loci[0]:
                chain1_embeddings[cell_id] = embeddings[i]
            else:
                chain2_embeddings[cell_id] = embeddings[i]

        assert set(chain1_embeddings.keys()) == set(chain2_embeddings.keys()), \
            f"{self.__class__.__name__}: {self.name}: some receptors are missing one of the chains"

        receptor_ids = list(chain1_embeddings.keys())
        concatenated_embeddings = np.array([
            np.concatenate([chain1_embeddings[cell_id], chain2_embeddings[cell_id]])
            for cell_id in receptor_ids
        ])

        labels = (data.topandas().groupby('cell_id').first()[params.label_config.get_labels_by_name()]
                  .to_dict(orient='list'))

        encoded_dataset = dataset.clone()
        encoded_dataset.encoded_data = EncodedData(
            examples=concatenated_embeddings,
            labels=labels, example_ids=receptor_ids,
            encoding=self._get_encoding_name()
        )

        return encoded_dataset

    def _encode_repertoire_dataset(self, dataset: RepertoireDataset, params: EncoderParams) -> Dataset:
        seq_field = get_sequence_field_name(self.region_type, SequenceType.AMINO_ACID)

        examples = []
        for repertoire in dataset.repertoires:
            examples.append(CacheHandler.memo_by_params((repertoire.identifier, self.__class__.__name__,
                                                         self.region_type.name, self._get_model_link()),
                                                        lambda: self._avg_sequence_set_embedding(
                                                            embedding=self._embed_sequence_set(repertoire.data,
                                                                                               seq_field))))

        encoded_dataset = dataset.clone()
        labels = dataset.get_metadata(params.label_config.get_labels_by_name())
        encoded_dataset.encoded_data = EncodedData(examples=np.array(examples), labels=labels,
                                                   example_ids=dataset.get_example_ids(),
                                                   encoding=self._get_encoding_name())

        return encoded_dataset

    def _avg_sequence_set_embedding(self, embedding: np.ndarray) -> np.ndarray:
        return embedding.mean(axis=1)

    @abstractmethod
    def _embed_sequence_set(self, sequence_set: AIRRSequenceSet, seq_field: str) -> np.ndarray:
        pass

    @abstractmethod
    def _get_encoding_name(self) -> str:
        pass

    @abstractmethod
    def _get_model_link(self) -> str:
        pass

    @abstractmethod
    def _get_caching_params(self, dataset, params: EncoderParams):
        pass
