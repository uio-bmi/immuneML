from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from sklearn.preprocessing import StandardScaler

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
from immuneML.encodings.preprocessing.FeatureScaler import FeatureScaler
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.util.PathBuilder import PathBuilder


class ProteinEmbeddingEncoder(DatasetEncoder, ABC):
    """
    Abstract base class for protein embedding encoders that handles dataset-type specific logic.
    Subclasses must implement the _embed_sequence_set method.
    """

    def __init__(self, region_type: RegionType, name: str = None, num_processes: int = 1, device: str = 'cpu',
                 batch_size: int = 4096, scale_to_zero_mean: bool = True, scale_to_unit_variance: bool = True):
        super().__init__(name)
        self.region_type = region_type
        self.num_processes = num_processes
        self.device = device
        self.batch_size = batch_size
        self.scale_to_zero_mean = scale_to_zero_mean
        self.scale_to_unit_variance = scale_to_unit_variance

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
        embeddings = self._scale_examples(dataset, embeddings, params)

        encoded_dataset = dataset.clone()
        encoded_dataset.encoded_data = EncodedData(examples=embeddings,
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

        concatenated_embeddings = self._scale_examples(dataset, concatenated_embeddings, params)

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

        examples = self._scale_examples(dataset, np.array(examples), params)

        encoded_dataset = dataset.clone()
        labels = dataset.get_metadata(params.label_config.get_labels_by_name())
        encoded_dataset.encoded_data = EncodedData(examples=examples, labels=labels,
                                                   example_ids=dataset.get_example_ids(),
                                                   encoding=self._get_encoding_name())

        return encoded_dataset

    def _scale_examples(self, dataset: Dataset, examples: np.ndarray, params: EncoderParams) -> np.ndarray:
        if params.learn_model:
            self.scaler = StandardScaler(with_mean=self.scale_to_zero_mean, with_std=self.scale_to_unit_variance)
            examples = CacheHandler.memo_by_params(
                self._get_caching_params(dataset, params, step='scaled'),
                lambda: FeatureScaler.standard_scale_fit(self.scaler, examples, with_mean=self.scale_to_zero_mean))
        else:
            examples = CacheHandler.memo_by_params(
                self._get_caching_params(dataset, params, step='scaled'),
                lambda: FeatureScaler.standard_scale(self.scaler, examples, with_mean=self.scale_to_zero_mean))

        return self._create_memmap_array(examples.shape, examples)

    def _create_memmap_array(self, shape: tuple, data: np.ndarray = None) -> np.ndarray:
        """Creates a memory-mapped array and optionally initializes it with data."""
        import uuid
        dir_path = PathBuilder.build(EnvironmentSettings.get_cache_path() / "memmap_storage")
        memmap_path = dir_path / f"temp_{uuid.uuid4()}.mmap"
        
        memmap_array = np.memmap(memmap_path, dtype='float32', mode='w+', shape=shape)
        if data is not None:
            memmap_array[:] = data[:]
        return memmap_array

    def _avg_sequence_set_embedding(self, embedding: np.ndarray) -> np.ndarray:
        return embedding.mean(axis=0)

    @abstractmethod
    def _embed_sequence_set(self, sequence_set: AIRRSequenceSet, seq_field: str) -> np.ndarray:
        pass

    @abstractmethod
    def _get_encoding_name(self) -> str:
        pass

    @abstractmethod
    def _get_model_link(self) -> str:
        pass

    def _get_caching_params(self, dataset, params: EncoderParams, step: str = None) -> tuple:
        return (dataset.identifier, tuple(params.label_config.get_labels_by_name()), self.scale_to_zero_mean,
                self.scale_to_unit_variance, step, self.region_type.name, self._get_encoding_name(),
                params.learn_model)
