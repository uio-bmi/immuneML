import numpy as np
from sklearn.preprocessing import StandardScaler

from immuneML.caching.CacheHandler import CacheHandler
from immuneML.data_model.EncodedData import EncodedData
from immuneML.data_model.SequenceParams import RegionType
from immuneML.data_model.bnp_util import get_sequence_field_name
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.data_model.datasets.ElementDataset import SequenceDataset, ReceptorDataset
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.preprocessing.FeatureScaler import FeatureScaler
from immuneML.environment.SequenceType import SequenceType
from immuneML.util.EncoderHelper import EncoderHelper
from immuneML.util.ParameterValidator import ParameterValidator


class SequenceLengthEncoder(DatasetEncoder):
    """
    Encodes a dataset based on the length of each receptor sequence in the specified region.

    Each sequence (or chain, in the case of a ReceptorDataset) is encoded as a single
    integer feature representing its length.

    For **SequenceDatasets** each sequence is one example with one feature: its length,
    giving output shape ``[n_sequences, 1]``.

    For **ReceptorDatasets** the two chains of each receptor are paired into a single
    example with two features (one length per chain), giving output shape
    ``[n_receptors, 2]``. The feature names are ``<locus>_length`` for each locus,
    ordered alphabetically (e.g. ``alpha_length``, ``beta_length``).

    **Dataset type:**

    - SequenceDatasets

    - ReceptorDatasets


    **Specification arguments:**

    - region_type (str): Which part of the receptor sequence to measure (e.g. ``imgt_cdr3``).

    - sequence_type (str): Whether to measure amino acid or nucleotide sequence length.
      Valid values: ``amino_acid``, ``nucleotide``. Defaults to ``amino_acid``.

    - scale_to_zero_mean (bool): Whether to scale each feature to zero mean across examples
      after encoding. Defaults to ``True``.

    - scale_to_unit_variance (bool): Whether to scale each feature to unit variance across
      examples after encoding. Defaults to ``True``.


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            encodings:
                my_length_encoder:
                    SequenceLength:
                        region_type: imgt_cdr3
                        sequence_type: amino_acid
                        scale_to_zero_mean: True
                        scale_to_unit_variance: True
    """

    def __init__(self, region_type: RegionType, sequence_type: SequenceType = SequenceType.AMINO_ACID,
                 scale_to_zero_mean: bool = False, scale_to_unit_variance: bool = False,
                 name: str = None):
        super().__init__(name=name)
        self.region_type = region_type
        self.sequence_type = sequence_type
        self.scale_to_zero_mean = scale_to_zero_mean
        self.scale_to_unit_variance = scale_to_unit_variance
        self.scaler = None

    @staticmethod
    def build_object(dataset: Dataset, **params):
        location = SequenceLengthEncoder.__name__
        ParameterValidator.assert_region_type(params, location)
        ParameterValidator.assert_sequence_type(params, location)
        ParameterValidator.assert_type_and_value(params['scale_to_zero_mean'], bool, location,
                                                 'scale_to_zero_mean')
        ParameterValidator.assert_type_and_value(params['scale_to_unit_variance'], bool, location,
                                                 'scale_to_unit_variance')
        return SequenceLengthEncoder(
            region_type=RegionType[params['region_type'].upper()],
            sequence_type=SequenceType[params['sequence_type'].upper()],
            scale_to_zero_mean=params['scale_to_zero_mean'],
            scale_to_unit_variance=params['scale_to_unit_variance'],
            name=params.get('name'),
        )

    def encode(self, dataset: Dataset, params: EncoderParams) -> Dataset:
        cache_params = self._get_caching_params(dataset, params)
        if isinstance(dataset, SequenceDataset):
            return CacheHandler.memo_by_params(cache_params,
                                               lambda: self._encode_sequence_dataset(dataset, params))
        elif isinstance(dataset, ReceptorDataset):
            return CacheHandler.memo_by_params(cache_params,
                                               lambda: self._encode_receptor_dataset(dataset, params))
        else:
            raise RuntimeError(f"{self.__class__.__name__}: {self.name}: unsupported dataset type "
                               f"'{type(dataset).__name__}'. "
                               f"Supported types are SequenceDataset and ReceptorDataset.")

    def _get_lengths(self, sequence_set, seq_field: str) -> np.ndarray:
        """Return a 1-D integer array of sequence lengths for the given region field.

        Lengths are read from the ``EncodedRaggedArray.lengths`` attribute of the
        bionumpy data object, which avoids decoding the character data entirely.
        """
        seq_array = getattr(sequence_set, seq_field)  # EncodedRaggedArray
        return np.asarray(seq_array.lengths, dtype=float)

    def _encode_sequence_dataset(self, dataset: SequenceDataset, params: EncoderParams) -> SequenceDataset:
        seq_field = get_sequence_field_name(self.region_type, self.sequence_type)
        lengths = self._get_lengths(dataset.data, seq_field)  # [n_sequences]
        examples = lengths.reshape(-1, 1)                      # [n_sequences, 1]
        examples = self._scale_examples(examples, params)

        labels = (EncoderHelper.encode_element_dataset_labels(dataset, params.label_config, data=dataset.data)
                  if params.encode_labels else None)

        encoded_dataset = dataset.clone()
        encoded_dataset.encoded_data = EncodedData(
            examples=examples,
            labels=labels,
            example_ids=dataset.data.sequence_id.tolist(),
            feature_names=['sequence_length'],
            encoding=SequenceLengthEncoder.__name__,
        )
        return encoded_dataset

    def _encode_receptor_dataset(self, dataset: ReceptorDataset, params: EncoderParams) -> ReceptorDataset:
        """Encode each receptor as one example with two features: the length of each chain.

        Chains are paired by locus and ordered alphabetically by locus name,
        giving output shape ``[n_receptors, 2]`` with feature names
        ``<locus>_length`` (e.g. ``alpha_length``, ``beta_length``).

        Relies on import-time ordering: the two chains of each receptor are stored
        consecutively and sorted by locus within each pair (see
        ``ImportHelper.pair_receptor_chains``). No re-sorting is done here, so
        the receptor order from the original dataset is preserved.
        """
        seq_field = get_sequence_field_name(self.region_type, self.sequence_type)
        data = dataset.data

        lengths = self._get_lengths(data, seq_field)  # [n_chains]
        receptor_ids, loci, mask1, mask2 = EncoderHelper.get_receptor_chain_masks(dataset)

        examples = np.column_stack([lengths[mask1], lengths[mask2]])  # [n_receptors, 2]
        examples = self._scale_examples(examples, params)

        labels = (EncoderHelper.encode_element_dataset_labels(dataset, params.label_config, data=data, mask=mask1)
                  if params.encode_labels else None)

        encoded_dataset = dataset.clone()
        encoded_dataset.encoded_data = EncodedData(
            examples=examples,
            labels=labels,
            example_ids=receptor_ids,
            feature_names=[f'{locus}_length' for locus in loci],
            encoding=SequenceLengthEncoder.__name__,
        )
        return encoded_dataset

    def _scale_examples(self, examples: np.ndarray, params: EncoderParams) -> np.ndarray:
        if not self.scale_to_zero_mean and not self.scale_to_unit_variance:
            return examples
        if params.learn_model:
            self.scaler = StandardScaler(with_mean=self.scale_to_zero_mean,
                                         with_std=self.scale_to_unit_variance)
            return FeatureScaler.standard_scale_fit(self.scaler, examples,
                                                    with_mean=self.scale_to_zero_mean)
        else:
            return FeatureScaler.standard_scale(self.scaler, examples,
                                                with_mean=self.scale_to_zero_mean)

    def _get_caching_params(self, dataset: Dataset, params: EncoderParams, step: str = None) -> tuple:
        return (dataset.identifier,
                tuple(params.label_config.get_labels_by_name()),
                self.region_type.name,
                self.sequence_type.name,
                self.scale_to_zero_mean,
                self.scale_to_unit_variance,
                params.learn_model,
                step)