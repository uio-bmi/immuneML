from enum import Enum
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from immuneML.analysis.data_manipulation.NormalizationType import NormalizationType
from immuneML.caching.CacheHandler import CacheHandler
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.abundance_encoding.AbundanceEncoderHelper import AbundanceEncoderHelper
from immuneML.encodings.kmer_frequency.KmerFreqRepertoireEncoder import KmerFreqRepertoireEncoder
from immuneML.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder
from immuneML.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType import SequenceEncodingType
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.EncoderHelper import EncoderHelper
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.ReadsType import ReadsType


class KmerAbundanceEncoder(DatasetEncoder):
    """
    This encoder is related to the :py:obj:`~immuneML.encodings.abundance_encoding.SequenceAbundanceEncoder.SequenceAbundanceEncoder`,
    but identifies label-associated subsequences (k-mers) instead of full label-associated sequences.

    This encoder represents the repertoires as vectors where:

    - the first element corresponds to the number of label-associated k-mers found in a repertoire
    - the second element is the total number of unique k-mers per repertoire

    The label-associated k-mers are determined based on a one-sided Fisher's exact test.

    The encoder also writes out files containing the contingency table used for fisher's exact test,
    the resulting p-values, and the significantly abundant k-mers.

    Note: to use this encoder, it is necessary to explicitly define the positive class for the label when defining the label
    in the instruction. With positive class defined, it can then be determined which sequences are indicative of the positive class.
    See :ref:`Reproduction of the CMV status predictions study` for an example using :py:obj:`~immuneML.encodings.abundance_encoding.SequenceAbundanceEncoder.SequenceAbundanceEncoder`.

    Specification arguments:

    - p_value_threshold (float): The p value threshold to be used by the statistical test.

    - sequence_encoding (:py:mod:`~immuneML.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType`): The type of k-mers that are used. The simplest (default) sequence_encoding is :py:mod:`~immuneML.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType.CONTINUOUS_KMER`, which uses contiguous subsequences of length k to represent the k-mers. When gapped k-mers are used (:py:mod:`~immuneML.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType.GAPPED_KMER`, :py:mod:`~immuneML.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType.GAPPED_KMER`), the k-mers may contain gaps with a size between min_gap and max_gap, and the k-mer length is defined as a combination of k_left and k_right. When IMGT k-mers are used (:py:mod:`~immuneML.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType.IMGT_CONTINUOUS_KMER`, :py:mod:`~immuneML.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType.IMGT_GAPPED_KMER`), IMGT positional information is taken into account (i.e. the same sequence in a different position is considered to be a different k-mer).

    - k (int): Length of the k-mer (number of amino acids) when ungapped k-mers are used. The default value for k is 3.

    - k_left (int): When gapped k-mers are used, k_left indicates the length of the k-mer left of the gap. The default value for k_left is 1.

    - k_right (int): Same as k_left, but k_right determines the length of the k-mer right of the gap. The default value for k_right is 1.

    - min_gap (int): Minimum gap size when gapped k-mers are used. The default value for min_gap is 0.

    - max_gap: (int): Maximum gap size when gapped k-mers are used. The default value for max_gap is 0.


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_sa_encoding:
            KmerAbundance:
                p_value_threshold: 0.05
                threads: 8

    """

    RELEVANT_SEQUENCE_ABUNDANCE = "relevant_sequence_abundance"
    TOTAL_SEQUENCE_ABUNDANCE = "total_sequence_abundance"

    def __init__(self, p_value_threshold: float, sequence_encoding: SequenceEncodingType, k: int,
                 k_left: int, k_right: int, min_gap: int, max_gap: int, name: str = None):
        self.name = name
        self.p_value_threshold = p_value_threshold

        self.kmer_frequency_params = {"normalization_type": NormalizationType.BINARY, "reads": ReadsType.UNIQUE,
                                      "sequence_encoding": sequence_encoding, "k": k, "k_left": k_left, "k_right": k_right,
                                      "min_gap": min_gap, "max_gap": max_gap, "scale_to_unit_variance": False,
                                      "scale_to_zero_mean": False, "sequence_type": EnvironmentSettings.get_sequence_type()}

        self.relevant_indices_path = None
        self.relevant_sequence_path = None
        self.contingency_table_path = None
        self.p_values_path = None

        self.context = None

        self.full_kmer_set = None
        self.raw_distance_matrix_np = None

    @staticmethod
    def _prepare_parameters(p_value_threshold: float, sequence_encoding: str, k: int,
                            k_left: int, k_right: int, min_gap: int, max_gap: int, name: str = None):
        ParameterValidator.assert_type_and_value(p_value_threshold, float, "KmerAbundanceEncoder", "p_value_threshold", min_inclusive=0,
                                                 max_inclusive=1)

        assert sequence_encoding.upper() != SequenceEncodingType.IDENTITY.name, "KmerAbundanceEncoder: sequence encoding type 'identity' is not a valid option for this encoder. To encode a dataset based on the presence or absence of complete sequences, please use SequenceAbundanceEncoder or CompAIRRSequenceAbundanceEncoder instead."

        kmerfreq_params = KmerFrequencyEncoder._prepare_parameters(normalization_type="binary", reads="unique",
                                                                   sequence_encoding=sequence_encoding,
                                                                   k=k, k_left=k_left, k_right=k_right, min_gap=min_gap,
                                                                   max_gap=max_gap, sequence_type=EnvironmentSettings.get_sequence_type().name)

        return {
            "p_value_threshold": p_value_threshold,
            "sequence_encoding": kmerfreq_params["sequence_encoding"],
            "k": kmerfreq_params["k"],
            "k_left": kmerfreq_params["k_left"],
            "k_right": kmerfreq_params["k_right"],
            "min_gap": kmerfreq_params["min_gap"],
            "max_gap": kmerfreq_params["max_gap"],
            "name": name
        }

    @staticmethod
    def build_object(dataset, **params):
        assert isinstance(dataset, RepertoireDataset), "KmerAbundanceEncoder: this encoding only works on repertoire datasets."
        prepared_params = KmerAbundanceEncoder._prepare_parameters(**params)
        return KmerAbundanceEncoder(**prepared_params)

    def encode(self, dataset, params: EncoderParams):
        EncoderHelper.check_positive_class_labels(params.label_config, KmerAbundanceEncoder.__name__)

        self._prepare_kmer_presence_data(dataset, params)
        return self._encode_data(dataset, params)

    def _prepare_kmer_presence_data(self, dataset, params):
        full_dataset = EncoderHelper.get_current_dataset(dataset, self.context)

        kmer_encoded_data = self._get_kmer_encoded_data(full_dataset, params)

        self.full_kmer_set = np.array(kmer_encoded_data.feature_names)
        self.kmer_presence_matrix = kmer_encoded_data.examples.toarray().T
        self.matrix_repertoire_ids = np.array(kmer_encoded_data.example_ids)

    def _get_kmer_encoded_data(self, full_dataset, params):
        return CacheHandler.memo_by_params(
            self._build_kmer_presence_params(full_dataset, self.kmer_frequency_params),
            lambda: self._compute_kmer_encoded_data(full_dataset, params))

    def _build_kmer_presence_params(self, dataset, kmer_frequency_params):
        kmer_params = [(key, value) if not issubclass(type(value), Enum) else (key, value.value) for key, value in kmer_frequency_params.items()]

        return (("dataset_identifier", dataset.identifier),
                ("repertoire_ids", tuple(dataset.get_repertoire_ids())),
                ("normalization_type", kmer_frequency_params["normalization_type"]),
                ("kmer_params", tuple(kmer_params)))

    def _compute_kmer_encoded_data(self, dataset, params):
        kmer_frequency_encoder = KmerFreqRepertoireEncoder(**self.kmer_frequency_params)

        encoder_params = EncoderParams(result_path=params.result_path / "kmer_frequency",
                                       label_config=params.label_config,
                                       pool_size=params.pool_size,
                                       learn_model=True,
                                       encode_labels=params.encode_labels)

        encoded_dataset = kmer_frequency_encoder.encode(dataset, encoder_params)

        return encoded_dataset.encoded_data

    def _encode_data(self, dataset: RepertoireDataset, params: EncoderParams):
        label = params.label_config.get_label_objects()[0]

        examples = self._calculate_abundance_matrix(dataset, self.kmer_presence_matrix, self.matrix_repertoire_ids, params)

        encoded_data = EncodedData(examples, dataset.get_metadata([label.name]) if params.encode_labels else None, dataset.get_repertoire_ids(),
                                   [KmerAbundanceEncoder.RELEVANT_SEQUENCE_ABUNDANCE,
                                    KmerAbundanceEncoder.TOTAL_SEQUENCE_ABUNDANCE],
                                   example_weights=dataset.get_example_weights(),
                                   encoding=KmerAbundanceEncoder.__name__,
                                   info={"relevant_sequence_path": self.relevant_sequence_path,
                                         "contingency_table_path": self.contingency_table_path,
                                         "p_values_path": self.p_values_path})

        encoded_dataset = RepertoireDataset(labels=dataset.labels, encoded_data=encoded_data, repertoires=dataset.repertoires)

        return encoded_dataset

    def _calculate_abundance_matrix(self, dataset: RepertoireDataset, sequence_presence_matrix, matrix_repertoire_ids,
                                    params: EncoderParams):
        relevant = np.isin(matrix_repertoire_ids, dataset.get_repertoire_ids())
        sequence_presence_matrix = sequence_presence_matrix[:, relevant]
        matrix_repertoire_ids = matrix_repertoire_ids[relevant]

        is_positive_class = AbundanceEncoderHelper.check_is_positive_class(dataset, matrix_repertoire_ids, params.label_config)

        relevant_sequence_indices, file_paths = AbundanceEncoderHelper.get_relevant_sequence_indices(sequence_presence_matrix, is_positive_class,
                                                                                                     self.p_value_threshold,
                                                                                                     self.relevant_indices_path, params,
                                                                                                     cache_params=(dataset.get_repertoire_ids(),
                                                                                                                   self.kmer_frequency_params))
        self._write_relevant_kmers_csv(relevant_sequence_indices, params.result_path)
        self._set_file_paths(file_paths)

        abundance_matrix = AbundanceEncoderHelper.build_abundance_matrix(sequence_presence_matrix, matrix_repertoire_ids,
                                                                         dataset.get_repertoire_ids(), relevant_sequence_indices)

        return abundance_matrix

    def _set_file_paths(self, file_paths):
        self.relevant_indices_path = file_paths["relevant_indices_path"]
        self.contingency_table_path = file_paths["contingency_table_path"] if "contingency_table_path" in file_paths else None
        self.p_values_path = file_paths["p_values_path"] if "p_values_path" in file_paths else None

    def _write_relevant_kmers_csv(self, relevant_sequence_indices, result_path):
        relevant_kmers = self.full_kmer_set[relevant_sequence_indices]

        if self.relevant_sequence_path is None:
            self.relevant_sequence_path = result_path / 'relevant_sequences.csv'

        df = pd.DataFrame(relevant_kmers, columns=["k-mer"])
        df.to_csv(self.relevant_sequence_path, sep=",", index=False)

    def set_context(self, context: dict):
        self.context = context
        return self

    def store(self, encoded_dataset, params: EncoderParams):
        EncoderHelper.store(encoded_dataset, params)

    @staticmethod
    def export_encoder(path: Path, encoder) -> Path:
        encoder_file = DatasetEncoder.store_encoder(encoder, path / "encoder.pickle")
        return encoder_file

    def get_additional_files(self) -> List[Path]:
        return [file for file in [self.relevant_indices_path, self.relevant_sequence_path, self.contingency_table_path, self.p_values_path] if file]

    @staticmethod
    def load_encoder(encoder_file: Path):
        encoder = DatasetEncoder.load_encoder(encoder_file)
        encoder.relevant_indices_path = DatasetEncoder.load_attribute(encoder, encoder_file, "relevant_indices_path")

        return encoder
