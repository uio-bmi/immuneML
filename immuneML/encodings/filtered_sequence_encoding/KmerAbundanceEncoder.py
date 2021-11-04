from enum import Enum
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from immuneML.analysis.data_manipulation.NormalizationType import NormalizationType
from immuneML.caching.CacheHandler import CacheHandler
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.filtered_sequence_encoding.AbundanceEncoderHelper import AbundanceEncoderHelper
from immuneML.encodings.kmer_frequency.KmerFreqRepertoireEncoder import KmerFreqRepertoireEncoder
from immuneML.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder
from immuneML.encodings.kmer_frequency.ReadsType import ReadsType
from immuneML.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType import SequenceEncodingType
from immuneML.environment.SequenceType import SequenceType
from immuneML.util.EncoderHelper import EncoderHelper
from immuneML.util.ParameterValidator import ParameterValidator
from scripts.specification_util import update_docs_per_mapping




class KmerAbundanceEncoder(DatasetEncoder):
    """
    This encoder works similarly to the :py:obj:`~immuneML.encodings.filtered_sequence_encoding.SequenceAbundanceEncoder.SequenceAbundanceEncoder`,
    but internally uses `CompAIRR <https://github.com/uio-bmi/compairr/>`_ to accelerate core computations.

    This encoder represents the repertoires as vectors where:

    - the first element corresponds to the number of label-associated clonotypes
    - the second element is the total number of unique clonotypes

    To determine what clonotypes (with or without matching V/J genes) are label-associated
    based on a statistical test. The statistical test used is Fisher's exact test (one-sided).

    Reference: Emerson, Ryan O. et al.
    ‘Immunosequencing Identifies Signatures of Cytomegalovirus Exposure History and HLA-Mediated Effects on the T Cell Repertoire’.
    Nature Genetics 49, no. 5 (May 2017): 659–65. `doi.org/10.1038/ng.3822 <https://doi.org/10.1038/ng.3822>`_.

    Note: to use this encoder, it is necessary to explicitly define the positive class for the label when defining the label
    in the instruction. With positive class defined, it can then be determined which sequences are indicative of the positive class.
    See :ref:`Reproduction of the CMV status predictions study` for an example using :py:obj:`~immuneML.encodings.filtered_sequence_encoding.SequenceAbundanceEncoder.SequenceAbundanceEncoder`.

    Arguments:

        p_value_threshold (float): The p value threshold to be used by the statistical test.

        compairr_path (Path): optional path to the CompAIRR executable. If not given, it is assumed that CompAIRR
        has been installed such that it can be called directly on the command line with the command 'compairr',
        or that it is located at /usr/local/bin/compairr.

        ignore_genes (bool): Whether to ignore V and J gene information. If False, the V and J genes between two receptor chains
        have to match. If True, gene information is ignored. By default, ignore_genes is False.

        sequence_batch_size (int): The number of sequences in a batch when comparing sequences across repertoires, typically 100s of thousands.
        This does not affect the results of the encoding, only the speed and memory usage.

        threads (int): The number of threads to use for parallelization. This does not affect the results of the encoding, only the speed.
        The default number of threads is 8.


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

    def __init__(self, p_value_threshold: float, sequence_encoding: SequenceEncodingType, k: int = 0,
                 k_left: int = 0, k_right: int = 0, min_gap: int = 0, max_gap: int = 0, name: str = None):
        self.name = name
        self.p_value_threshold = p_value_threshold

        self.kmer_frequency_params = {"normalization_type": NormalizationType.BINARY, "reads": ReadsType.UNIQUE,
                                      "sequence_encoding": sequence_encoding, "k": k, "k_left": k_left, "k_right": k_right,
                                      "min_gap": min_gap, "max_gap": max_gap, "scale_to_unit_variance": False,
                                      "scale_to_zero_mean": False, "sequence_type": SequenceType.AMINO_ACID}

        self.relevant_indices_path = None
        self.relevant_sequence_path = None
        self.contingency_table_path = None
        self.p_values_path = None

        self.context = None

        self.full_kmer_set = None
        self.raw_distance_matrix_np = None

    @staticmethod
    def _prepare_parameters(p_value_threshold: float, sequence_encoding: str, k: int = 0,
                 k_left: int = 0, k_right: int = 0, min_gap: int = 0, max_gap: int = 0, name: str = None):
        ParameterValidator.assert_type_and_value(p_value_threshold, float, "KmerAbundanceEncoder", "p_value_threshold", min_inclusive=0, max_inclusive=1)

        kmerfreq_params = KmerFrequencyEncoder._prepare_parameters(normalization_type="binary", reads= "unique",
                                                                   sequence_encoding=sequence_encoding,
                                                                   k= k, k_left=k_left, k_right=k_right, min_gap=min_gap,
                                                                   max_gap=max_gap, sequence_type="amino_acid")

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
        AbundanceEncoderHelper.check_labels(params.label_config, KmerAbundanceEncoder.__name__)

        self._prepare_kmer_presence_data(dataset, params)
        encoded_dataset = self._encode_data(dataset, params)

        return encoded_dataset

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

        examples = self.calculate_abundance_matrix(dataset, self.kmer_presence_matrix, self.matrix_repertoire_ids, params)

        encoded_data = EncodedData(examples, dataset.get_metadata([label.name]) if params.encode_labels else None, dataset.get_repertoire_ids(),
                                   [KmerAbundanceEncoder.RELEVANT_SEQUENCE_ABUNDANCE,
                                    KmerAbundanceEncoder.TOTAL_SEQUENCE_ABUNDANCE],
                                   encoding=KmerAbundanceEncoder.__name__,
                                   info={"relevant_sequence_path": self.relevant_sequence_path,
                                         "contingency_table_path": self.contingency_table_path,
                                         "p_values_path": self.p_values_path})

        encoded_dataset = RepertoireDataset(labels=dataset.labels, encoded_data=encoded_data, repertoires=dataset.repertoires)

        return encoded_dataset

    def calculate_abundance_matrix(self, dataset: RepertoireDataset, sequence_presence_matrix, matrix_repertoire_ids,
                                   params: EncoderParams):
        relevant = np.isin(matrix_repertoire_ids, dataset.get_repertoire_ids())
        sequence_presence_matrix = sequence_presence_matrix[:, relevant]
        matrix_repertoire_ids = matrix_repertoire_ids[relevant]

        is_positive_class = AbundanceEncoderHelper.check_is_positive_class(dataset, matrix_repertoire_ids, params)

        relevant_sequence_indices, file_paths = AbundanceEncoderHelper.get_relevant_sequence_indices(sequence_presence_matrix, is_positive_class,
                                                                                                     self.p_value_threshold, self.relevant_indices_path, params)
        self._write_relevant_kmers_csv(relevant_sequence_indices, params.result_path)
        self._set_file_paths(file_paths)

        abundance_matrix = AbundanceEncoderHelper.build_abundance_matrix(sequence_presence_matrix, matrix_repertoire_ids, dataset.get_repertoire_ids(), relevant_sequence_indices)

        return abundance_matrix

    def _set_file_paths(self, file_paths):
        self.relevant_indices_path = file_paths["relevant_indices_path"]
        self.contingency_table_path = file_paths["contingency_table_path"]
        self.p_values_path = file_paths["p_values_path"]

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
        return [self.relevant_indices_path, self.relevant_sequence_path, self.contingency_table_path, self.p_values_path]

    @staticmethod
    def load_encoder(encoder_file: Path):
        encoder = DatasetEncoder.load_encoder(encoder_file)
        encoder.relevant_indices_path = DatasetEncoder.load_attribute(encoder, encoder_file, "relevant_indices_path")
        encoder.relevant_sequence_path = DatasetEncoder.load_attribute(encoder, encoder_file, "relevant_sequence_path")
        encoder.contingency_table_path = DatasetEncoder.load_attribute(encoder, encoder_file, "contingency_table_path")
        encoder.p_values_path = DatasetEncoder.load_attribute(encoder, encoder_file, "p_values_path")

        return encoder

    @staticmethod
    def get_documentation():
        doc = str(KmerAbundanceEncoder.__doc__)

        valid_field_values = str(Repertoire.FIELDS)[1:-1].replace("'", "`")
        mapping = {
            "Valid comparison value can be any repertoire field name.": f"Valid values are {valid_field_values}."
        }
        doc = update_docs_per_mapping(doc, mapping)
        return doc
