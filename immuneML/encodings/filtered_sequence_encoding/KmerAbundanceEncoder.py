import copy
import math
import pickle
import subprocess
from multiprocessing.pool import Pool
from pathlib import Path
from typing import List

import fisher
import numpy as np
import pandas as pd

from immuneML.analysis.data_manipulation.NormalizationType import NormalizationType
from immuneML.caching.CacheHandler import CacheHandler
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.filtered_sequence_encoding.SequenceFilterHelper import SequenceFilterHelper
from immuneML.encodings.kmer_frequency.KmerFreqRepertoireEncoder import KmerFreqRepertoireEncoder
from immuneML.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder
from immuneML.encodings.kmer_frequency.ReadsType import ReadsType
from immuneML.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType import SequenceEncodingType
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.util.CompAIRRHelper import CompAIRRHelper
from immuneML.util.CompAIRRParams import CompAIRRParams
from immuneML.util.EncoderHelper import EncoderHelper
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder
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
    OUTPUT_FILENAME = "compairr_out.tsv"
    LOG_FILENAME = "compairr_log.txt"

    def __init__(self, p_value_threshold: float, name: str = None):
        self.name = name
        self.p_value_threshold = p_value_threshold

        self.relevant_indices_path = None
        self.relevant_sequence_csv_path = None
        self.repertoires_filepath = None
        self.sequences_filepaths = None
        self.context = None

        self.full_kmer_set = None
        self.raw_distance_matrix_np = None

        # todo get parameters for this

        self.kmer_frequency_encoder = KmerFreqRepertoireEncoder(normalization_type=NormalizationType.BINARY,
                                                           reads=ReadsType.UNIQUE,
                                                           sequence_encoding=SequenceEncodingType.CONTINUOUS_KMER,
                                                           k=3,
                                                           scale_to_unit_variance=False,
                                                           scale_to_zero_mean=False,
                                                           sequence_type=SequenceType.AMINO_ACID)

    @staticmethod
    def _prepare_parameters(p_value_threshold: float, name: str = None):
        ParameterValidator.assert_type_and_value(p_value_threshold, float, "KmerAbundanceEncoder", "p_value_threshold", min_inclusive=0,
                                                 max_inclusive=1)

        return {
            "p_value_threshold": p_value_threshold,
            "name": name
        }

    @staticmethod
    def build_object(dataset, **params):
        assert isinstance(dataset, RepertoireDataset), "KmerAbundanceEncoder: this encoding only works on repertoire datasets."
        prepared_params = KmerAbundanceEncoder._prepare_parameters(**params)
        return KmerAbundanceEncoder(**prepared_params)

    def encode(self, dataset, params: EncoderParams):
        EncoderHelper.check_positive_class_label(KmerAbundanceEncoder.__name__,
                                                 params.label_config.get_label_objects())
        self._prepare_kmer_presence_data(dataset, params)
        encoded_dataset = self._encode_data(dataset, params)

        return encoded_dataset


    def _prepare_kmer_presence_data(self, dataset, params):
        full_dataset = EncoderHelper.get_current_dataset(dataset, self.context)

        # kmer_freq_params = EncoderParams(result_path=params.result_path / "kmer_frequency",
        #                                  label_config=params.label_config,
        #                                  filename: str = "",
        #                                  pool_size: int = 4
        #                                 model: dict = None
        #                                 learn_model: bool = True
        #                                 encode_labels: bool = True
        #                             )

        # todo check /copy params

        kmer_encoded_data = self.kmer_frequency_encoder.encode(full_dataset, params)
        self.full_kmer_set = np.array(kmer_encoded_data.encoded_data.feature_names)
        self.kmer_presence_matrix = kmer_encoded_data.encoded_data.examples.toarray().T
        self.matrix_repertoire_ids = np.array(kmer_encoded_data.encoded_data.example_ids)

        # full_kmer_set = self._get_full_kmer_set(full_dataset)
        # kmer_presence_matrix, matrix_repertoire_ids = self._get_kmer_presence(full_dataset, full_kmer_set, params)

        # self.full_kmer_set = full_kmer_set
        # self.kmer_presence_matrix = kmer_presence_matrix
        # self.matrix_repertoire_ids = matrix_repertoire_ids


    def _build_sequence_presence_params(self, dataset, compairr_params):
        return (("dataset_identifier", dataset.identifier),
                ("repertoire_ids", tuple(dataset.get_repertoire_ids()))) # todo update

    def _build_dataset_params(self, dataset):
        return (("dataset_identifier", dataset.identifier),
                ("repertoire_ids", tuple(dataset.get_repertoire_ids())),
                "sequence_attributes", tuple(self.get_relevant_sequence_attributes()))


    def _encode_data(self, dataset: RepertoireDataset, params: EncoderParams):
        label = params.label_config.get_labels_by_name()[0]

        examples = self._calculate_sequence_abundance(dataset, self.kmer_presence_matrix, self.matrix_repertoire_ids, label, params)

        encoded_data = EncodedData(examples, dataset.get_metadata([label]) if params.encode_labels else None, dataset.get_repertoire_ids(),
                                   [KmerAbundanceEncoder.RELEVANT_SEQUENCE_ABUNDANCE,
                                    KmerAbundanceEncoder.TOTAL_SEQUENCE_ABUNDANCE],
                                   encoding=KmerAbundanceEncoder.__name__,
                                   info={'relevant_sequence_path': self.relevant_sequence_csv_path})

        encoded_dataset = RepertoireDataset(labels=dataset.labels, encoded_data=encoded_data, repertoires=dataset.repertoires)

        return encoded_dataset

    def _calculate_sequence_abundance(self, dataset: RepertoireDataset, sequence_presence_matrix, matrix_repertoire_ids, label_str: str,
                                      params: EncoderParams):
        sequence_p_values = self._find_label_associated_sequence_p_values(sequence_presence_matrix, matrix_repertoire_ids, dataset, params, label_str)
        relevant_sequence_indices = self._get_relevant_sequence_indices(params, label_str, sequence_p_values)
        abundance_matrix = self._build_abundance_matrix(sequence_presence_matrix, matrix_repertoire_ids, dataset.get_repertoire_ids(),
                                                        relevant_sequence_indices)

        return abundance_matrix

    def _prepare_compairr_input_files(self, dataset, full_sequence_set, result_path):
        PathBuilder.build(result_path)

        if self.repertoires_filepath is None:
            self.repertoires_filepath = result_path / "compairr_repertoires.tsv"

        CompAIRRHelper.write_repertoire_file(dataset, self.repertoires_filepath, self.compairr_params)

        if self.sequences_filepaths is None:
            self.sequences_filepaths = []

            n_sequence_files = math.ceil(len(full_sequence_set) / self.sequence_batch_size)
            subset_start_index = 0
            subset_end_index = min(self.sequence_batch_size, len(full_sequence_set))

            for file_index in range(n_sequence_files):
                filename = result_path / f"compairr_sequences_batch{file_index}.tsv"
                self.sequences_filepaths.append(filename)
                sequence_subset = full_sequence_set[subset_start_index:subset_end_index]

                self.write_sequence_set_file(sequence_subset, filename, offset=subset_start_index)

                subset_start_index += self.sequence_batch_size
                subset_end_index = min(subset_end_index + self.sequence_batch_size, len(full_sequence_set))

    def _get_relevant_sequence_indices(self, params, label_str, sequence_p_values):
        if self.relevant_indices_path is None:
            self.relevant_indices_path = params.result_path / 'relevant_sequence_indices.pickle'

        if params.learn_model:
            SequenceFilterHelper._check_label_object(params, label_str)

            relevant_sequence_indices = np.array(sequence_p_values) < self.p_value_threshold

            with self.relevant_indices_path.open("wb") as file:
                pickle.dump(relevant_sequence_indices, file)

            self._write_relevant_kmer_csv(self.full_kmer_set[relevant_sequence_indices], params.result_path)

        else:
            with self.relevant_indices_path.open("rb") as file:
                relevant_sequence_indices = pickle.load(file)

        return relevant_sequence_indices

    def _write_relevant_kmer_csv(self, relevant_sequences, result_path):
        if self.relevant_sequence_csv_path is None:
            self.relevant_sequence_csv_path = result_path / 'relevant_sequences.csv'

        df = pd.DataFrame(relevant_sequences,
                          columns=["k-mer"])

        df.to_csv(self.relevant_sequence_csv_path, sep=",", index=False)

    def _find_label_associated_sequence_p_values(self, kmer_presence_matrix, matrix_repertoire_ids, dataset, params, label_str):
        relevant = np.isin(matrix_repertoire_ids, dataset.get_repertoire_ids())
        kmer_presence_matrix = kmer_presence_matrix[:, relevant]
        matrix_repertoire_ids = matrix_repertoire_ids[relevant]

        is_first_class = self._is_first_class(dataset, matrix_repertoire_ids, params, label_str)

        return self._find_sequence_p_values_with_fisher(kmer_presence_matrix, is_first_class)

    def _find_sequence_p_values_with_fisher(self, sequence_presence_matrix, is_first_class):
        sequence_p_values = []

        for sequence_vector in sequence_presence_matrix:
            if sequence_vector.sum() > 1:

                first_class_present = np.sum(sequence_vector[np.logical_and(sequence_vector, is_first_class)])
                second_class_present = np.sum(
                    sequence_vector[np.logical_and(sequence_vector, np.logical_not(is_first_class))])
                first_class_absent = np.sum(np.logical_and(is_first_class, sequence_vector == 0))
                second_class_absent = np.sum(np.logical_and(np.logical_not(is_first_class), sequence_vector == 0))

                sequence_p_values.append(fisher.pvalue(first_class_present, second_class_present, first_class_absent,
                                                       second_class_absent).right_tail)
            else:
                sequence_p_values.append(SequenceFilterHelper.INVALID_P_VALUE)

        return sequence_p_values

    def _is_first_class(self, dataset, matrix_repertoire_ids, params, label_str):
        label = params.label_config.get_label_object(label_str)

        is_first_class = np.array(
            [dataset.get_repertoire(repertoire_identifier=repertoire_id).metadata[label.name] for repertoire_id in
             matrix_repertoire_ids]) == label.positive_class

        return is_first_class

    def _build_abundance_matrix(self, sequence_presence_matrix, matrix_repertoire_ids, dataset_repertoire_ids, sequence_p_values_indices):
        abundance_matrix = np.zeros((len(dataset_repertoire_ids), 2))

        for idx_in_dataset, dataset_repertoire_id in enumerate(dataset_repertoire_ids):
            relevant_row = np.where(matrix_repertoire_ids == dataset_repertoire_id)
            repertoire_vector = sequence_presence_matrix.T[relevant_row]
            relevant_sequence_abundance = np.sum(repertoire_vector[np.logical_and(sequence_p_values_indices, repertoire_vector)])
            total_sequence_abundance = np.sum(repertoire_vector)
            abundance_matrix[idx_in_dataset] = [relevant_sequence_abundance, total_sequence_abundance]

        return abundance_matrix

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
        return [self.relevant_indices_path, self.relevant_sequence_csv_path]

    @staticmethod
    def load_encoder(encoder_file: Path):
        encoder = DatasetEncoder.load_encoder(encoder_file)
        encoder.relevant_indices_path = DatasetEncoder.load_attribute(encoder, encoder_file, "relevant_indices_path")
        encoder.relevant_sequence_csv_path = DatasetEncoder.load_attribute(encoder, encoder_file, "relevant_sequence_csv_path")
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
