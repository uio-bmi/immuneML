from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from immuneML.IO.ml_method.UtilIO import UtilIO
from immuneML.caching.CacheHandler import CacheHandler
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.abundance_encoding.AbundanceEncoderHelper import AbundanceEncoderHelper
from immuneML.pairwise_repertoire_comparison.ComparisonData import ComparisonData
from immuneML.util.EncoderHelper import EncoderHelper
from scripts.specification_util import update_docs_per_mapping


class SequenceAbundanceEncoder(DatasetEncoder):
    """
    This encoder represents the repertoires as vectors where:

    - the first element corresponds to the number of label-associated clonotypes
    - the second element is the total number of unique clonotypes

    To determine what clonotypes (with features defined by comparison_attributes) are label-associated, one-sided Fisher's exact test is used.

    The encoder also writes out files containing the contingency table used for Fisher's exact test,
    the resulting p-values, and the significantly abundant sequences
    (use :py:obj:`~immuneML.reports.encoding_reports.RelevantSequenceExporter.RelevantSequenceExporter` to export these sequences in AIRR format).

    Reference: Emerson, Ryan O. et al.
    ‘Immunosequencing Identifies Signatures of Cytomegalovirus Exposure History and HLA-Mediated Effects on the T Cell Repertoire’.
    Nature Genetics 49, no. 5 (May 2017): 659–65. `doi.org/10.1038/ng.3822 <https://doi.org/10.1038/ng.3822>`_.

    Note: to use this encoder, it is necessary to explicitly define the positive class for the label when defining the label
    in the instruction. With positive class defined, it can then be determined which sequences are indicative of the positive class.
    For full example of using this encoder, see :ref:`Reproduction of the CMV status predictions study`.

    **Specification arguments:**

    - comparison_attributes (list): The attributes to be considered to group receptors into clonotypes. Only the fields specified in
      comparison_attributes will be considered, all other fields are ignored. Valid comparison value can be any repertoire field name.

    - p_value_threshold (float): The p value threshold to be used by the statistical test.

    - sequence_batch_size (int): The number of sequences in a batch when comparing sequences across repertoires, typically 100s of thousands.
      This does not affect the results of the encoding, only the speed. The default value is 1.000.000

    - repertoire_batch_size (int): How many repertoires will be loaded at once. This does not affect the result of the encoding, only the speed.
      This value is a trade-off between the number of repertoires that can fit the RAM at the time and loading time from disk.


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            encodings:
                my_sa_encoding:
                    SequenceAbundance:
                        comparison_attributes:
                            - sequence_aa
                            - v_call
                            - j_call
                            - chain
                            - region_type
                        p_value_threshold: 0.05
                        sequence_batch_size: 100000
                        repertoire_batch_size: 32

    """

    RELEVANT_SEQUENCE_ABUNDANCE = "relevant_sequence_abundance"
    TOTAL_SEQUENCE_ABUNDANCE = "total_sequence_abundance"

    def __init__(self, comparison_attributes, p_value_threshold: float, sequence_batch_size: int, repertoire_batch_size: int, name: str = None):
        super().__init__(name=name)
        self.comparison_attributes = comparison_attributes
        self.sequence_batch_size = sequence_batch_size
        self.relevant_sequence_indices = None
        self.context = None
        self.p_value_threshold = p_value_threshold
        self.relevant_indices_path = None
        self.relevant_sequence_path = None
        self.contingency_table_path = None
        self.p_values_path = None
        self.comparison_data = None
        self.repertoire_batch_size = repertoire_batch_size

    @staticmethod
    def build_object(dataset, **params):
        assert isinstance(dataset, RepertoireDataset), "SequenceAbundanceEncoder: this encoding only works on repertoire datasets."
        return SequenceAbundanceEncoder(**params)

    def encode(self, dataset, params: EncoderParams):
        EncoderHelper.check_positive_class_labels(params.label_config, SequenceAbundanceEncoder.__name__)

        self.comparison_data = self._build_comparison_data(dataset, params)
        return self._encode_data(dataset, params)

    def _build_comparison_data(self, dataset: RepertoireDataset, params: EncoderParams):

        current_dataset = EncoderHelper.get_current_dataset(dataset, self.context)
        comparison_data = CacheHandler.memo_by_params(
            EncoderHelper.build_comparison_params(current_dataset, self.comparison_attributes),
            lambda: EncoderHelper.build_comparison_data(current_dataset, params,
                                                        self.comparison_attributes,
                                                        self.sequence_batch_size))

        return comparison_data

    def _encode_data(self, dataset: RepertoireDataset, params: EncoderParams):

        label_name = params.label_config.get_labels_by_name()[0]

        examples = self._calculate_abundance_matrix(dataset, self.comparison_data, params)

        encoded_data = EncodedData(examples, dataset.get_metadata([label_name]) if params.encode_labels else None, dataset.get_repertoire_ids(),
                                   [SequenceAbundanceEncoder.RELEVANT_SEQUENCE_ABUNDANCE, SequenceAbundanceEncoder.TOTAL_SEQUENCE_ABUNDANCE],
                                   example_weights=dataset.get_example_weights(),
                                   encoding=SequenceAbundanceEncoder.__name__, info={'relevant_sequence_path': self.relevant_sequence_path,
                                                                                     "contingency_table_path": self.contingency_table_path,
                                                                                     "p_values_path": self.p_values_path})

        encoded_dataset = RepertoireDataset(labels=dataset.labels, encoded_data=encoded_data, repertoires=dataset.repertoires)

        return encoded_dataset

    def _calculate_abundance_matrix(self, dataset: RepertoireDataset, comparison_data: ComparisonData, params: EncoderParams):
        comparison_data.set_iteration_repertoire_ids(dataset.get_repertoire_ids())
        is_positive_class = AbundanceEncoderHelper.check_is_positive_class(dataset, dataset.get_repertoire_ids(), params.label_config)

        relevant_sequence_indices, file_paths = AbundanceEncoderHelper.get_relevant_sequence_indices(comparison_data, is_positive_class,
                                                                                                     self.p_value_threshold,
                                                                                                     self.relevant_indices_path, params,
                                                                                                     cache_params=(dataset.get_repertoire_ids(),
                                                                                                                   self.comparison_attributes))

        self._write_relevant_sequences_csv(comparison_data, relevant_sequence_indices, params.result_path)
        self._set_file_paths(file_paths)

        abundance_matrix = self._build_abundance_matrix(comparison_data, dataset.get_repertoire_ids(), relevant_sequence_indices)

        return abundance_matrix

    def _write_relevant_sequences_csv(self, comparison_data, relevant_sequence_indices, result_path):
        if self.relevant_sequence_path is None:
            self.relevant_sequence_path = result_path / 'relevant_sequences.csv'

        all_sequences = comparison_data.get_item_names()
        relevant_sequences = all_sequences[relevant_sequence_indices]
        df = pd.DataFrame(relevant_sequences, columns=self.comparison_attributes)
        sequence_csv_path = result_path / 'relevant_sequences.csv'
        df.to_csv(sequence_csv_path, sep=',', index=False)

    def _set_file_paths(self, file_paths):
        self.relevant_indices_path = file_paths["relevant_indices_path"]
        self.contingency_table_path = file_paths["contingency_table_path"] if "contingency_table_path" in file_paths else None
        self.p_values_path = file_paths["p_values_path"] if "p_values_path" in file_paths else None

    def _build_abundance_matrix(self, comparison_data, repertoire_ids, relevant_sequence_indices):
        abundance_matrix = np.zeros((len(repertoire_ids), 2))

        for index in range(0, len(repertoire_ids) + self.repertoire_batch_size, self.repertoire_batch_size):
            ind_start, ind_end = index, min(index + self.repertoire_batch_size, len(repertoire_ids))
            repertoire_vectors = comparison_data.get_repertoire_vectors(repertoire_ids[ind_start:ind_end])

            for rep_index in range(ind_start, ind_end):
                repertoire_vector = repertoire_vectors[repertoire_ids[rep_index]]
                relevant_sequence_abundance = np.sum(
                    repertoire_vector[np.logical_and(relevant_sequence_indices, repertoire_vector)])
                total_sequence_abundance = np.sum(repertoire_vector)
                abundance_matrix[rep_index] = [relevant_sequence_abundance, total_sequence_abundance]

        return abundance_matrix

    def set_context(self, context: dict):
        self.context = context
        return self

    def get_additional_files(self) -> List[Path]:
        return [self.relevant_indices_path]

    @staticmethod
    def load_encoder(encoder_file: Path):
        encoder = DatasetEncoder.load_encoder(encoder_file)
        encoder.relevant_indices_path = DatasetEncoder.load_attribute(encoder, encoder_file, "relevant_indices_path")
        encoder.comparison_data = UtilIO.import_comparison_data(encoder_file.parent)
        return encoder

    @staticmethod
    def get_documentation():
        doc = str(SequenceAbundanceEncoder.__doc__)

        valid_field_values = str(Repertoire.FIELDS)[1:-1].replace("'", "`")
        mapping = {
            "Valid comparison value can be any repertoire field name.": f"Valid values are {valid_field_values}."
        }
        doc = update_docs_per_mapping(doc, mapping)
        return doc
