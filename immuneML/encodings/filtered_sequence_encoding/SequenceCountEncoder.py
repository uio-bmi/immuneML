from pathlib import Path
from typing import List

import numpy as np

from immuneML.IO.ml_method.UtilIO import UtilIO
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.filtered_sequence_encoding.SequenceFilterHelper import SequenceFilterHelper
from immuneML.pairwise_repertoire_comparison.ComparisonData import ComparisonData
from immuneML.util.EncoderHelper import EncoderHelper
from scripts.specification_util import update_docs_per_mapping


class SequenceCountEncoder(DatasetEncoder):
    """
    This encoder represents the repertoires as a matrix of sequence counts for label-associated sequences.
    To determine what clonotypes (with features defined by comparison_attributes) are label-associated
    based on a statistical test. The statistical test used is Fisher's exact test (one-sided).

    Reference: Emerson, Ryan O. et al.
    ‘Immunosequencing Identifies Signatures of Cytomegalovirus Exposure History and HLA-Mediated Effects on the T Cell Repertoire’.
    Nature Genetics 49, no. 5 (May 2017): 659–65. `doi.org/10.1038/ng.3822 <https://doi.org/10.1038/ng.3822>`_.

    Arguments:

        comparison_attributes (list): The attributes to be considered to group receptors into clonotypes.
            Only the fields specified in comparison_attributes will be considered, all other fields are ignored.
            Valid comparison value can be any repertoire field name.

        p_value_threshold (float): The p value threshold to be used by the statistical test.

        sequence_batch_size (int): The pool size used for parallelization. This does not affect the results of the encoding,
            only the speed.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_seq_count_encoding: # user-defined encoding name
            SequenceCount: # name of the encoding followed by parameters
                comparison_attributes:
                    - sequence_aas
                    - v_genes
                    - j_genes
                    - chains
                    - region_types
                p_value_threshold: 0.05
                sequence_batch_size: 100000

    """

    def __init__(self, comparison_attributes, p_value_threshold: float, sequence_batch_size: int, name: str = None):
        self.comparison_attributes = comparison_attributes
        self.sequence_batch_size = sequence_batch_size
        self.name = name
        self.relevant_sequence_indices = None
        self.context = None
        self.p_value_threshold = p_value_threshold
        self.relevant_indices_path = None
        self.relevant_sequence_csv_path = None
        self.comparison_data = None

    @staticmethod
    def build_object(dataset, **params):
        assert isinstance(dataset, RepertoireDataset), "FilteredSequenceEncoder: this encoding only works on repertoire datasets."
        return SequenceCountEncoder(**params)

    def encode(self, dataset, params: EncoderParams):
        self.comparison_data = SequenceFilterHelper.build_comparison_data(dataset, self.context, self.comparison_attributes, params,
                                                                          self.sequence_batch_size)
        return self._encode_data(dataset, params)

    def _encode_data(self, dataset: RepertoireDataset, params: EncoderParams):
        labels = params.label_config.get_labels_by_name()

        assert len(labels) == 1, f"SequenceCountEncoder: this encoding works only for single label, got {labels} instead."

        encoded_data = self._encode_sequence_count(dataset, self.comparison_data, labels[0], params)

        encoded_dataset = RepertoireDataset(labels=dataset.labels, encoded_data=encoded_data, repertoires=dataset.repertoires)

        return encoded_dataset

    def _encode_sequence_count(self, dataset: RepertoireDataset, comparison_data: ComparisonData, label: str, params: EncoderParams) -> EncodedData:
        sequence_p_values_indices, indices_path, relevant_sequences_path = SequenceFilterHelper.get_relevant_sequences(dataset, params,
                                                                                                                       comparison_data, label,
                                                                                                                       self.p_value_threshold,
                                                                                                                       self.comparison_attributes,
                                                                                                                       self.relevant_indices_path)
        if self.relevant_indices_path is None:
            self.relevant_indices_path = indices_path
        if self.relevant_sequence_csv_path is None:
            self.relevant_sequence_csv_path = relevant_sequences_path

        count_matrix = self._build_count_matrix(comparison_data, dataset.get_repertoire_ids(), sequence_p_values_indices)
        feature_names = comparison_data.get_item_names()[sequence_p_values_indices]

        encoded_data = EncodedData(count_matrix, dataset.get_metadata([label]) if params.encode_labels else None,
                                   dataset.get_repertoire_ids(),
                                   feature_names,
                                   encoding=SequenceCountEncoder.__name__, info={'relevant_sequence_path': self.relevant_sequence_csv_path})

        return encoded_data

    def _build_count_matrix(self, comparison_data, repertoire_ids, sequence_p_values_indices):
        count_matrix = np.zeros((len(repertoire_ids), np.sum(sequence_p_values_indices)))

        for index, repertoire_id in enumerate(repertoire_ids):
            repertoire_vector = comparison_data.get_repertoire_vector(repertoire_id)
            relevant_sequences = repertoire_vector[sequence_p_values_indices]
            count_matrix[index] = relevant_sequences

        return count_matrix

    def set_context(self, context: dict):
        self.context = context
        return self

    def store(self, encoded_dataset, params: EncoderParams):
        EncoderHelper.store(encoded_dataset, params)

    @staticmethod
    def export_encoder(path: Path, encoder) -> Path:
        encoder_file = DatasetEncoder.store_encoder(encoder, path / "encoder.pickle")
        UtilIO.export_comparison_data(encoder.comparison_data, path)
        return encoder_file

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
        doc = str(SequenceCountEncoder.__doc__)

        valid_field_values = str(Repertoire.FIELDS)[1:-1].replace("'", "`")
        mapping = {
            "Valid comparison value can be any repertoire field name.": f"Valid values are {valid_field_values}."
        }
        doc = update_docs_per_mapping(doc, mapping)
        return doc
