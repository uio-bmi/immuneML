import numpy as np

from scripts.specification_util import update_docs_per_mapping
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.encoded_data.EncodedData import EncodedData
from source.data_model.repertoire.Repertoire import Repertoire
from source.encodings.DatasetEncoder import DatasetEncoder
from source.encodings.EncoderParams import EncoderParams
from source.encodings.filtered_sequence_encoding.SequenceFilterHelper import SequenceFilterHelper
from source.pairwise_repertoire_comparison.ComparisonData import ComparisonData
from source.util.EncoderHelper import EncoderHelper


class SequenceAbundanceEncoder(DatasetEncoder):
    """
    This encoder represents the repertoires as vectors where:
        - the first element corresponds to the number of label-associated clonotypes
        - the second element is the total number of unique clonotypes

    To determine what clonotypes (with features defined by comparison_attributes) are label-associated
    based on a statistical test. The statistical test used is Fisher's exact test (two-sided).

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

    Specification:

    .. indent with spaces
    .. code-block:: yaml

        my_sa_encoding:
            SequenceAbundance:
                comparison_attributes:
                    - sequence_aas
                    - v_genes
                    - j_genes
                    - chains
                    - region_types
                p_value_threshold: 0.05
                sequence_batch_size: 100000
    """

    RELEVANT_SEQUENCE_ABUNDANCE = "relevant_sequence_abundance"
    TOTAL_SEQUENCE_ABUNDANCE = "total_sequence_abundance"

    def __init__(self, comparison_attributes, p_value_threshold: float, sequence_batch_size: int, name: str = None):
        self.comparison_attributes = comparison_attributes
        self.sequence_batch_size = sequence_batch_size
        self.name = name
        self.relevant_sequence_indices = None
        self.context = None
        self.p_value_threshold = p_value_threshold

    @staticmethod
    def build_object(dataset, **params):
        assert isinstance(dataset, RepertoireDataset), "FilteredSequenceEncoder: this encoding only works on repertoire datasets."
        return SequenceAbundanceEncoder(**params)

    def encode(self, dataset, params: EncoderParams):
        return SequenceFilterHelper.encode(dataset, self.context, self.comparison_attributes, self.sequence_batch_size,
                                           params, self._encode_data)

    def _encode_data(self, dataset: RepertoireDataset, params: EncoderParams, comparison_data: ComparisonData):
        labels = params["label_configuration"].get_labels_by_name()

        assert len(labels) == 1, \
            "SequenceAbundanceEncoder: this encoding works only for single label."

        examples = self._calculate_sequence_abundance(dataset, comparison_data, labels[0], params)

        encoded_data = EncodedData(examples, dataset.get_metadata([labels[0]]), dataset.get_repertoire_ids(),
                                   [SequenceAbundanceEncoder.RELEVANT_SEQUENCE_ABUNDANCE,
                                    SequenceAbundanceEncoder.TOTAL_SEQUENCE_ABUNDANCE],
                                   encoding=SequenceAbundanceEncoder.__name__)

        encoded_dataset = RepertoireDataset(params=dataset.params, encoded_data=encoded_data, repertoires=dataset.repertoires)

        return encoded_dataset

    def _calculate_sequence_abundance(self, dataset: RepertoireDataset, comparison_data: ComparisonData, label: str, params: EncoderParams):

        sequence_p_values_indices = SequenceFilterHelper.get_relevant_sequences(dataset=dataset, params=params, comparison_data=comparison_data,
                                                                                label=label, p_value_threshold=self.p_value_threshold,
                                                                                comparison_attributes=self.comparison_attributes)

        abundance_matrix = self._build_abundance_matrix(comparison_data, dataset.get_repertoire_ids(), sequence_p_values_indices)

        return abundance_matrix

    def _build_abundance_matrix(self, comparison_data, repertoire_ids, sequence_p_values_indices):
        abundance_matrix = np.zeros((len(repertoire_ids), 2))

        for index, repertoire_id in enumerate(repertoire_ids):
            repertoire_vector = comparison_data.get_repertoire_vector(repertoire_id)
            relevant_sequence_abundance = np.sum(
                repertoire_vector[np.logical_and(sequence_p_values_indices, repertoire_vector)])
            total_sequence_abundance = np.sum(repertoire_vector)
            abundance_matrix[index] = [relevant_sequence_abundance, total_sequence_abundance]

        return abundance_matrix

    def set_context(self, context: dict):
        self.context = context
        return self

    def store(self, encoded_dataset, params: EncoderParams):
        EncoderHelper.store(encoded_dataset, params)

    @staticmethod
    def get_documentation():
        doc = str(SequenceAbundanceEncoder.__doc__)

        valid_field_values = str(Repertoire.FIELDS)[1:-1].replace("'", "`")
        mapping = {
            "Valid comparison value can be any repertoire field name.": f"Valid values are {valid_field_values}."
        }
        doc = update_docs_per_mapping(doc, mapping)
        return doc
