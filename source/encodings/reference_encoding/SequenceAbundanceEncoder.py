import pickle

import pandas as pd

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.caching.CacheHandler import CacheHandler
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.encoded_data.EncodedData import EncodedData
from source.encodings.DatasetEncoder import DatasetEncoder
from source.encodings.EncoderParams import EncoderParams
from source.pairwise_repertoire_comparison.ComparisonData import ComparisonData


class SequenceAbundanceEncoder(DatasetEncoder):
    """
    This encoder represents the repertoires as a vector where:
        - the first element corresponds to the number of label-associated clonotypes
        - the second element is the total number of unique clonotypes

    To determine what clonotypes (with features defined by comparison_attributes) are label-associated
    based on a statistical test. The statistical test used is Fisher's exact test (two-sided).

    Reference: Emerson, Ryan O. et al.
    ‘Immunosequencing Identifies Signatures of Cytomegalovirus Exposure History and HLA-Mediated Effects on the T Cell Repertoire’.
    Nature Genetics 49, no. 5 (May 2017): 659–65. https://doi.org/10.1038/ng.3822.

    Arguments:
        comparison_attributes (list): The attributes to be considered to group receptors into clonotypes.
            Only the fields specified in comparison_attributes will be considered, all other fields are ignored.
        p_value_threshold (float): The p value threshold to be used by the statistical test.

    Specification:

        encodings:
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

    @staticmethod
    def build_object(dataset, **params):
        assert isinstance(dataset, RepertoireDataset), "SequenceAbundanceEncoder: this encoding only works on repertoire datasets."
        return SequenceAbundanceEncoder(**params)

    def __init__(self, comparison_attributes, p_value_threshold: float, sequence_batch_size: int, name: str = None):
        self.comparison_attributes = comparison_attributes
        self.p_value_threshold = p_value_threshold
        self.sequence_batch_size = sequence_batch_size
        self.name = name
        self.relevant_sequence_indices = None
        self.context = None

    def build_comparison_params(self, dataset) -> tuple:
        return (("dataset_identifier", dataset.identifier),
                ("comparison_attributes", tuple(self.comparison_attributes)))

    def encode(self, dataset, params: EncoderParams):
        current_dataset = dataset if self.context is None or "dataset" not in self.context else self.context["dataset"]

        comparison_data = CacheHandler.memo_by_params(self.build_comparison_params(current_dataset),
                                                      lambda: self.build_comparison_data(current_dataset, params))

        encoded_dataset = self._calculate_sequence_abundance(dataset, params, comparison_data)

        return encoded_dataset

    def set_context(self, context: dict):
        self.context = context
        return self

    def _calculate_sequence_abundance(self, dataset: RepertoireDataset, params: EncoderParams, comparison_data: ComparisonData):

        labels = params["label_configuration"].get_labels_by_name()

        assert len(labels) == 1, "SequenceAbundanceEncoder: this encoding works only for single label."

        self._prepare_relevant_sequence_indices(dataset, params, comparison_data, labels)

        examples = comparison_data.build_abundance_matrix(dataset.get_example_ids(), self.relevant_sequence_indices)
        feature_names = [SequenceAbundanceEncoder.RELEVANT_SEQUENCE_ABUNDANCE, SequenceAbundanceEncoder.TOTAL_SEQUENCE_ABUNDANCE]

        encoded_dataset = RepertoireDataset(params=dataset.params, repertoires=dataset.repertoires,
                                            encoded_data=EncodedData(examples, dataset.get_metadata([labels[0]]), dataset.get_example_ids(),
                                                                     feature_names, encoding=SequenceAbundanceEncoder.__name__))

        return encoded_dataset

    def _prepare_relevant_sequence_indices(self, dataset: RepertoireDataset, params: EncoderParams, comparison_data: ComparisonData,
                                           labels: list):
        if params["learn_model"]:
            label_values = params["label_configuration"].get_label_values(labels[0])
            relevant_sequence_indices = comparison_data.get_relevant_sequence_indices(dataset, labels[0], label_values, self.p_value_threshold)
            self.relevant_sequence_indices = relevant_sequence_indices
            with open(f'{params["result_path"]}relevant_sequence_indices.pickle', "wb") as file:
                pickle.dump(relevant_sequence_indices, file)

            all_sequences = comparison_data.get_item_names()
            relevant_sequences = all_sequences[relevant_sequence_indices]
            df = pd.DataFrame(relevant_sequences, columns=self.comparison_attributes)
            df.to_csv(f'{params["result_path"]}relevant_sequences.csv', sep=',', index=False)
        else:
            with open(f'{params["result_path"]}relevant_sequence_indices.pickle', "rb") as file:
                self.relevant_sequence_indices = pickle.load(file)

    def build_comparison_data(self, dataset: RepertoireDataset, params: EncoderParams):

        comp_data = ComparisonData(dataset.get_repertoire_ids(), self.comparison_attributes,
                                   self.sequence_batch_size, params["result_path"])

        comp_data.process_dataset(dataset)

        return comp_data

    def store(self, encoded_dataset, params: EncoderParams):
        PickleExporter.export(encoded_dataset, params["result_path"])
