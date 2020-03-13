import pickle

from source.caching.CacheHandler import CacheHandler
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.encoded_data.EncodedData import EncodedData
from source.encodings.DatasetEncoder import DatasetEncoder
from source.encodings.EncoderParams import EncoderParams
from source.pairwise_repertoire_comparison.ComparisonData import ComparisonData
from source.util.PathBuilder import PathBuilder


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
        pool_size (int): The pool size used for parallelization. This does not affect the results of the encoding,
            only the speed.

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
                    pool_size: 4
    """

    @staticmethod
    def build_object(dataset, **params):
        assert isinstance(dataset, RepertoireDataset), "SequenceAbundanceEncoder: this encoding only works on repertoire datasets."
        return SequenceAbundanceEncoder(**params)

    def __init__(self, comparison_attributes, p_value_threshold: float, pool_size: int):
        self.comparison_attributes = comparison_attributes
        self.p_value_threshold = p_value_threshold
        self.pool_size = pool_size
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

    def prepare_encoding_params(self, dataset: RepertoireDataset, params: EncoderParams):
        PathBuilder.build(params["result_path"])
        if params["learn_model"]:
            train_repertoire_ids = dataset.get_repertoire_ids()
            with open(params["result_path"] + "repertoire_ids.pickle", "wb") as file:
                pickle.dump(train_repertoire_ids, file)
        else:
            with open(params["result_path"] + "repertoire_ids.pickle", "rb") as file:
                train_repertoire_ids = pickle.load(file)
        return train_repertoire_ids

    def _calculate_sequence_abundance(self, dataset: RepertoireDataset, params: EncoderParams, comparison_data: ComparisonData):

        labels = params["label_configuration"].get_labels_by_name()

        assert len(labels) == 1, \
            "SequenceAbundanceEncoder: this encoding works only for single label."

        train_repertoire_ids = self.prepare_encoding_params(dataset, params)
        examples = comparison_data.calculate_sequence_abundance(dataset, labels[0],
                                                                params["label_configuration"].get_label_values(labels[0]),
                                                                self.p_value_threshold)

        encoded_dataset = RepertoireDataset(params=dataset.params, encoded_data=EncodedData(examples, dataset.get_metadata([labels[0]]),
                                                                                            train_repertoire_ids,
                                                                                            ["relevant_sequence_abundance", "total_sequence_abundance"],
                                                                                            encoding=SequenceAbundanceEncoder.__name__),
                                            repertoires=dataset.repertoires)

        return encoded_dataset

    def build_comparison_data(self, dataset: RepertoireDataset, params: EncoderParams):

        comp_data = ComparisonData(dataset.get_repertoire_ids(), self.comparison_attributes,
                                   self.pool_size, params["batch_size"], params["result_path"])

        comp_data.process_dataset(dataset)

        return comp_data

    def store(self, encoded_dataset, params: EncoderParams):
        pass
