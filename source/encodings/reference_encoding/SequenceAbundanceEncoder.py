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
    This encoder represents the repertoires by a vector where:
        - the first element corresponds to the number / percentage of label-associated clonotypes / reads
        - the second element is the total number / percentage of unique clonotypes / reads

    To determine what clonotypes (with features defined by comparison_attributes) are label-associated,
    it performs the supplied statistical test using the given p_value cut-off.

    Reference: Emerson et al 2017
    """

    @staticmethod
    def create_encoder(dataset, params: dict = None):
        assert isinstance(dataset, RepertoireDataset), "SequenceAbundanceEncoder: this encoding only works on repertoire datasets."
        return SequenceAbundanceEncoder(**params)

    def __init__(self, comparison_attributes, p_value_threshold, pool_size: int):
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
