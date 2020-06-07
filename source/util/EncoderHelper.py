import pickle

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.encodings.EncoderParams import EncoderParams
from source.pairwise_repertoire_comparison.ComparisonData import ComparisonData
from source.util.PathBuilder import PathBuilder


class EncoderHelper:

    @staticmethod
    def prepare_training_ids(dataset: RepertoireDataset, params: EncoderParams):
        PathBuilder.build(params["result_path"])
        if params["learn_model"]:
            train_repertoire_ids = dataset.get_repertoire_ids()
            with open(params["result_path"] + "repertoire_ids.pickle", "wb") as file:
                pickle.dump(train_repertoire_ids, file)
        else:
            with open(params["result_path"] + "repertoire_ids.pickle", "rb") as file:
                train_repertoire_ids = pickle.load(file)
        return train_repertoire_ids

    @staticmethod
    def get_current_dataset(dataset, context):
        return dataset if context is None or "dataset" not in context else context["dataset"]

    @staticmethod
    def build_comparison_params(dataset, comparison_attributes) -> tuple:
        return (("dataset_identifier", dataset.identifier),
                ("comparison_attributes", tuple(comparison_attributes)),
                ("repertoire_ids", tuple(dataset.get_repertoire_ids())))

    @staticmethod
    def build_comparison_data(dataset: RepertoireDataset, params: EncoderParams,
                              comparison_attributes, sequence_batch_size):

        comp_data = ComparisonData(dataset.get_repertoire_ids(), comparison_attributes,
                                   sequence_batch_size, params["result_path"])

        comp_data.process_dataset(dataset)

        return comp_data

    @staticmethod
    def store(encoded_dataset, params: EncoderParams):
        PickleExporter.export(encoded_dataset, params["result_path"], params["filename"])
