import pickle

from immuneML.IO.dataset_export.PickleExporter import PickleExporter
from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.pairwise_repertoire_comparison.ComparisonData import ComparisonData
from immuneML.util.PathBuilder import PathBuilder


class EncoderHelper:

    @staticmethod
    def prepare_training_ids(dataset: Dataset, params: EncoderParams):
        PathBuilder.build(params.result_path)
        if params.learn_model:
            training_ids = dataset.get_example_ids()
            training_ids_path = params.result_path / "training_ids.pickle"
            with training_ids_path.open("wb") as file:
                pickle.dump(training_ids, file)
        else:
            training_ids_path = params.result_path / "training_ids.pickle"
            with training_ids_path.open("rb") as file:
                training_ids = pickle.load(file)
        return training_ids

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
                                   sequence_batch_size, params.result_path)

        comp_data.process_dataset(dataset)

        return comp_data

    @staticmethod
    def store(encoded_dataset, params: EncoderParams):
        PickleExporter.export(encoded_dataset, params.result_path)
