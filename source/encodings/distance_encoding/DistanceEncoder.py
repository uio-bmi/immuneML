import copy

import pandas as pd

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.encoded_data.EncodedData import EncodedData
from source.encodings.DatasetEncoder import DatasetEncoder
from source.encodings.EncoderParams import EncoderParams
from source.encodings.distance_encoding.DistanceMetricType import DistanceMetricType
from source.pairwise_repertoire_comparison.PairwiseRepertoireComparison import PairwiseRepertoireComparison
from source.util import DistanceMetrics
from source.util.ReflectionHandler import ReflectionHandler


class DistanceEncoder(DatasetEncoder):

    def __init__(self, distance_metric: DistanceMetricType, attributes_to_match: list, context: dict = None):
        self.distance_fn = ReflectionHandler.import_function(distance_metric.value, DistanceMetrics)
        self.attributes_to_match = attributes_to_match
        self.context = context

    def set_context(self, context: dict):
        self.context = context
        return self

    @staticmethod
    def create_encoder(dataset, params: dict = None):
        if isinstance(dataset, RepertoireDataset):
            return DistanceEncoder(**params)
        else:
            raise ValueError("DistanceEncoder is not defined for dataset types which are not RepertoireDataset.")

    def build_distance_matrix(self, dataset: RepertoireDataset, params: EncoderParams):
        comparison = PairwiseRepertoireComparison(self.attributes_to_match, self.attributes_to_match, params["result_path"],
                                                  params["batch_size"], self.build_matching_fn())

        current_dataset = dataset if self.context is None or "dataset" not in self.context else self.context["dataset"]

        distance_matrix = comparison.compare_repertoires(current_dataset, self.distance_fn)

        repertoire_ids = [repertoire.identifier for repertoire in dataset.get_data()]

        distance_matrix = distance_matrix.loc[repertoire_ids, repertoire_ids]

        return distance_matrix

    def build_labels(self, dataset: RepertoireDataset, params: EncoderParams) -> dict:

        labels = {label: [] for label in params["label_configuration"].get_labels_by_name()}

        for repertoire in dataset.get_data():
            for label in params["label_configuration"].get_labels_by_name():
                labels[label].append(repertoire.metadata.custom_params[label])

        return labels

    def encode(self, dataset, params: EncoderParams) -> RepertoireDataset:

        distance_matrix = self.build_distance_matrix(dataset, params)

        labels = self.build_labels(dataset, params)

        encoded_dataset = copy.deepcopy(dataset)
        encoded_dataset.encoded_data = EncodedData(examples=distance_matrix, labels=labels, example_ids=distance_matrix.columns.values,
                                                   encoding=DistanceEncoder.__name__)

        return encoded_dataset

    def build_matching_fn(self):
        return lambda repertoire: pd.DataFrame([[item.get_attribute(attribute) for attribute in self.attributes_to_match]
                                                for item in repertoire.sequences], columns=self.attributes_to_match)

    def store(self, encoded_dataset, params: EncoderParams):
        PickleExporter.export(encoded_dataset, params["result_path"], params["filename"])
