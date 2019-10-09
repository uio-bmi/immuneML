import copy
import pickle

import pandas as pd

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.encoded_data.EncodedData import EncodedData
from source.encodings.DatasetEncoder import DatasetEncoder
from source.encodings.EncoderParams import EncoderParams
from source.encodings.distance_encoding.DistanceMetricType import DistanceMetricType
from source.pairwise_repertoire_comparison.PairwiseRepertoireComparison import PairwiseRepertoireComparison
from source.util import DistanceMetrics
from source.util.PathBuilder import PathBuilder
from source.util.ReflectionHandler import ReflectionHandler


class DistanceEncoder(DatasetEncoder):

    def __init__(self, distance_metric: DistanceMetricType, attributes_to_match: list, context: dict = None):
        self.distance_fn = ReflectionHandler.import_function(distance_metric.value, DistanceMetrics)
        self.distance_metric = distance_metric
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

    def build_distance_matrix(self, dataset: RepertoireDataset, params: EncoderParams, train_repertoire_ids: list):
        comparison = PairwiseRepertoireComparison(self.attributes_to_match, self.attributes_to_match, params["result_path"],
                                                  params["batch_size"], self.build_matching_fn())

        current_dataset = dataset if self.context is None or "dataset" not in self.context else self.context["dataset"]

        distance_matrix = comparison.compare(current_dataset, self.distance_fn, self.distance_metric.value)

        repertoire_ids = dataset.get_repertoire_ids()

        distance_matrix = distance_matrix.loc[repertoire_ids, train_repertoire_ids]

        return distance_matrix

    def build_labels(self, dataset: RepertoireDataset, params: EncoderParams) -> dict:

        labels = {label: [] for label in params["label_configuration"].get_labels_by_name()}

        for repertoire in dataset.get_data():
            for label in params["label_configuration"].get_labels_by_name():
                labels[label].append(repertoire.metadata.custom_params[label])

        return labels

    def encode(self, dataset, params: EncoderParams) -> RepertoireDataset:

        train_repertoire_ids = self.prepare_encoding_params(dataset, params)
        distance_matrix = self.build_distance_matrix(dataset, params, train_repertoire_ids)
        labels = self.build_labels(dataset, params)

        encoded_dataset = copy.deepcopy(dataset)
        encoded_dataset.encoded_data = EncodedData(examples=distance_matrix, labels=labels, example_ids=distance_matrix.index.values,
                                                   encoding=DistanceEncoder.__name__)

        self.store(encoded_dataset, params)

        return encoded_dataset

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

    def build_matching_fn(self):
        return lambda repertoire: pd.DataFrame([[item.get_attribute(attribute) for attribute in self.attributes_to_match]
                                                for item in repertoire.sequences], columns=self.attributes_to_match)

    def store(self, encoded_dataset, params: EncoderParams):
        PickleExporter.export(encoded_dataset, params["result_path"], params["filename"])
