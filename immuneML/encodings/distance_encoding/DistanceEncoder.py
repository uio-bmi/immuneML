import warnings
from pathlib import Path

import pandas as pd

from immuneML.IO.ml_method.UtilIO import UtilIO
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.data_model.EncodedData import EncodedData
from immuneML.data_model.SequenceSet import Repertoire
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.distance_encoding.DistanceMetricType import DistanceMetricType
from immuneML.pairwise_repertoire_comparison.PairwiseRepertoireComparison import PairwiseRepertoireComparison
from immuneML.util import DistanceMetrics
from immuneML.util.EncoderHelper import EncoderHelper
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.ReflectionHandler import ReflectionHandler
from scripts.specification_util import update_docs_per_mapping


class DistanceEncoder(DatasetEncoder):
    """
    Encodes a given RepertoireDataset as distance matrix, where the pairwise distance between each of the repertoires
    is calculated. The distance is calculated based on the presence/absence of elements defined under attributes_to_match.
    Thus, if attributes_to_match contains only 'sequence_aas', this means the distance between two repertoires is maximal
    if they contain the same set of sequence_aas, and the distance is minimal if none of the sequence_aas are shared between
    two repertoires.

    **Specification arguments:**

    - distance_metric (:py:mod:`~immuneML.encodings.distance_encoding.DistanceMetricType`): The metric used to calculate the
      distance between two repertoires. Names of different distance metric types are allowed values in the specification.
      The default distance metric is JACCARD (inverse Jaccard).

    - sequence_batch_size (int): The number of sequences to be processed at once. Increasing this number increases the memory use.
      The default value is 1000.

    - attributes_to_match (list): The attributes to consider when determining whether a sequence is present in both repertoires.
      Only the fields defined under attributes_to_match will be considered, all other fields are ignored.
      Valid values include any repertoire attribute as defined in AIRR rearrangement schema (cdr3_aa, v_call, j_call, etc).

    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            encodings:
                my_distance_encoder:
                    Distance:
                        distance_metric: JACCARD
                        sequence_batch_size: 1000
                        attributes_to_match:
                            - cdr3_aa
                            - v_call
                            - j_call

    """

    def __init__(self, distance_metric: DistanceMetricType, attributes_to_match: list, sequence_batch_size: int, context: dict = None,
                 name: str = None):
        super().__init__(name=name)
        self.distance_metric = distance_metric
        self.distance_fn = ReflectionHandler.import_function(self.distance_metric.value, DistanceMetrics)
        self.attributes_to_match = attributes_to_match
        self.sequence_batch_size = sequence_batch_size
        self.context = context
        self.comparison = None

    def set_context(self, context: dict):
        self.context = context
        return self

    @staticmethod
    def _prepare_parameters(distance_metric: str, attributes_to_match: list, sequence_batch_size: int, context: dict = None, name: str = None):
        valid_metrics = [metric.name for metric in DistanceMetricType]
        ParameterValidator.assert_in_valid_list(distance_metric, valid_metrics, "DistanceEncoder", "distance_metric")

        return {
            "distance_metric": DistanceMetricType[distance_metric.upper()],
            "attributes_to_match": attributes_to_match,
            "sequence_batch_size": sequence_batch_size,
            "context": context,
            "name": name
        }

    @staticmethod
    def build_object(dataset, **params):
        if isinstance(dataset, RepertoireDataset):
            prepared_params = DistanceEncoder._prepare_parameters(**params)
            return DistanceEncoder(**prepared_params)
        else:
            raise ValueError("DistanceEncoder is not defined for dataset types which are not RepertoireDataset.")

    def build_distance_matrix(self, dataset: RepertoireDataset, params: EncoderParams, train_repertoire_ids: list):
        self.comparison = PairwiseRepertoireComparison(self.attributes_to_match, self.attributes_to_match, params.result_path,
                                                  sequence_batch_size=self.sequence_batch_size)

        current_dataset = dataset if self.context is None or "dataset" not in self.context else self.context["dataset"]

        distance_matrix = self.comparison.compare(current_dataset, self.distance_fn, self.distance_metric.value)

        repertoire_ids = dataset.get_repertoire_ids()

        distance_matrix = distance_matrix.loc[repertoire_ids, train_repertoire_ids]

        return distance_matrix

    def build_labels(self, dataset: RepertoireDataset, params: EncoderParams) -> dict:

        lbl = ["identifier"]
        lbl.extend(params.label_config.get_labels_by_name())

        tmp_labels = dataset.get_metadata(lbl, return_df=True)
        tmp_labels = tmp_labels.iloc[pd.Index(tmp_labels['identifier']).get_indexer(dataset.get_repertoire_ids())]
        tmp_labels = tmp_labels.to_dict("list")
        del tmp_labels["identifier"]

        return tmp_labels

    def encode(self, dataset, params: EncoderParams) -> RepertoireDataset:
        train_repertoire_ids = EncoderHelper.prepare_training_ids(dataset, params)
        distance_matrix = self.build_distance_matrix(dataset, params, train_repertoire_ids)
        labels = self.build_labels(dataset, params) if params.encode_labels else None

        encoded_dataset = dataset.clone()
        encoded_dataset.encoded_data = EncodedData(examples=distance_matrix, labels=labels, example_ids=distance_matrix.index.values,
                                                   encoding=DistanceEncoder.__name__)

        return encoded_dataset


    @staticmethod
    def load_encoder(encoder_file: Path):
        encoder = DatasetEncoder.load_encoder(encoder_file)
        encoder.comparison = UtilIO.import_comparison_data(encoder_file.parent)
        return encoder

    @staticmethod
    def get_documentation():
        doc = str(DistanceEncoder.__doc__)

        valid_values = [metric.name for metric in DistanceMetricType]
        valid_values = str(valid_values)[1:-1].replace("'", "`")
        mapping = {
            "Names of different distance metric types are allowed values in the specification.": f"Valid values are: {valid_values}.",
        }
        doc = update_docs_per_mapping(doc, mapping)
        return doc
