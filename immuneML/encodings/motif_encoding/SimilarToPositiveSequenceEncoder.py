from pathlib import Path

import numpy as np

from immuneML.analysis.SequenceMatcher import SequenceMatcher
from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.util.EncoderHelper import EncoderHelper
from immuneML.util.ParameterValidator import ParameterValidator

from immuneML.util.PathBuilder import PathBuilder


class SimilarToPositiveSequenceEncoder(DatasetEncoder):
    """
    A simple baseline encoding, to be used in combination with <todo add>
    This encoder keeps track of all positive sequences in the training set, and ignores the negative sequences.
    Any sequence within a given hamming distance from a positive training sequence will be classified positive,
    all other sequences will be classified negative.

    Arguments:

        hamming_distance (int):


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

            my_sequence_encoder:
                SimilarToPositiveSequenceEncoder:
                    hamming_distance: 2
    """

    def __init__(self, hamming_distance: int = None, name: str = None):
        self.hamming_distance = hamming_distance

        self.positive_sequences = None
        self.name = name
        self.context = None

    @staticmethod
    def _prepare_parameters(hamming_distance: int = None, name: str = None):
        location = SimilarToPositiveSequenceEncoder.__name__

        ParameterValidator.assert_type_and_value(hamming_distance, int, location, "hamming_distance", min_inclusive=0)

        return {
            "hamming_distance": hamming_distance,
            "name": name,
        }

    @staticmethod
    def build_object(dataset=None, **params):
        if isinstance(dataset, SequenceDataset):
            prepared_params = SimilarToPositiveSequenceEncoder._prepare_parameters(**params)
            return SimilarToPositiveSequenceEncoder(**prepared_params)
        else:
            raise ValueError(f"{SimilarToPositiveSequenceEncoder.__name__} is not defined for dataset types which are not SequenceDataset.")

    def encode(self, dataset, params: EncoderParams):
        if params.learn_model:
            EncoderHelper.check_positive_class_labels(params.label_config, SimilarToPositiveSequenceEncoder.__name__)

            self.positive_sequences = self._get_positive_sequences(dataset, params)

        return self._encode_data(dataset, params)

    def _get_positive_sequences(self, dataset, params):
        subset_path = PathBuilder.build(params.result_path / "positive_sequences")
        label_name = EncoderHelper.get_single_label_name_from_config(params.label_config,
                                                                    SimilarToPositiveSequenceEncoder.__name__)

        label_obj = params.label_config.get_label_object(label_name)
        classes = dataset.get_metadata([label_name])[label_name]

        subset_indices = [idx for idx in range(dataset.get_example_count()) if classes[idx] == label_obj.positive_class]

        return dataset.make_subset(subset_indices,
                                   path=subset_path,
                                   dataset_type=Dataset.SUBSAMPLED)

    def _encode_data(self, dataset, params: EncoderParams):
        examples = self.get_sequence_matching_feature(dataset)

        labels = EncoderHelper.encode_element_dataset_labels(dataset, params.label_config)

        encoded_dataset = dataset.clone()
        encoded_dataset.encoded_data = EncodedData(examples=examples,
                                                   labels=labels,
                                                   feature_names=["similar_to_positive_sequence"],
                                                   feature_annotations=None,
                                                   example_ids=dataset.get_example_ids(),
                                                   encoding=SimilarToPositiveSequenceEncoder.__name__,
                                                   example_weights=dataset.get_example_weights(),
                                                   info={})

        return encoded_dataset

    def get_sequence_matching_feature(self, dataset):
        matcher = SequenceMatcher()

        examples = []

        for sequence in dataset.get_data():
            is_matching = False

            for ref_sequence in self.positive_sequences.get_data():
                if matcher.matches_sequence(sequence, ref_sequence, self.hamming_distance):
                    is_matching = True
                    break

            examples.append(is_matching)

        return np.array([examples]).T

    def _get_y_true(self, dataset, label_config: LabelConfiguration):
        labels = EncoderHelper.encode_element_dataset_labels(dataset, label_config)

        label_name = EncoderHelper.get_single_label_name_from_config(label_config, SimilarToPositiveSequenceEncoder.__name__)
        label = label_config.get_label_object(label_name)

        return np.array([cls == label.positive_class for cls in labels[label_name]])

    def _get_positive_class(self, label_config):
        label_name = EncoderHelper.get_single_label_name_from_config(label_config, SimilarToPositiveSequenceEncoder.__name__)
        label = label_config.get_label_object(label_name)

        return label.positive_class

    def set_context(self, context: dict):
        self.context = context
        return self

    @staticmethod
    def export_encoder(path: Path, encoder) -> Path:
        encoder_file = DatasetEncoder.store_encoder(encoder, path / "encoder.pickle")
        return encoder_file

    @staticmethod
    def load_encoder(encoder_file: Path):
        encoder = DatasetEncoder.load_encoder(encoder_file)
        return encoder
