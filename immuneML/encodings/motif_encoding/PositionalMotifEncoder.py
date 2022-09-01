from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score

from immuneML.caching.CacheHandler import CacheHandler
from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.motif_encoding.PositionalMotifParams import PositionalMotifParams
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.util.EncoderHelper import EncoderHelper
from immuneML.util.ParameterValidator import ParameterValidator


from immuneML.encodings.motif_encoding.PositionalMotifHelper import PositionalMotifHelper
from immuneML.encodings.motif_encoding.WeightHelper import WeightHelper


class PositionalMotifEncoder(DatasetEncoder):
    """
    xxx
    todo docs

    can only be used for sequences of the same length

    Arguments:

        max_positions (int):

        min_precision (float):

        min_recall (float):

        min_true_positives (int):

        generalize_motifs (bool):

        motif_candidate_filepath (str):

        label (str):

        # todo should weighting be a parameter here?





    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

            my_motif_encoder:
                MotifEncoder:
                    ...


    """

    dataset_mapping = {
        "SequenceDataset": "PositionalMotifSequenceEncoder",
    }

    def __init__(self, max_positions: int = None, min_precision: float = 0, min_recall: float = 0, min_true_positives: int = 1,
                 generalize_motifs: bool = False, use_weights: bool = True, motif_candidate_filepath: str = None, label: str = None,
                 name: str = None):
        self.max_positions = max_positions
        self.min_precision = min_precision
        self.min_recall = min_recall
        self.min_true_positives = min_true_positives
        self.generalize_motifs = generalize_motifs
        self.use_weights = use_weights
        self.motif_candidate_filepath = motif_candidate_filepath
        self.label = label
        self.name = name
        self.context = None

    @staticmethod
    def _prepare_parameters(max_positions: int = None, min_precision: float = 0, min_recall: float = 0, min_true_positives: int = 1,
                            generalize_motifs: bool = False, use_weights: bool = False, motif_candidate_filepath: str = None,
                            label: str = None, name: str = None):

        location = PositionalMotifEncoder.__name__

        ParameterValidator.assert_type_and_value(max_positions, int, location, "max_positions", min_inclusive=1)
        ParameterValidator.assert_type_and_value(min_precision, (int, float), location, "min_precision", min_inclusive=0, max_inclusive=1)
        ParameterValidator.assert_type_and_value(min_recall, (int, float), location, "min_recall", min_inclusive=0, max_inclusive=1)
        ParameterValidator.assert_type_and_value(min_true_positives, int, location, "min_true_positives", min_inclusive=1)
        ParameterValidator.assert_type_and_value(generalize_motifs, bool, location, "generalize_motifs")
        ParameterValidator.assert_type_and_value(use_weights, bool, location, "use_weights")

        if motif_candidate_filepath is not None:
            ParameterValidator.assert_type_and_value(motif_candidate_filepath, str, location, "motif_candidate_filepath")

            motif_candidate_filepath = Path(motif_candidate_filepath)
            assert motif_candidate_filepath.is_file(), f"{location}: the file {motif_candidate_filepath} does not exist. " \
                                                       f"Specify the correct path under motif_filepath."

            file_columns = list(pd.read_csv(motif_candidate_filepath, sep="\t", iterator=False, dtype=str, nrows=0).columns)

            ParameterValidator.assert_all_in_valid_list(file_columns,
                                                        ["indices", "amino_acids"],
                                                        location, "motif_candidate_filepath (column names)")

        if label is not None:
            ParameterValidator.assert_type_and_value(label, str, location, "label")

        return {
            "max_positions": max_positions,
            "min_precision": min_precision,
            "min_recall": min_recall,
            "min_true_positives": min_true_positives,
            "generalize_motifs": generalize_motifs,
            "use_weights": use_weights,
            "label": label,
            "name": name,
        }

    @staticmethod
    def build_object(dataset=None, **params):
        if isinstance(dataset, SequenceDataset):
            prepared_params = PositionalMotifEncoder._prepare_parameters(**params)
            return PositionalMotifEncoder(**prepared_params)
        else:
            raise ValueError(f"{PositionalMotifEncoder.__name__} is not defined for dataset types which are not SequenceDataset.")

    def encode(self, dataset, params: EncoderParams):
        EncoderHelper.check_positive_class_labels(params.label_config, PositionalMotifEncoder.__name__)

        return self._encode_data(dataset, params)

    def _encode_data(self, dataset, params: EncoderParams):
        candidate_motifs = self._prepare_candidate_motifs(dataset, params)

        labels = EncoderHelper.encode_element_dataset_labels(dataset, params.label_config)
        y_true = self._get_y_true(labels, params.label_config)
        np_sequences = PositionalMotifHelper.get_numpy_sequence_representation(dataset)

        self.weight_pseudocount = 1  # todo these parameters need to be set by the user, probably when changing the YAML specification to specify weights
        self.weight_upper_limit = 1
        self.weight_lower_limit = 0

        positional_weights = self._get_positional_weights(np_sequences)
        sequence_weights = self._get_sequence_weights(np_sequences, positional_weights)
        examples, feature_names = self._construct_encoded_data_matrix(np_sequences, sequence_weights, y_true,
                                                                      candidate_motifs)

        encoded_dataset = dataset.clone()
        encoded_dataset.encoded_data = EncodedData(examples=examples,
                                                   labels=labels,
                                                   feature_names=feature_names,
                                                   example_ids=dataset.get_example_ids(),
                                                   encoding=PositionalMotifEncoder.__name__,
                                                   info={"positional_weights": positional_weights})

        return encoded_dataset


    def _prepare_candidate_motifs(self, dataset, params):
        full_dataset = EncoderHelper.get_current_dataset(dataset, self.context)
        candidate_motifs = self._get_candidate_motifs(full_dataset, params.pool_size)
        assert len(candidate_motifs) > 0, f"{PositionalMotifEncoder.__name__}: no candidate motifs were found. " \
                                          f"Please try decreasing the value for parameter 'min_true_positives'."

        PositionalMotifHelper.write_motifs_to_file(candidate_motifs, params.result_path / "candidate_motifs.tsv")

        return candidate_motifs

    def _get_candidate_motifs(self, full_dataset, pool_size=4):
        '''Returns all candidate motifs, which are either read from the input file or computed by finding
        all motifs occuring in at least a given number of sequences of the full dataset.'''
        if self.motif_candidate_filepath is None:
            return CacheHandler.memo_by_params(self._build_candidate_motifs_params(full_dataset),
                                               lambda: self._compute_candidate_motifs(full_dataset, pool_size))
        else:
            return PositionalMotifHelper.read_motifs_from_file(self.motif_candidate_filepath)

    def _build_candidate_motifs_params(self, dataset: SequenceDataset):
        return (("dataset_identifier", dataset.identifier),
                ("sequence_ids", tuple(dataset.get_example_ids()),
                ("max_positions", self.max_positions),
                ("min_true_positives", self.min_true_positives)))

    def _compute_candidate_motifs(self, full_dataset, pool_size=4):
        np_sequences = PositionalMotifHelper.get_numpy_sequence_representation(full_dataset)
        params = PositionalMotifParams(max_positions=self.max_positions, count_threshold=self.min_true_positives,
                                       pool_size=pool_size)
        return PositionalMotifHelper.compute_all_candidate_motifs(np_sequences, params)

    def _get_y_true(self, labels, label_config: LabelConfiguration):
        label_name = self._get_label_name(label_config)
        label = label_config.get_label_object(label_name)

        return np.array([cls == label.positive_class for cls in labels[label_name]])

    def _get_label_name(self, label_config: LabelConfiguration):
        if self.label is not None:
            assert self.label in label_config.get_labels_by_name(), f"{PositionalMotifEncoder.__name__}: specified label " \
                                                                    f"'{self.label}' was not present among the dataset labels: " \
                                                                    f"{', '.join(label_config.get_labels_by_name())}"
            label_name = self.label
        else:
            assert label_config.get_label_count() != 0, f"{PositionalMotifEncoder.__name__}: the dataset does not contain labels, please specify a label under 'instructions'."
            assert label_config.get_label_count() == 1, f"{PositionalMotifEncoder.__name__}: multiple labels were found: {', '.join(label_config.get_labels_by_name())}. " \
                                                        f"Please reduce the number of labels to one, or use the parameter 'label' to specify one of these labels. "

            label_name = label_config.get_labels_by_name()[0]

        return label_name

    def _get_positional_weights(self, np_sequences):
        return WeightHelper.compute_positional_aa_contributions(np_sequences, self.weight_pseudocount)

    def _get_sequence_weights(self, np_sequences, positional_weights):
        return np.apply_along_axis(WeightHelper.compute_sequence_weight, 1, np_sequences, positional_weights=positional_weights, lower_limit=self.weight_lower_limit, upper_limit=self.weight_upper_limit)

    def check_filtered_motifs(self, filtered_motifs):
        assert len(filtered_motifs) > 0, f"{PositionalMotifEncoder.__name__}: no significant motifs were found. " \
                                         f"Please try decreasing the values for parameters 'min_precision' and/or 'min_recall'."

    def _construct_encoded_data_matrix(self, np_sequences, sequence_weights, y_true, candidate_motifs):
        feature_names = []
        examples = []

        for indices, amino_acids in candidate_motifs:
            pred = PositionalMotifHelper.test_motif(np_sequences, indices, amino_acids)
            if sum(pred) >= self.min_true_positives:
                if precision_score(y_true=y_true, y_pred=pred, sample_weight=sequence_weights) >= self.min_precision:
                    if recall_score(y_true=y_true, y_pred=pred, sample_weight=sequence_weights) >= self.min_recall:
                        feature_names.append(PositionalMotifHelper.motif_to_string(indices, amino_acids, motif_sep="-", newline=False))
                        examples.append(pred)

        self.check_filtered_motifs(feature_names)

        return np.column_stack(examples), feature_names

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
