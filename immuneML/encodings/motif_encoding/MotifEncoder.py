import logging
from functools import partial
from multiprocessing.pool import Pool
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
from immuneML.util.NumpyHelper import NumpyHelper
from immuneML.util.ParameterValidator import ParameterValidator


from immuneML.encodings.motif_encoding.PositionalMotifHelper import PositionalMotifHelper


class MotifEncoder(DatasetEncoder):
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

        candidate_motif_filepath (str):

        label (str):


        # todo should weighting be a parameter here?





    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

            my_motif_encoder:
                MotifEncoder:
                    max_positions: 5
                    min_precision: 0.9
                    min_recall: 0.1
                    min_true_positives: 10
                    generalize_motifs: False




    """

    dataset_mapping = {
        "SequenceDataset": "PositionalMotifSequenceEncoder",
    }

    def __init__(self, max_positions: int = None, min_precision: float = None, min_recall: float = None,
                 min_true_positives: int = None, generalize_motifs: bool = False,
                 candidate_motif_filepath: str = None, label: str = None, name: str = None):
        self.max_positions = max_positions
        self.min_precision = min_precision
        self.min_recall = min_recall
        self.min_true_positives = min_true_positives
        self.generalize_motifs = generalize_motifs
        self.candidate_motif_filepath = Path(candidate_motif_filepath) if candidate_motif_filepath is not None else None
        self.learned_motif_filepath = None

        self.label = label
        self.name = name
        self.context = None

    @staticmethod
    def _prepare_parameters(max_positions: int = None, min_precision: float = None, min_recall: float = None,
                            min_true_positives: int = None, generalize_motifs: bool = False,
                            candidate_motif_filepath: str = None, label: str = None, name: str = None):

        location = MotifEncoder.__name__

        ParameterValidator.assert_type_and_value(max_positions, int, location, "max_positions", min_inclusive=1)
        ParameterValidator.assert_type_and_value(min_precision, (int, float), location, "min_precision", min_inclusive=0, max_inclusive=1)
        ParameterValidator.assert_type_and_value(min_recall, (int, float), location, "min_recall", min_inclusive=0, max_inclusive=1)
        ParameterValidator.assert_type_and_value(min_true_positives, int, location, "min_true_positives", min_inclusive=1)
        ParameterValidator.assert_type_and_value(generalize_motifs, bool, location, "generalize_motifs")

        if candidate_motif_filepath is not None:
            PositionalMotifHelper.check_motif_filepath(candidate_motif_filepath, location)

        if label is not None:
            ParameterValidator.assert_type_and_value(label, str, location, "label")

        return {
            "max_positions": max_positions,
            "min_precision": min_precision,
            "min_recall": min_recall,
            "min_true_positives": min_true_positives,
            "generalize_motifs": generalize_motifs,
            "candidate_motif_filepath": candidate_motif_filepath,
            "label": label,
            "name": name,
        }

    @staticmethod
    def build_object(dataset=None, **params):
        if isinstance(dataset, SequenceDataset):
            prepared_params = MotifEncoder._prepare_parameters(**params)
            return MotifEncoder(**prepared_params)
        else:
            raise ValueError(f"{MotifEncoder.__name__} is not defined for dataset types which are not SequenceDataset.")

    def encode(self, dataset, params: EncoderParams):
        if params.learn_model:
            EncoderHelper.check_positive_class_labels(params.label_config, MotifEncoder.__name__)
            return self._encode_data(dataset, params)
        else:
            learned_motifs = PositionalMotifHelper.read_motifs_from_file(self.learned_motif_filepath)
            return self.get_encoded_dataset_from_motifs(dataset, learned_motifs, params.label_config, params.pool_size)

    def _encode_data(self, dataset, params: EncoderParams):
        learned_motifs = self._compute_motifs(dataset, params)

        self.learned_motif_filepath = params.result_path / "significant_motifs.tsv"
        PositionalMotifHelper.write_motifs_to_file(learned_motifs, self.learned_motif_filepath)

        return self.get_encoded_dataset_from_motifs(dataset, learned_motifs, params.label_config, params.pool_size)

    def _compute_motifs(self, dataset, params):
        motifs = self._prepare_candidate_motifs(dataset, params)

        y_true = self._get_y_true(dataset, params.label_config)

        motifs = self._filter_motifs(motifs, dataset, y_true, params.pool_size,
                                     min_recall=self.min_recall, generalized=False)

        if self.generalize_motifs:
            generalized_motifs = PositionalMotifHelper.get_generalized_motifs(motifs)
            generalized_motifs = self._filter_motifs(generalized_motifs, dataset, y_true, params.pool_size,
                                                     min_recall=self.min_recall, generalized=True)

            motifs += generalized_motifs

        return motifs

    def get_encoded_dataset_from_motifs(self, dataset, motifs, label_config, number_of_processes):
        labels = EncoderHelper.encode_element_dataset_labels(dataset, label_config)

        examples, feature_names, feature_annotations = self._construct_encoded_data_matrix(dataset, motifs,
                                                                                           label_config, number_of_processes)

        encoded_dataset = dataset.clone()
        encoded_dataset.encoded_data = EncodedData(examples=examples,
                                                   labels=labels,
                                                   feature_names=feature_names,
                                                   feature_annotations=feature_annotations,
                                                   example_ids=dataset.get_example_ids(),
                                                   encoding=MotifEncoder.__name__,
                                                   example_weights=dataset.get_example_weights(),
                                                   info={"candidate_motif_filepath": self.candidate_motif_filepath,
                                                         "learned_motif_filepath": self.learned_motif_filepath,
                                                         "min_precision": self.min_precision,
                                                         "min_recall": self.min_recall})

        return encoded_dataset

    def _prepare_candidate_motifs(self, dataset, params):
        full_dataset = EncoderHelper.get_current_dataset(dataset, self.context)
        candidate_motifs = self._get_candidate_motifs(full_dataset, params.pool_size)
        assert len(candidate_motifs) > 0, f"{MotifEncoder.__name__}: no candidate motifs were found. " \
                                          f"Please try decreasing the value for parameter 'min_true_positives'."

        self.candidate_motif_filepath = params.result_path / "all_candidate_motifs.tsv"
        PositionalMotifHelper.write_motifs_to_file(candidate_motifs, self.candidate_motif_filepath)

        return candidate_motifs

    def _get_candidate_motifs(self, full_dataset, pool_size=4):
        '''Returns all candidate motifs, which are either read from the input file or computed by finding
        all motifs occuring in at least a given number of sequences of the full dataset.'''
        if self.candidate_motif_filepath is None:
            return CacheHandler.memo_by_params(self._build_candidate_motifs_params(full_dataset),
                                               lambda: self._compute_candidate_motifs(full_dataset, pool_size))
        else:
            return PositionalMotifHelper.read_motifs_from_file(self.candidate_motif_filepath)

    def _build_candidate_motifs_params(self, dataset: SequenceDataset):
        return (("dataset_identifier", dataset.identifier),
                ("sequence_ids", tuple(dataset.get_example_ids()),
                ("example_weights", type(dataset.get_example_weights())),
                ("max_positions", self.max_positions),
                ("min_true_positives", self.min_true_positives)))

    def _compute_candidate_motifs(self, full_dataset, pool_size=4):
        np_sequences = NumpyHelper.get_numpy_sequence_representation(full_dataset)
        params = PositionalMotifParams(max_positions=self.max_positions, count_threshold=self.min_true_positives,
                                       pool_size=pool_size)
        return PositionalMotifHelper.compute_all_candidate_motifs(np_sequences, params)

    def _get_y_true(self, dataset, label_config: LabelConfiguration):
        labels = EncoderHelper.encode_element_dataset_labels(dataset, label_config)

        label_name = self._get_label_name(label_config)
        label = label_config.get_label_object(label_name)

        return np.array([cls == label.positive_class for cls in labels[label_name]])

    def _get_label_name(self, label_config: LabelConfiguration):
        if self.label is not None:
            assert self.label in label_config.get_labels_by_name(), f"{MotifEncoder.__name__}: specified label " \
                                                                    f"'{self.label}' was not present among the dataset labels: " \
                                                                    f"{', '.join(label_config.get_labels_by_name())}"
            label_name = self.label
        else:
            assert label_config.get_label_count() != 0, f"{MotifEncoder.__name__}: the dataset does not contain labels, please specify a label under 'instructions'."
            assert label_config.get_label_count() == 1, f"{MotifEncoder.__name__}: multiple labels were found: {', '.join(label_config.get_labels_by_name())}. " \
                                                        f"Please reduce the number of labels to one, or use the parameter 'label' to specify one of these labels. "

            label_name = label_config.get_labels_by_name()[0]

        return label_name

    def check_filtered_motifs(self, filtered_motifs):
        assert len(filtered_motifs) > 0, f"{MotifEncoder.__name__}: no significant motifs were found. " \
                                         f"Please try decreasing the values for parameters 'min_precision' or 'min_recall'"

    def _filter_motifs(self, candidate_motifs, dataset, y_true, pool_size, min_recall, generalized=False):
        motif_type = "generalized motifs" if generalized else "motifs"

        logging.info(f"{MotifEncoder.__name__}: filtering {len(candidate_motifs)} {motif_type} with precision >= {self.min_precision} and recall >= {min_recall}")

        np_sequences = NumpyHelper.get_numpy_sequence_representation(dataset)
        weights = dataset.get_example_weights()

        with Pool(pool_size) as pool:
            partial_func = partial(self._check_motif,  np_sequences=np_sequences, y_true=y_true, weights=weights, min_recall=min_recall)

            filtered_motifs = list(filter(None, pool.map(partial_func, candidate_motifs)))

        if not generalized:
            self.check_filtered_motifs(filtered_motifs)

        logging.info(f"{MotifEncoder.__name__}: filtering {motif_type} done, {len(filtered_motifs)} motifs left")

        return filtered_motifs

    def _check_motif(self, motif, np_sequences, y_true, weights, min_recall):
        indices, amino_acids = motif

        pred = PositionalMotifHelper.test_motif(np_sequences, indices, amino_acids)

        if sum(pred & y_true) >= self.min_true_positives:
            if precision_score(y_true=y_true, y_pred=pred, sample_weight=weights) >= self.min_precision:
                if recall_score(y_true=y_true, y_pred=pred, sample_weight=weights) >= min_recall:
                    return motif

    def _construct_encoded_data_matrix(self, dataset, motifs, label_config, number_of_processes):
        feature_names = [PositionalMotifHelper.motif_to_string(indices, amino_acids, motif_sep="-", newline=False)
                         for indices, amino_acids in motifs]

        weights = dataset.get_example_weights()
        y_true = self._get_y_true(dataset, label_config)
        np_sequences = NumpyHelper.get_numpy_sequence_representation(dataset)
        predictions = self._get_predictions(np_sequences, motifs, number_of_processes)

        precision_scores = [precision_score(y_true=y_true, y_pred=pred, sample_weight=weights) for pred in predictions]
        recall_scores = [recall_score(y_true=y_true, y_pred=pred, sample_weight=weights) for pred in predictions]
        tp_counts = [sum(pred & y_true) for pred in predictions]

        prefix = "weighted_" if weights is not None else ""

        feature_annotations = pd.DataFrame({"feature_names": feature_names,
                                            f"{prefix}precision_scores": precision_scores,
                                            f"{prefix}recall_scores": recall_scores,
                                            "raw_tp_count": tp_counts})

        return np.column_stack(predictions), feature_names, feature_annotations

    def _get_predictions(self, np_sequences, motifs, number_of_processes):
        with Pool(number_of_processes) as pool:
            partial_func = partial(self._test_motif, np_sequences=np_sequences)
            predictions = pool.starmap(partial_func, motifs)

        return predictions

    def _test_motif(self, indices, amino_acids, np_sequences):
        return PositionalMotifHelper.test_motif(np_sequences=np_sequences, indices=indices, amino_acids=amino_acids)

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
