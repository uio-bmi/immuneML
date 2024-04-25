import logging
import warnings
from functools import partial
from multiprocessing.pool import Pool
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, confusion_matrix

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
from immuneML.util.PathBuilder import PathBuilder


class MotifEncoder(DatasetEncoder):
    """
    This encoder enumerates every possible positional motif, and keeps only the motifs associated with the positive class.
    A 'motif' is defined as a combination of position-specific amino acids. These motifs may contain one or multiple gaps.
    Motifs are filtered out based on a minimal precision and recall threshold for predicting the positive class.

    Note: the MotifEncoder can only be used for sequences of the same length.

    The ideal recall threshold(s) given a user-defined precision threshold can be calibrated using the
    :py:obj:`~immuneML.reports.data_reports.MotifGeneralizationAnalysis` report. It is recommended to first run this report
    in :py:obj:`~immuneML.workflows.instructions.exploratory_analysis.ExploratoryAnalysisInstruction` before using this encoder for ML.

    This encoder can be used in combination with the :py:obj:`~immuneML.ml_methods.BinaryFeatureClassifier` in order to
    learn a minimal set of compatible motifs for predicting the positive class.
    Alternatively, it may be combined with scikit-learn methods, such as for example :py:obj:`~immuneML.ml_methods.LogisticRegression`,
    to learn a weight per motif.


    **Specification arguments:**

    - max_positions (int): The maximum motif size. This is number of positional amino acids the motif consists of (excluding gaps). The default value for max_positions is 4.

    - min_positions (int): The minimum motif size (see also: max_positions). The default value for max_positions is 1.

    - min_precision (float): The minimum precision threshold for keeping a motif. The default value for min_precision is 0.8.

    - min_recall (float): The minimum recall threshold for keeping a motif. The default value for min_precision is 0.
      It is also possible to specify a recall threshold for each motif size. In this case, a dictionary must be specified where
      the motif sizes are keys and the recall values are values. Use the :py:obj:`~immuneML.reports.data_reports.MotifGeneralizationAnalysis` report
      to calibrate the optimal recall threshold given a user-defined precision threshold to ensure generalisability to unseen data.

    - min_true_positives (int): The minimum number of true positive sequences that a motif needs to occur in. The default value for min_true_positives is 10.

    - candidate_motif_filepath (str): Optional filepath for pre-filterd candidate motifs. This may be used to save time. Only the given candidate motifs are considered.
      When this encoder has been run previously, a candidate motifs file named 'all_candidate_motifs.tsv' will have been exported. This file contains all
      possible motifs with high enough min_true_positives without applying precision and recall thresholds.
      The file must be a tab-separated file, structured as follows:

      ========  ==============
      indices    amino_acids
      ========  ==============
      1&2&3      A&G&C
      5&7        E&D
      ========  ==============

      The example above contains two motifs: AGC in positions 123, and E-D in positions 5-7 (with a gap at position 6).

    - label (str): The name of the binary label to train the encoder for. This is only necessary when the dataset contains multiple labels.


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            encodings:
                my_motif_encoder:
                    MotifEncoder:
                        max_positions: 4
                        min_precision: 0.8
                        min_recall:  # different recall thresholds for each motif size
                            1: 0.5   # For shorter motifs, a stricter recall threshold is used
                            2: 0.1
                            3: 0.01
                            4: 0.001
                        min_true_positives: 10




    """

    def __init__(self, max_positions: int = None, min_positions: int = None,
                 min_precision: float = None, min_recall: dict = None,
                 min_true_positives: int = None,
                 candidate_motif_filepath: str = None, label: str = None, name: str = None):
        super().__init__(name=name)
        self.max_positions = max_positions
        self.min_positions = min_positions
        self.min_precision = min_precision
        self.min_recall = min_recall
        self.min_true_positives = min_true_positives
        self.candidate_motif_filepath = Path(candidate_motif_filepath) if candidate_motif_filepath is not None else None
        self.learned_motif_filepath = None

        self.label = label
        self.context = None

    @staticmethod
    def _prepare_parameters(max_positions: int = None, min_positions: int = None, min_precision: float = None, min_recall: dict = None,
                            min_true_positives: int = None, candidate_motif_filepath: str = None, label: str = None, name: str = None):

        location = MotifEncoder.__name__

        ParameterValidator.assert_type_and_value(max_positions, int, location, "max_positions", min_inclusive=1)
        ParameterValidator.assert_type_and_value(min_positions, int, location, "min_positions", min_inclusive=1)
        assert max_positions >= min_positions, f"{location}: max_positions ({max_positions}) must be greater than or equal to min_positions ({min_positions})"

        ParameterValidator.assert_type_and_value(min_precision, (int, float), location, "min_precision", min_inclusive=0, max_inclusive=1)
        ParameterValidator.assert_type_and_value(min_true_positives, int, location, "min_true_positives", min_inclusive=1)

        if isinstance(min_recall, dict):
            assert set(min_recall.keys()) == set(range(min_positions, max_positions+1)), f"{location}: {min_recall} is not a valid value for parameter min_recall. " \
                                                                             f"When setting separate recall cutoffs for each motif size, the keys of the dictionary " \
                                                                             f"must equal to {list(range(min_positions, max_positions+1))}."
            for recall_cutoff in min_recall.values():
                assert isinstance(recall_cutoff, (int, float)) or recall_cutoff is None, f"{location}: {min_recall} is not a valid value for parameter min_recall. " \
                                                                                        f"When setting separate recall cutoffs for each motif size, the values of the dictionary " \
                                                                                        f"must be numeric or None."

            min_recall = {key: value if isinstance(value, (int, float)) else 1 for key, value in min_recall.items()}

        else:
            ParameterValidator.assert_type_and_value(min_recall, (int, float), location, "min_recall", min_inclusive=0, max_inclusive=1)
            min_recall = {motif_size: min_recall for motif_size in range(min_positions, max_positions+1)}

        if candidate_motif_filepath is not None:
            PositionalMotifHelper.check_motif_filepath(candidate_motif_filepath, location, "candidate_motif_filepath")

        if label is not None:
            ParameterValidator.assert_type_and_value(label, str, location, "label")

        return {
            "max_positions": max_positions,
            "min_positions": min_positions,
            "min_precision": min_precision,
            "min_recall": min_recall,
            "min_true_positives": min_true_positives,
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
            return self.get_encoded_dataset_from_motifs(dataset, learned_motifs, params)

    def _encode_data(self, dataset, params: EncoderParams):
        learned_motifs = self._compute_motifs(dataset, params)

        self.learned_motif_filepath = params.result_path / "significant_motifs.tsv"
        self.motif_stats_filepath = params.result_path / "motif_stats.tsv"

        PositionalMotifHelper.write_motifs_to_file(learned_motifs, self.learned_motif_filepath)
        self._write_motif_stats(learned_motifs, self.motif_stats_filepath)

        return self.get_encoded_dataset_from_motifs(dataset, learned_motifs, params)

    def _compute_motifs(self, dataset, params):
        motifs = self._prepare_candidate_motifs(dataset, params)

        y_true = self._get_y_true(dataset, params.label_config)

        motifs = self._filter_motifs(motifs, dataset, y_true, params.pool_size, generalized=False)

        # Option disabled for now
        # if self.generalize_motifs:
        #     motifs += self._filter_motifs(PositionalMotifHelper.get_generalized_motifs(motifs), dataset, y_true, params.pool_size, generalized=True)

        return motifs

    def _write_motif_stats(self, learned_motifs, motif_stats_filepath):
        try:
            data = {}

            data["motif_size"] = list(range(self.min_positions, self.max_positions + 1))
            data["min_precision"] = [self.min_precision] * self.max_positions
            data["min_recall"] = [self.min_recall.get(motif_size, 1) for motif_size in range(self.min_positions, self.max_positions + 1)]

            all_motif_sizes = [len(motif[0]) for motif in learned_motifs]
            data["n_motifs"] = [all_motif_sizes.count(motif_size) for motif_size in range(self.min_positions, self.max_positions + 1)]

            df = pd.DataFrame(data)
            df.to_csv(motif_stats_filepath, index=False, sep="\t")
        except Exception as e:
            warnings.warn(f"{MotifEncoder.__name__}: could not write motif stats. Exception was: {e}")

    def get_encoded_dataset_from_motifs(self, dataset, motifs, params):
        labels = EncoderHelper.encode_element_dataset_labels(dataset, params.label_config)

        examples, feature_names, feature_annotations = self._construct_encoded_data_matrix(dataset, motifs,
                                                                                           params.label_config, params.pool_size)

        self._export_confusion_matrix(params.result_path, feature_annotations)

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
                                                         "positive_class": self._get_positive_class(params.label_config)})

        return encoded_dataset

    def _export_confusion_matrix(self, result_path, feature_annotations):
        try:
            PathBuilder.build(result_path)
            feature_annotations.to_csv(result_path / "confusion_matrix.tsv", index=False, sep="\t")
        except Exception as e:
            logging.exception(f"MotifEncoder: An exception occurred while exporting the confusion matrix: {e}")

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
                ("min_positions", self.min_positions),
                ("min_true_positives", self.min_true_positives)))

    def _compute_candidate_motifs(self, full_dataset, pool_size=4):
        np_sequences = PositionalMotifHelper.get_numpy_sequence_representation(full_dataset)
        params = PositionalMotifParams(max_positions=self.max_positions, min_positions=self.min_positions,
                                       count_threshold=self.min_true_positives, pool_size=pool_size)
        return PositionalMotifHelper.compute_all_candidate_motifs(np_sequences, params)

    def _get_y_true(self, dataset, label_config: LabelConfiguration):
        labels = EncoderHelper.encode_element_dataset_labels(dataset, label_config)

        label_name = self._get_label_name(label_config)
        label = label_config.get_label_object(label_name)

        return np.array([cls == label.positive_class for cls in labels[label_name]])

    def _get_positive_class(self, label_config):
        label_name = self._get_label_name(label_config)
        label = label_config.get_label_object(label_name)

        return label.positive_class

    def _get_label_name(self, label_config: LabelConfiguration):
        if self.label is not None:
            assert self.label in label_config.get_labels_by_name(), f"{MotifEncoder.__name__}: specified label " \
                                                                    f"'{self.label}' was not present among the dataset labels: " \
                                                                    f"{', '.join(label_config.get_labels_by_name())}"
            label_name = self.label
        else:
            label_name = EncoderHelper.get_single_label_name_from_config(label_config, MotifEncoder.__name__)

        return label_name

    def check_filtered_motifs(self, filtered_motifs):
        assert len(filtered_motifs) > 0, f"{MotifEncoder.__name__}: no significant motifs were found. " \
                                         f"Please try decreasing the values for parameters 'min_precision' or 'min_recall'"

    def _get_recall_repr(self):
        '''Returns a string representation of the recall cutoff.'''
        if len(set(self.min_recall.values())) == 1:
            return str(list(self.min_recall.values())[0])
        else:
            return ", ".join([f"{recall} (motif size {motif_size})" for motif_size, recall in self.min_recall.items()])

    def _filter_motifs(self, candidate_motifs, dataset, y_true, pool_size, generalized=False):
        motif_type = "generalized motifs" if generalized else "motifs"

        logging.info(f"{MotifEncoder.__name__}: filtering {len(candidate_motifs)} {motif_type} with precision >= {self.min_precision} and recall >= {self._get_recall_repr()}")

        np_sequences = PositionalMotifHelper.get_numpy_sequence_representation(dataset)
        weights = dataset.get_example_weights()

        with Pool(pool_size) as pool:
            partial_func = partial(self._check_motif,  np_sequences=np_sequences, y_true=y_true, weights=weights)

            filtered_motifs = list(filter(None, pool.map(partial_func, candidate_motifs)))

        if not generalized:
            self.check_filtered_motifs(filtered_motifs)

        logging.info(f"{MotifEncoder.__name__}: filtering {motif_type} done, {len(filtered_motifs)} motifs left")

        return filtered_motifs

    def _check_motif(self, motif, np_sequences, y_true, weights):
        indices, amino_acids = motif

        pred = PositionalMotifHelper.test_motif(np_sequences, indices, amino_acids)

        if sum(pred & y_true) >= self.min_true_positives:
            if precision_score(y_true=y_true, y_pred=pred, sample_weight=weights) >= self.min_precision:
                if len(indices) in self.min_recall.keys():
                    if recall_score(y_true=y_true, y_pred=pred, sample_weight=weights) >= self.min_recall[len(indices)]:
                        return motif



    def _construct_encoded_data_matrix(self, dataset, motifs, label_config, number_of_processes):
        feature_names = [PositionalMotifHelper.motif_to_string(indices, amino_acids, motif_sep="-", newline=False)
                         for indices, amino_acids in motifs]

        weights = dataset.get_example_weights()
        y_true = self._get_y_true(dataset, label_config)
        np_sequences = PositionalMotifHelper.get_numpy_sequence_representation(dataset)

        logging.info(f"{MotifEncoder.__name__}: building encoded data matrix...")

        with Pool(number_of_processes) as pool:
            predictions = pool.starmap(partial(self._test_motif, np_sequences=np_sequences), motifs)
            conf_matrix_raw = np.array(pool.map(partial(self._get_confusion_matrix, y_true=y_true, weights=None), predictions))

            if weights is not None:
                conf_matrix_weighted = np.array(pool.map(partial(self._get_confusion_matrix, y_true=y_true, weights=weights), predictions))
            else:
                conf_matrix_weighted = None

            # precision_scores = pool.map(partial(self._get_precision, y_true=y_true, weights=weights), predictions)
            # recall_scores = pool.map(partial(self._get_recall, y_true=y_true, weights=weights), predictions)
            # tp_counts = pool.map(partial(self._get_tp, y_true=y_true), predictions)

        logging.info(f"{MotifEncoder.__name__}: building encoded data matrix done")

        prefix = "weighted_" if weights is not None else ""

        feature_annotations = self._get_feature_annotations(feature_names, conf_matrix_raw, conf_matrix_weighted)

        return np.column_stack(predictions), feature_names, feature_annotations

    def _get_feature_annotations(self, feature_names, conf_matrix_raw, conf_matrix_weighted):
        feature_annotations_mapping = {"feature_names": feature_names,
                                       "TN": conf_matrix_raw.T[0],
                                       "FP": conf_matrix_raw.T[1],
                                       "FN": conf_matrix_raw.T[2],
                                       "TP": conf_matrix_raw.T[3]}

        if conf_matrix_weighted is not None:
            feature_annotations_mapping["weighted_TN"] = conf_matrix_weighted.T[0]
            feature_annotations_mapping["weighted_FP"] = conf_matrix_weighted.T[1]
            feature_annotations_mapping["weighted_FN"] = conf_matrix_weighted.T[2]
            feature_annotations_mapping["weighted_TP"] = conf_matrix_weighted.T[3]

        return pd.DataFrame(feature_annotations_mapping)

    def _get_predictions(self, np_sequences, motifs, number_of_processes):
        with Pool(number_of_processes) as pool:
            partial_func = partial(self._test_motif, np_sequences=np_sequences)
            predictions = pool.starmap(partial_func, motifs)

        return predictions

    def _test_motif(self, indices, amino_acids, np_sequences):
        return PositionalMotifHelper.test_motif(np_sequences=np_sequences, indices=indices, amino_acids=amino_acids)

    def _get_confusion_matrix(self, pred, y_true, weights):
        return confusion_matrix(y_true=y_true, y_pred=pred, sample_weight=weights).ravel()
    #
    # def _get_precision(self, pred, y_true, weights):
    #     return precision_score(y_true=y_true, y_pred=pred, sample_weight=weights, zero_division=0)
    #
    # def _get_recall(self, pred, y_true, weights):
    #     return recall_score(y_true=y_true, y_pred=pred, sample_weight=weights, zero_division=0)

    # def _get_tp(self, pred, y_true):
    #     return sum(pred & y_true)

    def set_context(self, context: dict):
        self.context = context
        return self

