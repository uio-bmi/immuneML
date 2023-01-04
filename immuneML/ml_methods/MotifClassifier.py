import copy
import logging
import warnings
from pathlib import Path

import numpy as np
import yaml

from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.environment.Label import Label
from immuneML.ml_methods.MLMethod import MLMethod
from immuneML.ml_methods.util.Util import Util
from immuneML.ml_metrics.Metric import Metric
from immuneML.ml_metrics.MetricUtil import MetricUtil
from immuneML.util.PathBuilder import PathBuilder


class MotifClassifier(MLMethod): # todo name? (Greedy)BinaryFeatureClassifier? RuleTree? something with OR?
    """

    Arguments:

        training_percentage (float): what percentage of data to use for training (the rest will be used for validation); values between 0 and 1



    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_motif_classifier:
            MotifClassifier:
                ...

    """

    def __init__(self, training_percentage: float = None, max_motifs: int = None,
                 patience: int = None, min_delta: float = None, keep_all: bool = None,
                 result_path: Path = None):
        super().__init__()
        self.training_percentage = training_percentage
        self.max_motifs = max_motifs
        self.patience = patience
        self.min_delta = min_delta
        self.keep_all = keep_all

        self.feature_names = None
        self.rule_tree_indices = None
        self.rule_tree_features = None
        self.label = None
        self.optimization_metric = None
        self.class_mapping = None
        self.result_path = result_path

    def predict(self, encoded_data: EncodedData, label: Label):
        return {self.label.name: self._get_rule_tree_predictions_class(encoded_data, self.rule_tree_indices)}

    def predict_proba(self, encoded_data: EncodedData, label: Label):
        warnings.warn(f"{MotifClassifier.__name__}: cannot predict probabilities.")
        return None

    def fit(self, encoded_data: EncodedData, label: Label, optimization_metric: str, cores_for_training: int = 2):
        self.feature_names = encoded_data.feature_names
        self.label = label
        self.class_mapping = Util.make_binary_class_mapping(encoded_data.labels[self.label.name])
        self.optimization_metric = optimization_metric
        
        # todo deal with positive_class, what if it is not explicitly set?
        # todo generalize with the positive class label stuff in MotifEncoder
        # todo weights in immuneML general must also be recalculated here for specific training and validation sets!

        self.rule_tree_indices = self._build_rule_tree(encoded_data)
        self.rule_tree_features = self._get_rule_tree_features_from_indices(self.rule_tree_indices, self.feature_names)

        logging.info(f"{MotifClassifier.__name__}: finished training.")

    def _build_rule_tree(self, encoded_data):
        if self.keep_all:
            return list(range(len(self.feature_names)))
        else:
            encoded_train_data, encoded_val_data = self._prepare_and_split_data(encoded_data)
            return self._recursively_select_rules(encoded_train_data=encoded_train_data,
                                                  encoded_val_data=encoded_val_data,
                                                  last_val_scores=[], prev_rule_indices=[])

    def _get_rule_tree_features_from_indices(self, rule_tree_indices, feature_names):
        return [feature_names[idx] for idx in rule_tree_indices]

    def _recursively_select_rules(self, encoded_train_data, encoded_val_data, last_val_scores, prev_rule_indices):
        new_rule_indices = self._add_next_best_rule(encoded_train_data, prev_rule_indices)

        if new_rule_indices == prev_rule_indices or len(new_rule_indices) > self.max_motifs:
            logging.info(f"{MotifClassifier.__name__}: no improvement on training set or max motifs reached")

            is_improvement = self._test_is_improvement(last_val_scores, self.min_delta)
            return self._get_optimal_indices(new_rule_indices, is_improvement)

        val_scores = last_val_scores + [self._test_performance_rule_tree(encoded_data=encoded_val_data, rule_indices=new_rule_indices)]
        is_improvement = self._test_is_improvement(val_scores, self.min_delta)

        if self._test_earlystopping(is_improvement): # originally also included 'if args.earlystopping'
            logging.info(f"{MotifClassifier.__name__}: reached earlystopping criterion")

            return self._get_optimal_indices(new_rule_indices, is_improvement)

        return self._recursively_select_rules(encoded_train_data, encoded_val_data,
                                        last_val_scores=val_scores, prev_rule_indices=new_rule_indices)

    def _test_earlystopping(self, is_improvement):
        # patience has not reached yet, continue training
        if len(is_improvement) < self.patience:
            return False

        # last few trees did not improve, stop training
        if not any(is_improvement[-self.patience:]):
            return True

        return False

    def _test_is_improvement(self, scores, min_delta):
        best = scores[0]
        is_improvement = [True]

        for score in scores[1:]:
            if score > best + min_delta:
                best = score
                is_improvement.append(True)
            else:
                is_improvement.append(False)

        return is_improvement

    def _get_optimal_indices(self, rule_indices, is_improvement):
        optimal_tree_idx = max([i if is_improvement[i] else -1 for i in range(len(is_improvement))])

        return rule_indices[:optimal_tree_idx + 1]

    def _add_next_best_rule(self, encoded_train_data, prev_rule_indices):
        prev_train_performance = self._get_prev_train_performance(encoded_train_data, prev_rule_indices)

        unused_indices = self._get_unused_rule_indices(encoded_train_data, prev_rule_indices)

        if len(unused_indices) == 0:
            return prev_rule_indices

        new_training_performances = self._get_new_performances(encoded_train_data, prev_rule_indices=prev_rule_indices, new_indices_to_test=unused_indices)

        best_new_performance = max(new_training_performances)
        best_new_index = unused_indices[new_training_performances.index(best_new_performance)]

        if best_new_performance > prev_train_performance:
            return prev_rule_indices + [best_new_index]
        else:
            return prev_rule_indices

    def _get_prev_train_performance(self, encoded_train_data, prev_rule_indices):
        if not prev_rule_indices:
            return 0
        else:
            return self._test_performance_rule_tree(encoded_train_data, rule_indices=prev_rule_indices)

    def _get_unused_rule_indices(self, encoded_train_data, rule_indices):
        return [idx for idx in range(encoded_train_data.examples.shape[1]) if idx not in rule_indices]

    def _get_new_performances(self, encoded_data, prev_rule_indices, new_indices_to_test):
        return [self._test_performance_rule_tree(encoded_data, rule_indices=prev_rule_indices + [idx]) for idx in new_indices_to_test]

    def _test_performance_rule_tree(self, encoded_data, rule_indices):
        optimization_scoring_fn = MetricUtil.get_metric_fn(Metric[self.optimization_metric.upper()])
        pred = self._get_rule_tree_predictions_bool(encoded_data, rule_indices)

        y_true = Util.map_to_new_class_values(encoded_data.labels[self.label.name], self.class_mapping)

        return optimization_scoring_fn(y_true=y_true, y_pred=pred, sample_weight=encoded_data.example_weights)

    def _get_rule_tree_predictions_bool(self, encoded_data, rule_indices):
        self._check_features(encoded_data.feature_names)
        return np.logical_or.reduce([encoded_data.examples[:, i] for i in rule_indices])

    def _get_rule_tree_predictions_class(self, encoded_data, rule_indices):
        y = self._get_rule_tree_predictions_bool(encoded_data, rule_indices).astype(int)
        return Util.map_to_old_class_values(y, self.class_mapping)

    def _check_features(self, encoded_data_features):
        if self.feature_names != encoded_data_features:
            mssg = f"{MotifClassifier.__name__}: features during evaluation did not match the features set during fitting."

            logging.info(mssg + f"\n\nEvaluation features: {encoded_data_features}\nFitting features: {self.feature_names}")
            raise ValueError(mssg + " See the log file for more info.")

    def fit_by_cross_validation(self, encoded_data: EncodedData, label: Label = None, optimization_metric: str = None,
                                number_of_splits: int = 5, cores_for_training: int = -1):
        logging.warning(f"{MotifClassifier.__name__}: cross_validation is not implemented for this method. Using standard fitting instead...")
        self.fit(encoded_data=encoded_data, label=label)


    def _prepare_and_split_data(self, encoded_data: EncodedData):
        train_indices, val_indices = Util.get_train_val_indices(len(encoded_data.example_ids), self.training_percentage)

        train_data = Util.subset_encoded_data(encoded_data, train_indices)
        val_data = Util.subset_encoded_data(encoded_data, val_indices)

        return train_data, val_data

    def store(self, path: Path, feature_names=None, details_path: Path = None):
        PathBuilder.build(path)

        custom_vars = copy.deepcopy(vars(self))
        del custom_vars["result_path"]

        if self.label:
            custom_vars["label"] = vars(self.label)

        params_path = path / "custom_params.yaml"
        with params_path.open('w') as file:
            yaml.dump(custom_vars, file)

    def load(self, path):
        params_path = path / "custom_params.yaml"
        with params_path.open("r") as file:
            custom_params = yaml.load(file, Loader=yaml.SafeLoader)

        for param, value in custom_params.items():
            if hasattr(self, param):
                if param == "label":
                    setattr(self, "label", Label(**value))
                else:
                    setattr(self, param, value)

    def check_if_exists(self, path):
        return self.rule_tree_indices is not None

    def get_params(self):
        params = copy.deepcopy(vars(self))
        return params

    def get_label_name(self):
        return self.label.name

    def get_package_info(self) -> str:
        return Util.get_immuneML_version()

    def get_feature_names(self) -> list:
        return self.feature_names

    def can_predict_proba(self) -> bool:
        return False

    def get_class_mapping(self) -> dict:
        return self.class_mapping

    def get_compatible_encoders(self):
        from immuneML.encodings.motif_encoding.MotifEncoder import MotifEncoder
        return [MotifEncoder]




