import sys
import copy
import logging
import warnings
from pathlib import Path

import numpy as np
import yaml

from functools import partial
from multiprocessing.pool import Pool

from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.environment.Label import Label
from immuneML.ml_methods.MLMethod import MLMethod
from immuneML.ml_methods.util.Util import Util
from immuneML.ml_metrics.Metric import Metric
from immuneML.ml_metrics.MetricUtil import MetricUtil
from immuneML.util.PathBuilder import PathBuilder


class BinaryFeatureClassifier(MLMethod):
    """
    A simple classifier that takes in encoded data containing features with only 1/0 or True/False values.
    This classifier tries to select an optimal subset of such binary features, and gives a positive prediction
    if any of the features are 'True'.

    This classifier can be used in combination with the :py:obj:`~immuneML.encodings.motif_encoding.MotifEncoder.MotifEncoder`
    and the todo formatting: SimilarToPositiveSequenceEncoder

    Arguments:

        training_percentage (float): what percentage of data to use for training (the rest will be used for validation); values between 0 and 1

        random_seed (int):


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_motif_classifier:
            MotifClassifier:
                ...

    """

    def __init__(self, training_percentage: float = None,
                 random_seed: int = None, max_features: int = None, patience: int = None,
                 min_delta: float = None, keep_all: bool = None, learn_all: bool = None,
                 result_path: Path = None):
        super().__init__()
        self.training_percentage = training_percentage
        self.random_seed = random_seed
        self.max_features = max_features
        self.patience = patience
        self.min_delta = min_delta
        self.keep_all = keep_all
        self.learn_all = learn_all

        self.train_indices = None
        self.val_indices = None
        self.feature_names = None
        self.rule_tree_indices = None
        self.rule_tree_features = None
        self.label = None
        self.optimization_metric = None
        self.class_mapping = None

        self.result_path = result_path

    def predict(self, encoded_data: EncodedData, label: Label):
        self._check_features(encoded_data.feature_names)

        return {self.label.name: self._get_rule_tree_predictions_class(encoded_data, self.rule_tree_indices)}

    def predict_proba(self, encoded_data: EncodedData, label: Label):
        warnings.warn(f"{BinaryFeatureClassifier.__name__}: cannot predict probabilities.")
        return None

    def fit(self, encoded_data: EncodedData, label: Label, optimization_metric: str, cores_for_training: int = 2):
        self.feature_names = encoded_data.feature_names
        self.label = label
        self.class_mapping = Util.make_binary_class_mapping(encoded_data.labels[self.label.name])
        self.optimization_metric = optimization_metric

        self.rule_tree_indices = self._build_rule_tree(encoded_data, cores_for_training)
        self.rule_tree_features = self._get_rule_tree_features_from_indices(self.rule_tree_indices, self.feature_names)
        self._export_selected_features(self.result_path, self.rule_tree_features)

        logging.info(f"{BinaryFeatureClassifier.__name__}: finished training.")

    def _get_optimization_scoring_fn(self):
        return MetricUtil.get_metric_fn(Metric[self.optimization_metric.upper()])

    def _build_rule_tree(self, encoded_data, cores_for_training):
        if self.keep_all:
            rules = list(range(len(self.feature_names)))
            logging.info(f"{BinaryFeatureClassifier.__name__}: all {len(rules)} rules kept.")
        else:
            encoded_train_data, encoded_val_data = self._prepare_and_split_data(encoded_data)
            if self.learn_all or self.max_features is None:
                self.max_features = encoded_train_data.examples.shape[1]

            rules = self._start_recursive_search(encoded_train_data, encoded_val_data, cores_for_training)

            logging.info(f"{BinaryFeatureClassifier.__name__}: selected {len(rules)} out of {len(self.feature_names)} rules.")

        return rules

    def _start_recursive_search(self, encoded_train_data, encoded_val_data, cores_for_training):
        old_recursion_limit = sys.getrecursionlimit()
        new_recursion_limit = old_recursion_limit + encoded_train_data.examples.shape[1]
        sys.setrecursionlimit(new_recursion_limit)

        rules = self._recursively_select_rules(encoded_train_data=encoded_train_data,
                                               encoded_val_data=encoded_val_data,
                                               prev_rule_indices=[],
                                               prev_train_predictions=np.array([False] * encoded_train_data.examples.shape[0]),
                                               prev_val_predictions=np.array([False] * encoded_val_data.examples.shape[0]),
                                               prev_val_scores=[],
                                               cores_for_training=cores_for_training)

        sys.setrecursionlimit(old_recursion_limit)

        return rules

    def _get_rule_tree_features_from_indices(self, rule_tree_indices, feature_names):
        return [feature_names[idx] for idx in rule_tree_indices]

    def _recursively_select_rules(self, encoded_train_data, encoded_val_data, prev_rule_indices, prev_train_predictions, prev_val_predictions, prev_val_scores, cores_for_training):
        logging.info(f"{BinaryFeatureClassifier.__name__}: adding next best rule")
        new_rule_indices, new_train_predictions = self._add_next_best_rule(encoded_train_data, prev_rule_indices, prev_train_predictions, cores_for_training)
        logging.info(f"{BinaryFeatureClassifier.__name__}: next best rule added")

        if new_rule_indices == prev_rule_indices:
            logging.info(f"{BinaryFeatureClassifier.__name__}: no improvement on training set")
            return self._get_optimal_indices(prev_rule_indices, self._test_is_improvement(prev_val_scores, self.min_delta))

        logging.info(f"{BinaryFeatureClassifier.__name__}: added rule {len(new_rule_indices)}/{min(self.max_features, encoded_train_data.examples.shape[1])}")
        logging.info(f"{BinaryFeatureClassifier.__name__}: computing new val score")
        new_val_predictions = np.logical_or(prev_val_predictions, encoded_val_data.examples[:, new_rule_indices[-1]])
        new_val_scores = prev_val_scores + [self._test_performance_predictions(encoded_val_data, pred=new_val_predictions)]
        logging.info(f"{BinaryFeatureClassifier.__name__}: new val score computed")

        logging.info(f"{BinaryFeatureClassifier.__name__}: computing is improvement")
        is_improvement = self._test_is_improvement(new_val_scores, self.min_delta)
        logging.info(f"{BinaryFeatureClassifier.__name__}: is improvement computed")

        if len(new_rule_indices) >= self.max_features:
            logging.info(f"{BinaryFeatureClassifier.__name__}: max features reached")
            return self._get_optimal_indices(new_rule_indices, is_improvement)

        if self._test_earlystopping(is_improvement):
            logging.info(f"{BinaryFeatureClassifier.__name__}: earlystopping criterion reached")
            return self._get_optimal_indices(new_rule_indices, is_improvement)

        return self._recursively_select_rules(encoded_train_data, encoded_val_data,
                                              prev_rule_indices=new_rule_indices,
                                              prev_train_predictions=new_train_predictions,
                                              prev_val_predictions=new_val_predictions,
                                              prev_val_scores=new_val_scores,
                                              cores_for_training=cores_for_training)

    def _test_earlystopping(self, is_improvement):
        if self.learn_all:
            return False

        # patience has not reached yet, continue training
        if len(is_improvement) < self.patience:
            return False

        # last few trees did not improve, stop training
        if not any(is_improvement[-self.patience:]):
            return True

        return False

    def _test_is_improvement(self, scores, min_delta):
        if len(scores) == 0:
            return []

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
        if self.learn_all:
            return rule_indices
        else:
            if len(rule_indices) == 0:
                return []

            optimal_tree_idx = max([i if is_improvement[i] else -1 for i in range(len(is_improvement))])

            return rule_indices[:optimal_tree_idx + 1]

    def _add_next_best_rule(self, encoded_train_data, prev_rule_indices, prev_predictions, cores_for_training):
        logging.info(f"{BinaryFeatureClassifier.__name__}: getting unused indices")
        unused_indices = self._get_unused_rule_indices(encoded_train_data, prev_rule_indices)
        logging.info(f"{BinaryFeatureClassifier.__name__}: unused indices gotten")

        if len(unused_indices) == 0:
            return prev_rule_indices, prev_predictions

        # prev_train_performance = self._test_performance_predictions(encoded_train_data, pred=prev_predictions)

        logging.info(f"{BinaryFeatureClassifier.__name__}: testing prev train performance")
        prev_train_performance = self._test_performance_predictions(encoded_train_data, pred=prev_predictions)
        logging.info(f"{BinaryFeatureClassifier.__name__}: prev train performance tested")

        logging.info(f"{BinaryFeatureClassifier.__name__}: testing new train performances")
        # new_training_performances = self._get_new_performances(encoded_train_data, prev_predictions=prev_predictions, new_indices_to_test=unused_indices)

        new_training_performances = self._test_new_train_performances(encoded_train_data, prev_predictions,
                                                                      unused_indices, prev_train_performance, cores_for_training)

        logging.info(f"{BinaryFeatureClassifier.__name__}: new train performances tested")

        logging.info(f"{BinaryFeatureClassifier.__name__}: getting best index")
        best_new_performance = max(new_training_performances)
        best_new_index = unused_indices[new_training_performances.index(best_new_performance)]
        logging.info(f"{BinaryFeatureClassifier.__name__}: best index gotten")

        if best_new_performance > prev_train_performance or self.learn_all:
            new_rule_indices = prev_rule_indices + [best_new_index]
            logging.info(f"{BinaryFeatureClassifier.__name__}: getting new train predictions")
            new_predictions = np.logical_or(prev_predictions, encoded_train_data.examples[:, best_new_index])
            logging.info(f"{BinaryFeatureClassifier.__name__}: new train predictions gotten")

            return new_rule_indices, new_predictions
        else:
            return prev_rule_indices, prev_predictions

    def _test_new_train_performances(self, encoded_train_data, prev_predictions, unused_indices,
                                     prev_train_performance, cores_for_training):
        y_true_train = Util.map_to_new_class_values(encoded_train_data.labels[self.label.name], self.class_mapping)
        optimization_scoring_fn = self._get_optimization_scoring_fn()

        example_weights = encoded_train_data.example_weights

        with Pool(cores_for_training) as pool:
            partial_func = partial(self._apply_optimization_fn_to_new_rule_combo,
                                   optimization_scoring_fn=optimization_scoring_fn, y_true_train=y_true_train,
                                   example_weights=example_weights, prev_predictions=prev_predictions,
                                   prev_train_performance=prev_train_performance)
            scores = pool.map(partial_func, encoded_train_data.examples[:, unused_indices].T)
        return scores

    def _apply_optimization_fn_to_new_rule_combo(self, new_rule_predictions, optimization_scoring_fn,
                                                 y_true_train, example_weights,
                                                 prev_predictions, prev_train_performance):
        new_predictions = np.logical_or(prev_predictions, new_rule_predictions)

        if np.array_equal(new_predictions, prev_predictions):
            logging.info("eq")
            return prev_train_performance
        else:
            return optimization_scoring_fn(y_true=y_true_train,
                                           y_pred=new_predictions,
                                           sample_weight=example_weights)


    # def _apply_optimization_fn_to_new_rule_combo(self, optimization_scoring_fn, examples, y_true_train,
    #                                              example_weights, prev_predictions, new_rule_idx):
    #     return optimization_scoring_fn(y_true=y_true_train,
    #                                    y_pred=np.logical_or(prev_predictions, examples[:, new_rule_idx]),
    #                                    sample_weight=example_weights)
    #

    def _get_unused_rule_indices(self, encoded_train_data, rule_indices):
        return [idx for idx in range(encoded_train_data.examples.shape[1]) if idx not in rule_indices]

    def _test_performance_predictions(self, encoded_data, pred):
        y_true = Util.map_to_new_class_values(encoded_data.labels[self.label.name], self.class_mapping)
        optimization_scoring_fn = self._get_optimization_scoring_fn()

        return optimization_scoring_fn(y_true=y_true, y_pred=pred, sample_weight=encoded_data.example_weights)


    def _get_new_performances(self, encoded_data, prev_predictions, new_indices_to_test):
        return [self._test_performance_predictions(encoded_data=encoded_data,
                                                   pred=np.logical_or(prev_predictions, encoded_data.examples[:, idx]))
                for idx in new_indices_to_test]


    # def _get_new_performances(self, encoded_data, prev_rule_indices, new_indices_to_test):
    #     return [self._test_performance_rule_tree(encoded_data, rule_indices=prev_rule_indices + [idx]) for idx in new_indices_to_test]

    def _test_performance_rule_tree(self, encoded_data, rule_indices):
        pred = self._get_rule_tree_predictions_bool(encoded_data, rule_indices)
        return self._test_performance_predictions(encoded_data, pred=pred)

    def _get_rule_tree_predictions_bool(self, encoded_data, rule_indices):
        return np.logical_or.reduce([encoded_data.examples[:, i] for i in rule_indices])

    def _get_rule_tree_predictions_class(self, encoded_data, rule_indices):
        y = self._get_rule_tree_predictions_bool(encoded_data, rule_indices).astype(int)
        return Util.map_to_old_class_values(y, self.class_mapping)

    def _check_features(self, encoded_data_features):
        if self.feature_names != encoded_data_features:
            mssg = f"{BinaryFeatureClassifier.__name__}: features during evaluation did not match the features set during fitting."

            logging.info(mssg + f"\n\nEvaluation features: {encoded_data_features}\nFitting features: {self.feature_names}")
            raise ValueError(mssg + " See the log file for more info.")

    def _export_selected_features(self, path, rule_tree_features):
        if path is not None:
            PathBuilder.build(path)
            with open(path / "selected_features.txt", "w") as file:
                file.writelines([f"{feature}\n" for feature in rule_tree_features])

    def fit_by_cross_validation(self, encoded_data: EncodedData, label: Label = None, optimization_metric: str = None,
                                number_of_splits: int = 5, cores_for_training: int = -1):
        logging.warning(f"{BinaryFeatureClassifier.__name__}: cross_validation is not implemented for this method. Using standard fitting instead...")
        self.fit(encoded_data=encoded_data, label=label)

    def _prepare_and_split_data(self, encoded_data: EncodedData):
        train_indices, val_indices = Util.get_train_val_indices(len(encoded_data.example_ids), self.training_percentage, random_seed=self.random_seed)

        self.train_indices = train_indices
        self.val_indices = val_indices

        train_data = Util.subset_encoded_data(encoded_data, train_indices)
        val_data = Util.subset_encoded_data(encoded_data, val_indices)

        return train_data, val_data

    def store(self, path: Path, feature_names=None, details_path: Path = None):
        PathBuilder.build(path)

        custom_vars = copy.deepcopy(vars(self))
        del custom_vars["result_path"]

        if self.label:
            custom_vars["label"] = {key.lstrip("_"): value for key, value in vars(self.label).items()}

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
        from immuneML.encodings.motif_encoding.SimilarToPositiveSequenceEncoder import SimilarToPositiveSequenceEncoder
        return [MotifEncoder, SimilarToPositiveSequenceEncoder]




