import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

from sklearn.metrics import precision_score, recall_score, accuracy_score, balanced_accuracy_score

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.encodings.motif_encoding.MotifEncoder import MotifEncoder
from immuneML.hyperparameter_optimization.HPSetting import HPSetting
from immuneML.ml_methods.BinaryFeatureClassifier import BinaryFeatureClassifier
from immuneML.ml_methods.LogisticRegression import LogisticRegression
from immuneML.ml_methods.MLMethod import MLMethod
from immuneML.ml_methods.RandomForestClassifier import RandomForestClassifier
from immuneML.ml_methods.SVC import SVC
from immuneML.ml_methods.SVM import SVM
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.ml_reports.MLReport import MLReport
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder

from immuneML.ml_methods.util.Util import Util



class BinaryFeaturePrecisionRecall(MLReport):
    """


    Arguments:



    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_report: BinaryFeaturePrecisionRecall
    """

    @classmethod
    def build_object(cls, **kwargs):

        return BinaryFeaturePrecisionRecall(**kwargs)

    def __init__(self, train_dataset: Dataset = None, test_dataset: Dataset = None,
                 method: BinaryFeatureClassifier = None, result_path: Path = None, name: str = None, hp_setting: HPSetting = None,
                 label=None, number_of_processes: int = 1):
        super().__init__(train_dataset=train_dataset, test_dataset=test_dataset, method=method, result_path=result_path,
                         name=name, hp_setting=hp_setting, label=label, number_of_processes=number_of_processes)

    def _generate(self):
        PathBuilder.build(self.result_path)

        plotting_data_train = self._compute_plotting_data(self.train_dataset.encoded_data)
        plotting_data_test = self._compute_plotting_data(self.test_dataset.encoded_data)


        train_table = self._write_output_table(plotting_data_train, self.result_path / "training_performance.tsv", name="Training set performance of every subset of binary features")
        test_table = self._write_output_table(plotting_data_test, self.result_path / "test_performance.tsv", name="Test set performance of every subset of binary features")

        train_fig = self._plot(plotting_data_train, dataset_type="training")
        test_fig = self._plot(plotting_data_test, dataset_type="test")

        return ReportResult(self.name,
                            info="Precision and recall scores for each subset of learned binary motifs",
                            output_tables=[table for table in [train_table, test_table] if table is not None],
                            output_figures=[fig for fig in [train_fig, test_fig] if fig is not None])

    def _compute_plotting_data(self, encoded_data):
        rule_tree_indices = self.method.rule_tree_indices

        data = {"n_rules": [],
                "precision": [],
                "recall": [],
                "accuracy": [],
                "balanced_accuracy": []}

        y_true_bool = np.array([cls == self.label.positive_class for cls in encoded_data.labels[self.label.name]])

        for n_rules in range(1, len(rule_tree_indices) + 1):
            rule_subtree = rule_tree_indices[:n_rules]

            y_pred_bool = self.method._get_rule_tree_predictions_bool(encoded_data, rule_subtree)

            data["n_rules"].append(n_rules)
            data["precision"].append(precision_score(y_true_bool, y_pred_bool))
            data["recall"].append(recall_score(y_true_bool, y_pred_bool))
            data["accuracy"].append(accuracy_score(y_true_bool, y_pred_bool))
            data["balanced_accuracy"].append(balanced_accuracy_score(y_true_bool, y_pred_bool))

        return pd.DataFrame(data)

    def _plot(self, plotting_data, dataset_type):
        fig = px.line(plotting_data, x="recall", y="precision",
                      range_x=[0, 1.01], range_y=[0, 1.01],
                      template="plotly_white",
                      hover_data=["n_rules"],
                      color_discrete_sequence=px.colors.diverging.Tealrose,
                      markers=True)

        file_path = self.result_path / f"{dataset_type}_precision_recall.html"

        fig.write_html(str(file_path))

        return ReportOutput(path=file_path,
                            name=f"Precision and recall scores on the {dataset_type} set for motif subsets")

    def check_prerequisites(self):
        location = BinaryFeaturePrecisionRecall.__name__

        run_report = True

        if not isinstance(self.method, BinaryFeatureClassifier):
            logging.warning(f"{location} report can only be created for {BinaryFeatureClassifier.__name__}, but got "
                            f"{type(self.method).__name__} instead. {location} report will not be created.")
            run_report = False

        if self.train_dataset.encoded_data is None or self.train_dataset.encoded_data.examples is None or self.train_dataset.encoded_data.feature_names is None or self.train_dataset.encoded_data.encoding != MotifEncoder.__name__:
            warnings.warn(
                f"{location}: this report can only be created for a dataset encoded with the {MotifEncoder.__name__}. Report {self.name} will not be created.")
            run_report = False

        if self.method.keep_all:
            warnings.warn(f"{location}: motifs will not be displayed in the correct order due to using the parameter "
                          f"keep_all = True for ML method {self.method.name}. To learn all motifs and display performances "
                          f"in the correct order, use learn_all = True instead. ")

        return run_report
