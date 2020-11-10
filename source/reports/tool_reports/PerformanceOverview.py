import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn import metrics
from sklearn.metrics import precision_recall_curve

from source.hyperparameter_optimization.states.TrainMLModelState import TrainMLModelState
from source.reports.Report import Report
from source.reports.ReportOutput import ReportOutput
from source.reports.ReportResult import ReportResult
from source.util.PathBuilder import PathBuilder


class PerformanceOverview(Report):
    """
    PerformanceOverview report creates AUC plot and precision-recall plot for optimal trained models on multiple datasets. The labels on the plots
    are the names of the datasets, so it might be good to have user-friendly names when defining datasets that are still a combination of
    letters, numbers and the underscore sign.

    This plot can be used only with MultiDatasetBenchmarkTool as it will plot AUC and PR curve for trained models across datasets.

    If datasets have the same number of examples, the baseline PR curve will be plotted as described in this publication:
    Saito T, Rehmsmeier M. The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets.
    PLOS ONE. 2015;10(3):e0118432. doi:10.1371/journal.pone.0118432

    If the datasets have different number of examples, the baseline PR curve will not be plotted.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        reports:
            my_performance_report:
                PerformanceOverview:
                    positive_class: True

    """

    @classmethod
    def build_object(cls, **kwargs):
        return PerformanceOverview(**kwargs)

    def __init__(self, instruction_states: List[TrainMLModelState] = None, name: str = None, result_path: str = None, positive_class=None):
        super().__init__(name)
        self.instruction_states = instruction_states
        self.result_path = result_path
        self.positive_class = positive_class

    def generate(self) -> ReportResult:

        self.result_path = PathBuilder.build(self.result_path + f'{self.name}/')

        assert all(self.instruction_states[0].label_configuration.get_labels_by_name() == state.label_configuration.get_labels_by_name() and
                   self.instruction_states[0].label_configuration.get_label_values(
                       self.instruction_states[0].label_configuration.get_labels_by_name()[0]) ==
                   state.label_configuration.get_label_values(state.label_configuration.get_labels_by_name()[0])
                   for state in self.instruction_states), \
            "PerformanceOverview: there is a difference in labels between instructions, the plots cannot be created."
        assert len(self.instruction_states[0].label_configuration.get_labels_by_name()) == 1, \
            'PerformanceOverview: multiple labels were provided, but only one can be used in this report.'

        label = self.instruction_states[0].label_configuration.get_labels_by_name()[0]

        optimal_hp_items = [list(state.optimal_hp_items.values())[0] for state in self.instruction_states]

        colors = ["#332288", "#882255", "#88CCEE", "#DDCC77", "#CC6677", "#AA4499", "#117733", "#44AA99"]
        figure_auc, table_aucs = self.plot_roc(optimal_hp_items, label, colors)
        figure_pr, table_pr = self.plot_precision_recall(optimal_hp_items, label, colors)

        return ReportResult(output_figures=[figure_auc, figure_pr], output_tables=table_aucs + table_pr)

    def plot_roc(self, optimal_hp_items, label, colors) -> Tuple[ReportOutput, List[ReportOutput]]:
        report_data_outputs = []
        figure = go.Figure()

        figure.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='baseline', line=dict(color='black', dash='dash'), hoverinfo="skip"))

        for index, item in enumerate(optimal_hp_items):
            if item.test_predictions_path is None:
                logging.warning(f'{PerformanceOverview.__name__}: there are no test predictions for dataset '
                                f'{self.instruction_states[index].dataset.name}, skipping this dataset when generating performance overview...')
                continue

            df = pd.read_csv(item.test_predictions_path)
            true_class = df[f"{label}_true_class"].values
            predicted_class = df[f"{label}_predicted_class"].values
            fpr, tpr, _ = metrics.roc_curve(y_true=true_class, y_score=predicted_class)
            auc = metrics.roc_auc_score(true_class, predicted_class)
            name = self.instruction_states[index].dataset.name + f' (AUC = {round(auc, 2)})'
            figure.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=name, marker=dict(color=colors[index]), hoverinfo="skip"))

            data_path = self.result_path + f"roc_curve_data_{name}.csv"
            pd.DataFrame({"FPR": fpr, "TPR": tpr}).to_csv(data_path, index=False)
            report_data_outputs.append(ReportOutput(data_path, f'ROC curve data for dataset {name} (csv)'))

        figure_path = self.result_path + "roc_curve.html"
        figure.update_layout(template='plotly_white', xaxis_title='false positive rate', yaxis_title='true positive rate')
        figure.write_html(figure_path)

        return ReportOutput(figure_path, 'ROC curve'), report_data_outputs

    def plot_precision_recall(self, optimal_hp_items: list, label, colors):
        report_data_outputs = []
        figure = go.Figure()

        test_labels = np.array(optimal_hp_items[0].test_dataset.get_metadata([label])[label])
        y = test_labels[test_labels == self.positive_class].shape[0] / len(test_labels)

        add_baseline = True

        for index, item in enumerate(optimal_hp_items):
            df = pd.read_csv(item.test_predictions_path)

            if df.shape[0] != len(test_labels):
                add_baseline = False

            true_class = df[f"{label}_true_class"].values
            predicted_proba = df[f"{label}_predicted_class"].values
            precision, recall, _ = precision_recall_curve(y_true=true_class, probas_pred=predicted_proba)
            name = self.instruction_states[index].dataset.name
            figure.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=name, marker=dict(color=colors[index]), hoverinfo="skip"))

            data_path = self.result_path + f"precision_recall_data_{name}.csv"
            pd.DataFrame({"precision": precision, "recall": recall}).to_csv(data_path, index=False)
            report_data_outputs.append(ReportOutput(data_path, f'precision-recall curve data for dataset {name}'))

        if add_baseline:
            figure.add_trace(go.Scatter(x=[0, 1], y=[y, y], mode='lines', name='baseline', line=dict(color='black', dash='dash'), hoverinfo="skip"))

        figure_path = self.result_path + "precision_recall_curve.html"
        figure.update_layout(template='plotly_white', xaxis_title="recall", yaxis_title="precision")
        figure.write_html(figure_path)

        return ReportOutput(figure_path, 'precision-recall curve'), report_data_outputs
