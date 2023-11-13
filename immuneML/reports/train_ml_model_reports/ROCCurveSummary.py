import logging

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve

from immuneML.environment.Constants import Constants
from immuneML.environment.Label import Label
from immuneML.hyperparameter_optimization.states.HPItem import HPItem
from immuneML.ml_metrics.ml_metrics import roc_auc_score
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.train_ml_model_reports.TrainMLModelReport import TrainMLModelReport
from immuneML.util.PathBuilder import PathBuilder


class ROCCurveSummary(TrainMLModelReport):
    """
    This report plots ROC curves for all trained ML settings ([preprocessing], encoding, ML model) in the outer loop of cross-validation in the
    :ref:`TrainMLModel` instruction. If there are multiple splits in the outer loop, this report will make one plot per split. This report is
    defined only for binary classification. If there are multiple labels defined in the instruction, each label has to have two classes to be included
    in this report.

    Specification arguments: there are no arguments for this report.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

    reports:
        my_roc_summary_report: ROCCurveSummary

    """

    @classmethod
    def build_object(cls, **kwargs):
        return ROCCurveSummary(**kwargs)

    def _generate(self) -> ReportResult:
        report_result = ReportResult(name=self.name,
                                     info="Plots ROC curves for all trained ML settings ([preprocessing], encoding, ML model) in the outer loop of cross-validation in the TrainMLModel instruction")

        PathBuilder.build(self.result_path)

        for label in self.state.label_configuration.get_label_objects():
            if len(label.values) != 2:
                logging.warning(f"{ROCCurveSummary.__name__}: report {self.name} is skipping label {label.name} as it has {len(label.values)} "
                                f"classes, while this report expects 2 classes.")
            elif label.positive_class is None:
                logging.warning(f"{ROCCurveSummary.__name__}: report {self.name} is skipping label {label.name} because 'positive_class' parameter "
                                f"is not set.")
            else:
                for index in range(self.state.assessment.split_count):
                    figure = self._create_figure_for_assessment_split(index, label)
                    report_result.output_figures.append(figure)

        return report_result

    def _create_figure_for_assessment_split(self, index, label: Label):
        data = []
        for hp_item_name, hp_item in self.state.assessment_states[index].label_states[label.name].assessment_items.items():
            data.append(self._make_roc_curve(hp_item, label.name, f"{label.name}_{label.positive_class}_proba"))

        figure = self._draw_rocs(data=data, roc_legends=[f"{item['HPItem']} (AUROC = {round(item['AUC'], 2)})" for item in data],
                                 figure_name=f"ROC curves for label {label.name} on assessment split {index + 1}.html")
        return figure

    def _make_roc_curve(self, hp_item: HPItem, label_name: str, proba_name: str) -> dict:
        df = pd.read_csv(hp_item.test_predictions_path)

        true_y = df[f"{label_name}_true_class"].values

        if hp_item.method.can_predict_proba():
            predicted_y = df[proba_name].values
        else:
            predicted_y = df[f"{label_name}_predicted_class"].values

        fpr, tpr, _ = roc_curve(y_true=true_y, y_score=predicted_y)

        return {
            "FPR": fpr,
            "TPR": tpr,
            "AUC": roc_auc_score(true_y=true_y, predicted_y=predicted_y),
            "HPItem": str(hp_item.hp_setting)
        }

    def _draw_rocs(self, data: list, roc_legends: list, figure_name: str) -> ReportOutput:
        figure = go.Figure()
        colors = px.colors.sequential.Viridis[::2][::-1]

        figure.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='baseline (AUC = 0.5)', line=dict(color=Constants.PLOTLY_BLACK, dash='dash'),
                       hoverinfo="skip"))

        for index, item in enumerate(data):
            figure.add_trace(
                go.Scatter(x=item["FPR"], y=item["TPR"], mode='lines', name=roc_legends[index], marker=dict(color=colors[index], line=dict(width=3)),
                           hoverinfo="skip"))

        figure.update_layout(template='plotly_white', xaxis_title='false positive rate', yaxis_title='true positive rate')
        figure.update_layout(legend=dict(yanchor="bottom", y=0.06, xanchor="right", x=0.99), font_size=15)

        file_path = self.result_path / figure_name.replace(" ", "_")
        figure.write_html(str(file_path))

        return ReportOutput(path=file_path, name=figure_name.split(".")[0])
