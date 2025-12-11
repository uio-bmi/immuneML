import logging

import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import precision_recall_curve, average_precision_score

from immuneML.environment.Label import Label
from immuneML.hyperparameter_optimization.states.HPItem import HPItem
from immuneML.reports.PlotlyUtil import PlotlyUtil
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.train_ml_model_reports.TrainMLModelReport import TrainMLModelReport
from immuneML.util.PathBuilder import PathBuilder


class PrecisionRecallCurveSummary(TrainMLModelReport):
    """
    This report plots Precision-Recall curves for all trained ML settings ([preprocessing], encoding, ML model) in the outer loop of
    cross-validation in the :ref:`TrainMLModel` instruction. It also reports the average precision (AP) for each setting.
     If there are multiple splits in the outer loop, this report will make one
    plot per split. This report is defined only for binary classification.


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            reports:
                my_pr_summary_report: PrecisionRecallCurveSummary

    """

    @classmethod
    def build_object(cls, **kwargs):
        return PrecisionRecallCurveSummary(**kwargs)

    def _generate(self) -> ReportResult:
        report_result = ReportResult(name=self.name,
                                     info="Plots Precision-Recall curves for all trained ML settings ([preprocessing], "
                                          "encoding, ML model) in the outer loop of cross-validation in the TrainMLModel "
                                          "instruction and reports the average precision (AP) for each curve.",)

        PathBuilder.build(self.result_path)

        for label in self.state.label_configuration.get_label_objects():
            if len(label.values) != 2:
                logging.warning(
                    f"{PrecisionRecallCurveSummary.__name__}: report {self.name} is skipping label {label.name} "
                    f"as it has {len(label.values)} classes, while this report expects 2 classes.")
            elif label.positive_class is None:
                logging.warning(
                    f"{PrecisionRecallCurveSummary.__name__}: report {self.name} is skipping label {label.name} "
                    f"because 'positive_class' parameter is not set.")
            else:
                for index in range(self.state.assessment.split_count):
                    figure = self._create_figure_for_assessment_split(index, label)
                    report_result.output_figures.append(figure)

        return report_result

    def _create_figure_for_assessment_split(self, index, label: Label):
        data = []
        for hp_item_name, hp_item in self.state.assessment_states[index].label_states[
            label.name].assessment_items.items():
            data.append(self._make_pr_curve(hp_item, label, f"{label.name}_{label.positive_class}_proba"))

        figure = self._draw_pr_curves(data=data,
                                      pr_legends=[f"{item['HPItem']} (AP = {round(item['AP'], 2)})" for item in data],
                                      figure_name=f"Precision-Recall curves for label {label.name} on assessment split {index + 1}.html")
        return figure

    def _make_pr_curve(self, hp_item: HPItem, label: Label, proba_name: str) -> dict:
        df = pd.read_csv(hp_item.test_predictions_path)

        label_mapping = {label.positive_class: 1, label.get_binary_negative_class(): 0}

        true_y = [label_mapping[val] for val in df[f"{label.name}_true_class"].values]

        if hp_item.method.can_predict_proba():
            predicted_y = df[proba_name].values
        else:
            predicted_y = [label_mapping[val] for val in df[f"{label.name}_predicted_class"].values]

        precision, recall, _ = precision_recall_curve(y_true=true_y, y_score=predicted_y,
                                                      pos_label=label.positive_class)

        return {
            "Precision": precision,
            "Recall": recall,
            "AP": average_precision_score(y_true=true_y, y_score=predicted_y),
            "HPItem": str(hp_item.hp_setting)
        }

    def _draw_pr_curves(self, data: list, pr_legends: list, figure_name: str) -> ReportOutput:
        figure = go.Figure()

        for index, item in enumerate(data):
            figure.add_trace(
                go.Scatter(x=item["Recall"], y=item["Precision"], mode='lines', name=pr_legends[index],
                           marker=dict(line=dict(width=3)),
                           hovertemplate="%{name}<extra></extra>"))

        figure.update_layout(template='plotly_white', xaxis_title='recall',
                             yaxis_title='precision', legend=dict(yanchor="top", y=0.94, xanchor="right", x=0.99),
                             font_size=15)

        file_path = self.result_path / figure_name.replace(" ", "_")
        file_path = PlotlyUtil.write_image_to_file(figure, file_path)

        return ReportOutput(path=file_path, name=figure_name.split(".")[0])
