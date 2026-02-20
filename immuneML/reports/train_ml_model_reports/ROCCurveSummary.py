import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import roc_curve

from immuneML.environment.Constants import Constants
from immuneML.environment.Label import Label
from immuneML.hyperparameter_optimization.states.HPItem import HPItem
from immuneML.ml_metrics.ml_metrics import roc_auc_score
from immuneML.reports.PlotlyUtil import PlotlyUtil
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


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            reports:
                my_roc_summary_report: ROCCurveSummary

    """

    @classmethod
    def build_object(cls, **kwargs):
        return ROCCurveSummary(**kwargs)

    def _generate(self) -> ReportResult:
        report_result = ReportResult(name=self.name,
                                     info="Plots ROC curves for all trained ML settings ([preprocessing], encoding, "
                                          "ML model) in the outer loop of cross-validation in the TrainMLModel "
                                          "instruction")

        PathBuilder.build(self.result_path)

        for label in self.state.label_configuration.get_label_objects():
            if len(label.values) != 2:
                logging.warning(f"{ROCCurveSummary.__name__}: report {self.name} is skipping label {label.name} "
                                f"as it has {len(label.values)} classes, while this report expects 2 classes.")
            elif label.positive_class is None:
                logging.warning(f"{ROCCurveSummary.__name__}: report {self.name} is skipping label {label.name} "
                                f"because 'positive_class' parameter is not set.")
            else:
                for index in range(self.state.assessment.split_count):
                    figure = self._create_figure_for_assessment_split(index, label)
                    report_result.output_figures.append(figure)

                best_model_figure = self._create_figure_for_best_model_across_splits(label)
                if best_model_figure is not None:
                    report_result.output_figures.append(best_model_figure)

        return report_result

    def _create_figure_for_assessment_split(self, index, label: Label):
        data = []
        for hp_item_name, hp_item in self.state.assessment_states[index].label_states[label.name].assessment_items.items():
            data.append(self._make_roc_curve(hp_item, label, f"{label.name}_{label.positive_class}_proba"))

        figure = self._draw_rocs(data=data, roc_legends=[f"{item['HPItem']} (AUROC = {round(item['AUC'], 2)})" for item in data],
                                 figure_name=f"ROC curves for label {label.name} on assessment split {index + 1}.html")
        return figure

    def _make_roc_curve(self, hp_item: HPItem, label: Label, proba_name: str) -> dict:
        df = pd.read_csv(hp_item.test_predictions_path)

        label_mapping = {label.positive_class: 1, label.get_binary_negative_class(): 0}

        true_y = [label_mapping[val] for val in df[f"{label.name}_true_class"].values]

        if hp_item.method.can_predict_proba():
            predicted_y = df[proba_name].values
        else:
            predicted_y = [label_mapping[val] for val in df[f"{label.name}_predicted_class"].values]

        fpr, tpr, _ = roc_curve(y_true=true_y, y_score=predicted_y)

        return {
            "FPR": fpr,
            "TPR": tpr,
            "AUC": roc_auc_score(true_y=true_y, predicted_y=predicted_y),
            "HPItem": str(hp_item.hp_setting)
        }

    def _draw_rocs(self, data: list, roc_legends: list, figure_name: str) -> ReportOutput:
        figure = go.Figure()

        figure.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='baseline (AUC = 0.5)',
                       line=dict(color=Constants.PLOTLY_BLACK, dash='dash'), hoverinfo="skip"))

        for index, item in enumerate(data):
            figure.add_trace(
                go.Scatter(x=item["FPR"], y=item["TPR"], mode='lines', name=roc_legends[index],
                           marker=dict(line=dict(width=3)),
                           hovertemplate="%{name}<extra></extra>"))

        figure.update_layout(template='plotly_white', xaxis_title='false positive rate',
                             yaxis_title='true positive rate')
        figure.update_layout(legend=dict(yanchor="bottom", y=0.06, xanchor="right", x=0.99), font_size=15)

        file_path = self.result_path / figure_name.replace(" ", "_")
        file_path = PlotlyUtil.write_image_to_file(figure, file_path)

        return ReportOutput(path=file_path, name=figure_name.split(".")[0])

    def _create_figure_for_best_model_across_splits(self, label: Label):
        """Create a plot showing ROC curves for the best model across all assessment splits"""
        if label.name not in self.state.optimal_hp_items:
            logging.warning(f"{ROCCurveSummary.__name__}: report {self.name} could not find optimal HP item "
                            f"for label {label.name}.")
            return None

        optimal_hp_setting = self.state.optimal_hp_items[label.name]
        data = []

        for index in range(self.state.assessment.split_count):
            hp_item = self.state.assessment_states[index].label_states[label.name].assessment_items[optimal_hp_setting.hp_setting.get_key()]
            roc_data = self._make_roc_curve(hp_item, label, f"{label.name}_{label.positive_class}_proba")
            roc_data['split'] = index + 1
            data.append(roc_data)

        if not data:
            logging.warning(f"{ROCCurveSummary.__name__}: report {self.name} could not find ROC data for "
                            f"optimal model {optimal_hp_setting.hp_setting.get_key()} for label {label.name}.")
            return None

        figure = self._draw_mean_roc_with_std(
            data=data,
            optimal_hp_setting=optimal_hp_setting,
            figure_name=f"Mean ROC curve for selected optimal model ({optimal_hp_setting.hp_setting.get_key()}) across splits for label {label.name}.html"
        )

        return figure

    def _draw_mean_roc_with_std(self, data: list, optimal_hp_setting, figure_name: str) -> ReportOutput:
        """Draw mean ROC curve with standard deviation bands across splits"""
        figure = go.Figure()

        # Add baseline
        figure.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='baseline (AUC = 0.5)',
                       line=dict(color=Constants.PLOTLY_BLACK, dash='dash'), hoverinfo="skip"))

        # Interpolate all curves to common FPR points
        mean_fpr = np.linspace(0, 1, 100)
        interp_tprs = []
        aucs = []

        for item in data:
            interp_tpr = np.interp(mean_fpr, item["FPR"], item["TPR"])
            interp_tpr[0] = 0.0
            interp_tprs.append(interp_tpr)
            aucs.append(item["AUC"])

        # Calculate mean and std
        mean_tpr = np.mean(interp_tprs, axis=0)
        mean_tpr[-1] = 1.0
        std_tpr = np.std(interp_tprs, axis=0)

        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)

        # Plot individual fold curves with low opacity
        for idx, item in enumerate(data):
            figure.add_trace(
                go.Scatter(x=item["FPR"], y=item["TPR"], mode='lines',
                           name=f"Split {item['split']}",
                           line=dict(width=1),
                           opacity=0.3,
                           showlegend=True,
                           hovertemplate=f"Split {item['split']}<br>AUROC = {round(item['AUC'], 2)}<extra></extra>"))

        # Calculate upper and lower bounds
        tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
        tpr_lower = mean_tpr - std_tpr

        # Add shaded std area
        figure.add_trace(
            go.Scatter(x=np.concatenate([mean_fpr, mean_fpr[::-1]]),
                       y=np.concatenate([tpr_upper, tpr_lower[::-1]]),
                       fill='toself',
                       fillcolor='rgba(128, 128, 128, 0.3)',
                       line=dict(color='rgba(255, 255, 255, 0)'),
                       showlegend=True,
                       name='±1 std. dev.',
                       hoverinfo='skip'))

        # Add mean ROC curve
        figure.add_trace(
            go.Scatter(x=mean_fpr, y=mean_tpr, mode='lines',
                       name=f'Mean ROC (AUC = {round(mean_auc, 2)} ± {round(std_auc, 2)})',
                       line=dict(width=3, color='blue'),
                       hovertemplate=f"Mean ROC<br>AUROC = {round(mean_auc, 2)} ± {round(std_auc, 2)}<extra></extra>"))

        figure.update_layout(template='plotly_white', xaxis_title='false positive rate',
                             yaxis_title='true positive rate',
                             title=f'Selected optimal model: {optimal_hp_setting.hp_setting.get_key()}')
        figure.update_layout(legend=dict(yanchor="bottom", y=0.06, xanchor="right", x=0.99), font_size=15)

        file_path = self.result_path / figure_name.replace(" ", "_")
        file_path = PlotlyUtil.write_image_to_file(figure, file_path)

        return ReportOutput(path=file_path, name=figure_name.split(".")[0])
