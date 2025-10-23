import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix

from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.ml_methods.classifiers.MLMethod import MLMethod
from immuneML.reports.PlotlyUtil import PlotlyUtil
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.ml_reports.MLReport import MLReport
from immuneML.util.PathBuilder import PathBuilder


class ConfusionMatrix(MLReport):
    """
    A report that plots the confusion matrix for a trained ML method.
    Supports both binary and multiclass classification.

    **Specification arguments:**

    - alternative_label (str): optionally, the confusion matrix can be split between different values of an alternative label.
      This may be useful to compare performance across different data subsets (e.g., batches, sources).
      If specified, separate confusion matrices will be generated for each value of the alternative label. Default is None.

    Example output:

    .. image:: ../../_static/images/reports/confusion_matrix_example.png
       :alt: Confusion matrix report
       :width: 650

    **YAML specification:**

    .. code-block:: yaml

        definitions:
            reports:
                my_conf_mat_report: ConfusionMatrix

    """

    @classmethod
    def build_object(cls, **kwargs):
        return ConfusionMatrix(**kwargs)

    def __init__(self, train_dataset: Dataset = None, test_dataset: Dataset = None, method: MLMethod = None,
                 result_path: Path = None, name: str = None, hp_setting=None, label=None, number_of_processes: int = 1,
                 alternative_label: str = None):
        super().__init__(train_dataset=train_dataset, test_dataset=test_dataset, method=method,
                         result_path=result_path, name=name, hp_setting=hp_setting, label=label,
                         number_of_processes=number_of_processes)
        self.alternative_label = alternative_label

    def _generate(self):
        PathBuilder.build(self.result_path)

        y_true = np.array(self.test_dataset.encoded_data.labels[self.label.name])
        y_pred = self.method.predict(self.test_dataset.encoded_data, self.label)[self.label.name]

        labels = self.label.values
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        cm_df = pd.DataFrame(cm, index=labels, columns=labels)

        heatmap_path = self._plot_confusion_matrix(cm_df)

        table_path = self.result_path / "confusion_matrix.csv"
        cm_df.rename_axis('true/predicted').reset_index().to_csv(table_path, index=False)

        output_figures = [ReportOutput(heatmap_path)]

        self._write_settings()

        if self.alternative_label is not None:
            try:
                alt_label_filename = self._generate_for_alternative_label(y_true, y_pred)
                alt_label_output = ReportOutput(alt_label_filename, f"Confusion matrix split by {self.alternative_label}")
                output_figures.append(alt_label_output)
            except Exception as e:
                logging.warning(f"Could not generate confusion matrix for alternative label "
                                f"'{self.alternative_label}': {e}")

        return ReportResult(self.name,
                            info=f"Confusion matrix for {type(self.method).__name__}",
                            output_tables=[ReportOutput(table_path, "Confusion matrix CSV")],
                            output_figures=output_figures)

    def _generate_for_alternative_label(self, y_true, y_pred):
        alt_labels = self.test_dataset.get_metadata([self.alternative_label], return_df=True)
        alt_label_values = sorted(alt_labels[self.alternative_label].unique())

        fig = make_subplots(cols=2, rows=(len(alt_label_values) + 1) // 2, subplot_titles=alt_label_values,
                            vertical_spacing=0.1,
                            shared_xaxes=True, shared_yaxes=True, x_title="Predicted Label", y_title='True Label')
        subplot_index = 0

        for alt_label_value in alt_label_values:
            indices = (alt_labels[self.alternative_label] == alt_label_value).values.astype(bool)
            y_true_subset = y_true[indices]
            y_pred_subset = y_pred[indices]

            labels = self.label.values
            cm = confusion_matrix(y_true_subset, y_pred_subset, labels=labels)

            cm_df = pd.DataFrame(cm, index=labels, columns=labels)

            fig.add_trace(go.Heatmap(z=cm_df.values, texttemplate="%{text}", text=cm_df.values, colorscale='Viridis',
                                   hovertemplate="True value: %{y}<br>Predicted value: %{x}"
                                                 "<br>Count: %{z}<extra></extra>", showscale=False,
                                   x=[str(lbl) for lbl in cm_df.index.tolist()],
                                   y=[str(lbl) for lbl in cm_df.columns.tolist()]),
                          row=(subplot_index // 2) + 1, col=(subplot_index % 2) + 1)
            subplot_index += 1

            table_path = self.result_path / f"confusion_matrix_{alt_label_value}.csv"
            cm_df.rename_axis('true/predicted').reset_index().to_csv(table_path, index=False)

        fig.update_layout(title_text=f"Confusion matrix across {self.alternative_label} values",
                          template="plotly_white")
        filename = self.result_path / f"confusion_matrix_{self.alternative_label}.html"
        PlotlyUtil.write_image_to_file(fig, filename, self.test_dataset.get_example_count())
        return filename

    def _plot_confusion_matrix(self, cm_df: pd.DataFrame):

        fig = go.Figure(go.Heatmap(z=cm_df.values, texttemplate="%{text}", text=cm_df.values, colorscale='Viridis',
                                   hovertemplate="True value: %{y}<br>Predicted value: %{x}"
                                                 "<br>Count: %{z}<extra></extra>", showscale=False,
                                   x=[str(lbl) for lbl in cm_df.index.tolist()],
                                   y=[str(lbl) for lbl in cm_df.columns.tolist()]))

        fig.update_layout(title_text="Confusion Matrix", xaxis_title="Predicted class", yaxis_title="True class",
                          template="plotly_white")

        filename = self.result_path / "confusion_matrix.html"
        fig.write_html(str(filename))
        return filename

    def _write_settings(self):
        if self.hp_setting is not None:
            file_path = self.result_path / "settings.yaml"
            with file_path.open("w") as file:
                yaml.dump({"preprocessing": self.hp_setting.preproc_sequence_name,
                           "encoder": self.hp_setting.encoder_name,
                           "ml_method": self.hp_setting.ml_method_name},
                          file)

    def check_prerequisites(self):
        if self.test_dataset is None or self.label is None:
            logging.warning("ConfusionMatrixReport requires a test dataset and a specified label.")
            return False
        return True
