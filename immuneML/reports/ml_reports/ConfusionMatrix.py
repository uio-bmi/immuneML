import logging
from pathlib import Path

import pandas as pd
import plotly.figure_factory as ff
import yaml
from sklearn.metrics import confusion_matrix

from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.ml_methods.classifiers.MLMethod import MLMethod
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.ml_reports.MLReport import MLReport
from immuneML.util.PathBuilder import PathBuilder


class ConfusionMatrix(MLReport):
    """
    A report that plots the confusion matrix for a trained ML method.
    Supports both binary and multiclass classification.

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
                 result_path: Path = None, name: str = None, hp_setting=None, label=None, number_of_processes: int = 1):
        super().__init__(train_dataset=train_dataset, test_dataset=test_dataset, method=method,
                         result_path=result_path, name=name, hp_setting=hp_setting, label=label,
                         number_of_processes=number_of_processes)

    def _generate(self):
        PathBuilder.build(self.result_path)

        y_true = self.test_dataset.encoded_data.labels[self.label.name]
        y_pred = self.method.predict(self.test_dataset.encoded_data, self.label)[self.label.name]

        labels = self.label.values
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        cm_df = pd.DataFrame(cm, index=labels, columns=labels)

        table_path = self.result_path / "confusion_matrix.csv"
        cm_df.to_csv(table_path)

        heatmap_path = self._plot_confusion_matrix(cm_df)

        self._write_settings()

        return ReportResult(self.name,
                            info=f"Confusion matrix for {type(self.method).__name__}",
                            output_tables=[ReportOutput(table_path, "Confusion matrix CSV")],
                            output_figures=[ReportOutput(heatmap_path)])

    def _plot_confusion_matrix(self, cm_df: pd.DataFrame):
        fig = ff.create_annotated_heatmap(
            z=cm_df.values,
            x=cm_df.columns.tolist(),
            y=cm_df.index.tolist(),
            colorscale="Viridis",
            showscale=True
        )
        fig.update_layout(title_text="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual",
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
