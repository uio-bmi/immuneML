import warnings

import numpy as np
import plotly.graph_objs as go
from sklearn.metrics import roc_curve, auc

from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.ml_reports.MLReport import MLReport
from immuneML.util.PathBuilder import PathBuilder


class ROCCurve(MLReport):
    """
    A report that plots the ROC curve for a binary classifier.


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        reports:
          my_roc_report: ROCCurve

    """

    def __init__(self, name: str = None):
        super().__init__(name=name)

    @classmethod
    def build_object(cls, **kwargs):
        name = kwargs["name"] if "name" in kwargs else "ROC_curve"
        return ROCCurve(name=name)

    def _generate(self) -> ReportResult:
        x = self.test_dataset.encoded_data
        y_score = self.method.predict_proba(x, self.label)[self.label]
        fpr, tpr, _ = roc_curve(x.labels[self.label], y_score[:, 0])
        roc_auc = auc(fpr, tpr)

        trace1 = go.Scatter(x=fpr, y=tpr,
                            mode='lines',
                            line=dict(color='darkorange', width=2),
                            name=f"ROC curve (area = {roc_auc})")
        trace2 = go.Scatter(x=[0, 1], y=[0, 1],
                            mode='lines',
                            line=dict(color='navy', width=2, dash='dash'),
                            showlegend=False)
        layout = go.Layout(title='Receiver operating characteristic example',
                           xaxis=dict(title='False Positive Rate'),
                           yaxis=dict(title='True Positive Rate'))

        fig = go.Figure(data=[trace1, trace2], layout=layout)

        PathBuilder.build(self.result_path)
        path_htm = self.result_path / f"{self.name}.html"
        path_csv = self.result_path / f"{self.name}.csv"
        csv_result = np.concatenate((fpr.reshape(1, -1), tpr.reshape(1, -1)))
        fig.write_html(str(path_htm))
        np.savetxt(str(path_csv), csv_result, header="fpr,tpr")
        return ReportResult(self.name,
                            output_figures=[ReportOutput(path_htm)],
                            output_tables=[ReportOutput(path_csv)])

    def check_prerequisites(self):
        if not hasattr(self, "result_path") or self.result_path is None:
            warnings.warn(f"{self.__class__.__name__} requires an output"
                          f" 'path' to be set. {self.__class__.__name__}"
                          f" report will not be created.")
            return False

        if self.test_dataset.encoded_data is None:
            warnings.warn(
                f"{self.__class__.__name__}: test dataset is"
                f" not encoded and can not be run."
                f"{self.__class__.__name__} report will not be created.")
            return False

        if self.method is None:
            warnings.warn(
                f"{self.__class__.__name__}: method is"
                f" not defined and can not be run."
                f"{self.__class__.__name__} report will not be created.")
        return True
