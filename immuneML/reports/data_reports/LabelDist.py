from pathlib import Path

import math
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.reports.PlotlyUtil import PlotlyUtil
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


class LabelDist(DataReport):
    """
    LabelDist report plots the distribution of label values for all labels provided as
    input to the report.

    Specification arguments:

    - labels (list): list of label names as they appear in the metadata file (RepertoireDataset)
      or in data files (Receptor/SequenceDataset).

    YAML specification:

    .. code-block: yaml

        reports:
            label_count_report:
                LabelCount:
                    labels: ['diagnosis', 'age_group', 'batch']
    """

    def __init__(self, dataset: Dataset = None, result_path: Path = None, name: str = None,
                 number_of_processes: int = 1, labels: list = None):
        super().__init__(dataset=dataset, result_path=result_path, name=name, number_of_processes=number_of_processes)
        self.labels = labels

    @classmethod
    def build_object(cls, **kwargs):
        ParameterValidator.assert_keys(list(kwargs.keys()), ["labels", 'name'], "LabelDist report", "LabelCount")
        ParameterValidator.assert_all_type_and_value(kwargs["labels"], str, "LabelDist", "labels")
        return cls(name=kwargs["name"], labels=kwargs["labels"])

    def _generate(self) -> ReportResult:

        df = self.dataset.get_metadata(self.labels, return_df=True)

        n_cols = 2
        n_rows = math.ceil(len(df.columns) / n_cols)

        fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=df.columns)

        colors = px.colors.qualitative.Plotly * math.ceil(len(df.columns) / len(px.colors.qualitative.Plotly))

        for i, col in enumerate(df.columns):
            row = i // n_cols + 1
            col_pos = i % n_cols + 1
            color = colors[i]

            if pd.api.types.is_numeric_dtype(df[col]):
                trace = go.Histogram(x=df[col], name=col, marker_color=color)
            else:
                counts = df[col].value_counts()
                trace = go.Bar(x=counts.index.astype(str), y=counts.values, name=col, marker_color=color)

            fig.add_trace(trace, row=row, col=col_pos)

        fig.update_layout(template='plotly_white', height=300 * n_rows, showlegend=False,
                          title_text="Label distributions")

        path = PathBuilder.build(self.result_path) / f"{self.name}_label_distributions.html"
        PlotlyUtil.write_image_to_file(fig, path, df.shape[0])

        return ReportResult(name=self.name, info='Label distributions',
                            output_figures=[ReportOutput(name='Label distributions', path=path)])

