import warnings
from collections import Counter
from pathlib import Path
from typing import Union

import pandas as pd
import plotly.express as px

from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.environment.SequenceType import SequenceType
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.util.Logger import print_log
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


class DimensionalityReduction(DataReport):
    """
    This report visualizes the data obtained by dimensionality reduction.

    Specification arguments:

    - label (str): name of the label to use for highlighting data points

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        rep1:
            DimensionalityReduction:
                label: epitope

    """

    @classmethod
    def build_object(cls, **kwargs):
        return DimensionalityReduction(**{**kwargs})

    def __init__(self, dataset: Union[SequenceDataset] = None, batch_size: int = 1, result_path: Path = None,
                 name: str = None, label: str = None):
        super().__init__(dataset=dataset, result_path=result_path, name=name)
        self._label = label

    def check_prerequisites(self):
        if isinstance(self.dataset, SequenceDataset):
            return True
        else:
            warnings.warn(
                "DimensionalityReduction: report can be generated only from sequence datasets. Skipping this report...")
            return False

    def _generate(self) -> ReportResult:
        assert self.dataset.encoded_data.dimensionality_reduced_data is not None
        dim_reduced_data = self.dataset.encoded_data.dimensionality_reduced_data
        assert dim_reduced_data.shape[1] == 2

        try:
            data_labels = self.dataset.get_attribute(self._label)
        except AttributeError:
            warnings.warn(f"Label {self._label} not found in the dataset. Skipping label coloring in the plot.")

        PathBuilder.build(self.result_path)
        # Convering labels to list in case labels are strings
        df = pd.DataFrame(
            {"x": dim_reduced_data[:, 0], 'y': dim_reduced_data[:, 1], self._label: data_labels.tolist()})
        df.to_csv(self.result_path / 'dimensionality_reduced_data.csv', index=False)

        report_output_fig = self._safe_plot(df=df, output_written=False)
        output_figures = None if report_output_fig is None else [report_output_fig]

        return ReportResult(name=self.name,
                            info="A scatter plot of the dimensionality reduced data",
                            output_figures=output_figures,
                            #output_tables=[ReportOutput(self.result_path / 'sequence_length_distribution.csv',
                            #                            'lengths of sequences in the dataset')]
                            )

    def _plot(self, df: pd.DataFrame) -> ReportOutput:
        figure = px.scatter(df, x="x", y="y", color=self._label)
        figure.update_layout(title="Umap", template="plotly_white")
        #figure.update_traces(marker_color=px.colors.diverging.Tealrose[0])
        PathBuilder.build(self.result_path)

        file_path = self.result_path / "dimensionality_reduction.html"
        figure.write_html(str(file_path))
        return ReportOutput(path=file_path, name="Plot of dimensionality reduced data")
