import logging
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


class LabelOverlap(DataReport):
    """
    This report creates a heatmap where the columns are the values of one label and rows are the values of another label,
    and the cells contain the number of samples that have both label values. It works for any dataset type.

    **Specification arguments:**

    - column_label (str): Name of the label to be used as columns in the heatmap.

    - row_label (str): Name of the label to be used as rows in the heatmap.

    **YAML specification:**

    .. code-block:: yaml

        my_data_report:
            LabelOverlap:
                column_label: epitope
                row_label: batch

    """

    def __init__(self, dataset: Dataset = None, result_path: Path = None, name: str = None,
                 number_of_processes: int = 1,
                 column_label: str = None, row_label: str = None):
        super().__init__(dataset, result_path, name, number_of_processes)
        self.column_label = column_label
        self.row_label = row_label

    @classmethod
    def build_object(cls, **kwargs):
        ParameterValidator.assert_keys_present(list(kwargs.keys()), ["column_label", "row_label"], "LabelOverlap",
                                               'LabelOverlap')
        ParameterValidator.assert_type_and_value(kwargs["column_label"], str, "LabelOverlap", "column_label")
        ParameterValidator.assert_type_and_value(kwargs["row_label"], str, "LabelOverlap", "row_label")
        return LabelOverlap(column_label=kwargs["column_label"], row_label=kwargs["row_label"])

    def check_prerequisites(self):
        if self.column_label not in self.dataset.get_label_names() or self.row_label not in self.dataset.get_label_names():
            logging.warning(f"One or both of the specified labels ({[self.column_label, self.row_label]}) do not exist in the dataset.")
            return False
        return True

    def _generate(self) -> ReportResult:
        # Get metadata for both labels
        metadata = self.dataset.get_metadata([self.column_label, self.row_label])

        # Create a cross-tabulation of the two labels
        overlap_matrix = pd.crosstab(metadata[self.row_label], metadata[self.column_label])

        # Save as CSV
        PathBuilder.build(self.result_path)
        csv_path = self.result_path / 'label_overlap.csv'
        overlap_matrix.to_csv(csv_path)

        # Create heatmap using plotly
        fig = go.Figure(data=go.Heatmap(
            z=overlap_matrix.values,
            x=overlap_matrix.columns,
            y=overlap_matrix.index,
            colorscale=[[0, '#e5f6f6'], [0.5, '#66b2b2'], [1, '#006666']],  # Custom teal colorscale
            text=overlap_matrix.values,
            texttemplate="%{text}",
            hovertemplate=f"{self.row_label}: " + "%{y}<br>" + f"{self.column_label}: "
                          + "%{x}<br>Count: %{z}<extra></extra>",
            textfont={"size": 14},
            showscale=False,  # Hide the color scale legend
            hoverongaps=False,
        ))

        # Update layout for better readability
        fig.update_layout(
            title=f"Label Overlap: {self.row_label} vs {self.column_label}",
            xaxis_title=self.column_label,
            yaxis_title=self.row_label,
            template="plotly_white",
            font=dict(size=12)
        )

        # Save plot
        plot_path = self.result_path / 'label_overlap.html'
        fig.write_html(str(plot_path))

        return ReportResult(
            name=self.name,
            info=f"Shows overlap between {self.row_label} and {self.column_label} labels.",
            output_figures=[ReportOutput(plot_path, "Label overlap heatmap")],
            output_tables=[ReportOutput(csv_path, "Label overlap counts")]
        )
