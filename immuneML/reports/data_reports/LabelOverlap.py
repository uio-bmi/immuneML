import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.reports.PlotlyUtil import PlotlyUtil
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


class LabelOverlap(DataReport):
    """
    This report creates a heatmap where the columns are the values of one label and rows are the values of another label,
    and the cells contain the number of samples that have both label values. It works for any dataset type.

    For labels with many values, only the most frequent values are shown in the heatmap and the rest are merged into
    one 'other' row or column; the full table is always exported as CSV. Counts are coloured on a log scale, empty
    cells are left blank, and the numbers are written in the cells only when the heatmap is small enough to read
    (at most 900 cells).

    **Specification arguments:**

    - column_label (str): Name of the label to be used as columns in the heatmap.

    - row_label (str): Name of the label to be used as rows in the heatmap.

    Either label may also be a V or J gene column (v_call, j_call).

    - max_row_values (int): the maximum number of row label values to show in the heatmap; the most frequent values
      are kept and the rest are merged into 'other'. If null, all values are shown. By default, it is 30.

    - max_column_values (int): the same as max_row_values, for the column label. By default, it is 30.

    - normalize (str): if set to 'row', each cell shows the fraction of the examples in that row that have the column
      value (e.g., which share of each epitope's receptors comes from each study), and the row-normalized full table
      is exported as an additional CSV. If null, raw counts are shown. By default, it is null.

    **YAML specification:**

    .. code-block:: yaml

        my_data_report:
            LabelOverlap:
                column_label: epitope
                row_label: batch

        # how much of the epitope label is already determined by the V gene
        my_germline_report:
            LabelOverlap:
                column_label: v_call
                row_label: epitope

        # which share of each epitope comes from each study, for the 20 largest epitopes and studies
        my_confounding_report:
            LabelOverlap:
                column_label: study_id
                row_label: epitope
                max_row_values: 20
                max_column_values: 20
                normalize: row

    """

    MAX_CELLS_WITH_TEXT = 900

    def __init__(self, dataset: Dataset = None, result_path: Path = None, name: str = None,
                 number_of_processes: int = 1,
                 column_label: str = None, row_label: str = None, max_row_values: int = None,
                 max_column_values: int = None, normalize: str = None):
        super().__init__(dataset, result_path, name, number_of_processes)
        self.column_label = column_label
        self.row_label = row_label
        self.max_row_values = max_row_values
        self.max_column_values = max_column_values
        self.normalize = normalize

    @classmethod
    def build_object(cls, **kwargs):
        ParameterValidator.assert_keys_present(list(kwargs.keys()), ["column_label", "row_label"], "LabelOverlap",
                                               'LabelOverlap')
        ParameterValidator.assert_type_and_value(kwargs["column_label"], str, "LabelOverlap", "column_label")
        ParameterValidator.assert_type_and_value(kwargs["row_label"], str, "LabelOverlap", "row_label")
        for param in ["max_row_values", "max_column_values"]:
            ParameterValidator.assert_type_and_value(kwargs[param], int, "LabelOverlap", param,
                                                     min_inclusive=1, nullable=True)
        if kwargs["normalize"] is not None:
            ParameterValidator.assert_in_valid_list(kwargs["normalize"], ["row"], "LabelOverlap", "normalize")
        return LabelOverlap(column_label=kwargs["column_label"], row_label=kwargs["row_label"],
                            max_row_values=kwargs["max_row_values"], max_column_values=kwargs["max_column_values"],
                            normalize=kwargs["normalize"], name=kwargs.get("name", "LabelOverlap"))

    def check_prerequisites(self):
        # gene labels are included here: comparing a label against V or J gene is one of the main uses of
        # this report, even though the genes are not prediction targets
        label_names = self.dataset.get_label_names(include_gene_labels=True)
        if self.column_label not in label_names or self.row_label not in label_names:
            logging.warning(f"One or both of the specified labels ({[self.column_label, self.row_label]}) do not exist in the dataset.")
            return False
        return True

    def _generate(self) -> ReportResult:
        # Get metadata for both labels
        metadata = self.dataset.get_metadata([self.column_label, self.row_label])

        # Create a cross-tabulation of the two labels
        overlap_matrix = pd.crosstab(metadata[self.row_label], metadata[self.column_label])

        # Save the full table as CSV
        PathBuilder.build(self.result_path)
        csv_path = self.result_path / 'label_overlap.csv'
        overlap_matrix.to_csv(csv_path)
        output_tables = [ReportOutput(csv_path, "Label overlap counts")]

        if self.normalize == 'row':
            fractions_path = self.result_path / 'label_overlap_row_fractions.csv'
            overlap_matrix.div(overlap_matrix.sum(axis=1), axis=0).to_csv(fractions_path)
            output_tables.append(ReportOutput(fractions_path, f"Label overlap as fractions of each {self.row_label}"))

        # Keep only the most frequent values for the plot
        counts = self._keep_top_values(overlap_matrix, self.max_row_values)
        counts = self._keep_top_values(counts.T, self.max_column_values).T

        plot_path = self._plot(counts)

        return ReportResult(
            name=self.name,
            info=f"Shows overlap between {self.row_label} and {self.column_label} labels.",
            output_figures=[ReportOutput(plot_path, "Label overlap heatmap")],
            output_tables=output_tables
        )

    def _keep_top_values(self, matrix: pd.DataFrame, max_values: int) -> pd.DataFrame:
        # sorts rows by their total count and merges the rows outside the top max_values into one 'other' row
        matrix = matrix.loc[matrix.sum(axis=1).sort_values(ascending=False, kind="stable").index]
        matrix.index = matrix.index.astype(str)

        if max_values is None or matrix.shape[0] <= max_values:
            return matrix

        rest = matrix.iloc[max_values:]
        other = pd.DataFrame([rest.sum(axis=0)], index=[f"other ({rest.shape[0]} values)"])
        return pd.concat([matrix.iloc[:max_values], other])

    def _plot(self, counts: pd.DataFrame) -> Path:
        count_values = counts.values
        is_empty = count_values == 0

        if self.normalize == 'row':
            values = counts.div(counts.sum(axis=1), axis=0).values
            z = np.where(is_empty, np.nan, values)
            text = np.where(is_empty, "", np.vectorize(lambda v: f"{v:.2f}")(values))
            colorbar = dict(title="fraction of row")
            value_hover = "Fraction of row: %{z:.3f}<br>Count: %{customdata}"
        else:
            # log scale, so that small counts are still visible next to large ones
            z = np.where(is_empty, np.nan, np.log10(np.where(is_empty, 1, count_values)))
            text = np.where(is_empty, "", count_values.astype(str))
            tick_values = list(range(int(np.nanmax(z)) + 1)) if not np.all(is_empty) else [0]
            colorbar = dict(title="count", tickvals=tick_values, ticktext=[str(10 ** t) for t in tick_values])
            value_hover = "Count: %{customdata}"

        show_text = counts.size <= self.MAX_CELLS_WITH_TEXT

        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=counts.columns.astype(str),
            y=counts.index,
            customdata=count_values,
            colorscale='Blues',
            text=text if show_text else None,
            texttemplate="%{text}" if show_text else None,
            hovertemplate=f"{self.row_label}: " + "%{y}<br>" + f"{self.column_label}: " + "%{x}<br>"
                          + value_hover + "<extra></extra>",
            textfont={"size": 14},
            colorbar=colorbar,
            hoverongaps=False,
        ))

        # Update layout for better readability
        fig.update_layout(
            title=f"Label Overlap: {self.row_label} vs {self.column_label}",
            xaxis_title=self.column_label,
            yaxis_title=self.row_label,
            yaxis_autorange="reversed",
            template="plotly_white",
            font=dict(size=12),
            height=max(600, 40 * len(counts.index) + 200),
        )

        plot_path = self.result_path / 'label_overlap.html'
        return PlotlyUtil.write_image_to_file(fig, plot_path)
