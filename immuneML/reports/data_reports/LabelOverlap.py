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
    one 'other' row or column; the full table is always exported as CSV. Rows are sorted by size, and columns are
    ordered so that each column sits next to the row it is most common in: when most row values come mainly from one
    column value (e.g., each epitope from one study), the dark cells form a diagonal, and a row spread over several
    columns stands out. Counts are coloured on a log scale, empty cells are left blank, and the numbers are written in
    the cells only when the heatmap is small enough to read (at most 900 cells).

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

    - delimiter (str): if set, label values that contain the delimiter are split into separate values, and the example
      is counted once under each of them (e.g., with delimiter ';', a receptor with study_id 'ref_1;ref_2' is counted
      under both ref_1 and ref_2). This applies to both labels. With normalize set to 'row', the fractions in a row can
      then add up to more than 1. If null, values are not split. By default, it is null.

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

        # which share of each epitope comes from each study, for the 20 largest epitopes and studies,
        # where receptors reported in several studies have the study ids separated by ';'
        my_confounding_report:
            LabelOverlap:
                column_label: study_id
                row_label: epitope
                max_row_values: 20
                max_column_values: 20
                normalize: row
                delimiter: ';'

    """

    MAX_CELLS_WITH_TEXT = 900

    def __init__(self, dataset: Dataset = None, result_path: Path = None, name: str = None,
                 number_of_processes: int = 1,
                 column_label: str = None, row_label: str = None, max_row_values: int = None,
                 max_column_values: int = None, normalize: str = None, delimiter: str = None):
        super().__init__(dataset, result_path, name, number_of_processes)
        self.column_label = column_label
        self.row_label = row_label
        self.max_row_values = max_row_values
        self.max_column_values = max_column_values
        self.normalize = normalize
        self.delimiter = delimiter

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
        ParameterValidator.assert_type_and_value(kwargs["delimiter"], str, "LabelOverlap", "delimiter", nullable=True)
        return LabelOverlap(column_label=kwargs["column_label"], row_label=kwargs["row_label"],
                            max_row_values=kwargs["max_row_values"], max_column_values=kwargs["max_column_values"],
                            normalize=kwargs["normalize"], delimiter=kwargs["delimiter"],
                            name=kwargs.get("name", "LabelOverlap"))

    def check_prerequisites(self):
        # gene labels are included here: comparing a label against V or J gene is one of the main uses of
        # this report, even though the genes are not prediction targets
        label_names = self.dataset.get_label_names(include_gene_labels=True)
        if self.column_label not in label_names or self.row_label not in label_names:
            logging.warning(f"One or both of the specified labels ({[self.column_label, self.row_label]}) do not exist in the dataset.")
            return False
        return True

    def _generate(self) -> ReportResult:
        metadata = self._get_label_pairs()

        # Create a cross-tabulation of the two labels; the row totals count each example once per row value,
        # also when it has several column values
        overlap_matrix = pd.crosstab(metadata[self.row_label], metadata[self.column_label])
        row_totals = metadata.drop_duplicates(subset=["example_index", self.row_label])[self.row_label] \
            .value_counts().reindex(overlap_matrix.index)

        # Save the full table as CSV
        PathBuilder.build(self.result_path)
        csv_path = self.result_path / 'label_overlap.csv'
        overlap_matrix.to_csv(csv_path)
        output_tables = [ReportOutput(csv_path, "Label overlap counts")]

        if self.normalize == 'row':
            fractions_path = self.result_path / 'label_overlap_row_fractions.csv'
            overlap_matrix.div(row_totals, axis=0).to_csv(fractions_path)
            output_tables.append(ReportOutput(fractions_path, f"Label overlap as fractions of each {self.row_label}"))

        counts, row_totals, row_count = self._keep_top_rows(overlap_matrix, row_totals)
        counts = self._keep_top_columns(counts, row_count)

        plot_path = self._plot(counts, row_totals)

        return ReportResult(
            name=self.name,
            info=f"Shows overlap between {self.row_label} and {self.column_label} labels.",
            output_figures=[ReportOutput(plot_path, "Label overlap heatmap")],
            output_tables=output_tables
        )

    def _get_label_pairs(self) -> pd.DataFrame:
        # one row per example and pair of label values (more than one pair per example only if a delimiter is set)
        metadata = pd.DataFrame(self.dataset.get_metadata([self.column_label, self.row_label]))
        metadata = metadata[[self.row_label, self.column_label]].reset_index(drop=True)
        metadata["example_index"] = metadata.index

        if self.delimiter is not None:
            for label in [self.row_label, self.column_label]:
                metadata[label] = metadata[label].map(self._split_value)
                metadata = metadata.explode(label)

        return metadata.dropna(subset=[self.row_label, self.column_label])

    def _split_value(self, value):
        if isinstance(value, str):
            return list(dict.fromkeys(part.strip() for part in value.split(self.delimiter) if part.strip()))
        return value

    def _keep_top_rows(self, matrix: pd.DataFrame, row_totals: pd.Series):
        # sorts rows by size and merges the rows outside the top max_row_values into one 'other' row
        order = row_totals.sort_values(ascending=False, kind="stable").index
        matrix, row_totals = matrix.loc[order], row_totals.loc[order]
        matrix.index = row_totals.index = matrix.index.astype(str)

        if self.max_row_values is None or matrix.shape[0] <= self.max_row_values:
            return matrix, row_totals, matrix.shape[0]

        other_name = f"other ({matrix.shape[0] - self.max_row_values} values)"
        rest = matrix.iloc[self.max_row_values:]
        matrix = pd.concat([matrix.iloc[:self.max_row_values], pd.DataFrame([rest.sum(axis=0)], index=[other_name])])
        row_totals = pd.concat([row_totals.iloc[:self.max_row_values],
                                pd.Series([row_totals.iloc[self.max_row_values:].sum()], index=[other_name])])
        return matrix, row_totals, self.max_row_values

    def _keep_top_columns(self, matrix: pd.DataFrame, row_count: int) -> pd.DataFrame:
        # keeps the top max_column_values columns, orders them along the diagonal and merges the rest into 'other'
        matrix = matrix[matrix.sum(axis=0).sort_values(ascending=False, kind="stable").index]
        matrix.columns = matrix.columns.astype(str)

        other = None
        if self.max_column_values is not None and matrix.shape[1] > self.max_column_values:
            other = matrix.iloc[:, self.max_column_values:].sum(axis=1)
            other.name = f"other ({matrix.shape[1] - self.max_column_values} values)"
            matrix = matrix.iloc[:, :self.max_column_values]

        matrix = self._order_columns_along_diagonal(matrix, row_count)

        return matrix if other is None else pd.concat([matrix, other], axis=1)

    def _order_columns_along_diagonal(self, matrix: pd.DataFrame, row_count: int) -> pd.DataFrame:
        # each column goes to the position of the row (not 'other') where it has the most examples;
        # columns found only in the 'other' row go last
        if row_count == 0 or matrix.shape[1] == 0:
            return matrix

        values = matrix.iloc[:row_count].values
        peak_rows, peak_counts = values.argmax(axis=0), values.max(axis=0)
        order = sorted(range(matrix.shape[1]),
                       key=lambda i: (peak_rows[i] if peak_counts[i] > 0 else row_count, -peak_counts[i]))
        return matrix.iloc[:, order]

    def _plot(self, counts: pd.DataFrame, row_totals: pd.Series) -> Path:
        count_values = counts.values
        is_empty = count_values == 0

        if self.normalize == 'row':
            values = counts.div(row_totals, axis=0).values
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
            x=counts.columns,
            y=counts.index,
            # counts as strings: plotly writes numeric 2D arrays in binary form, from which hover templates in
            # heatmaps cannot read %{customdata}
            customdata=count_values.astype(str),
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
