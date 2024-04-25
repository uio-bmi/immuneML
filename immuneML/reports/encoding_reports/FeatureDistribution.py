from pathlib import Path

import plotly.express as px

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.encoding_reports.EncodingReport import EncodingReport
from immuneML.reports.encoding_reports.FeatureReport import FeatureReport


class FeatureDistribution(FeatureReport):
    """
    Plots a boxplot for each feature in the encoded data matrix. Can be used in combination
    with any encoding and dataset type.
    Each boxplot represents a feature and shows the distribution of values for that feature.
    For example, when :ref:`KmerFrequency` encoder is used, the features are the
    k-mers (AAA, AAC, etc..) and the feature values are the frequencies per k-mer.

    Two modes can be used: in the 'normal' mode there are normal boxplots corresponding to each column of the
    encoded dataset matrix; in the 'sparse' mode all zero cells are eliminated before
    passing the data to the boxplots. If mode is set to 'auto', then it will automatically
    set to 'sparse' if the density of the matrix is below 0.01

    Optional metadata labels can be specified to divide the boxplots into groups based on color, row facets or column facets.
    These labels are specified in the metadata file for repertoire datasets, or as metadata columns for sequence and receptor datasets.

    Alternatively, when only the mean feature values are of interest (as opposed to showing the complete distribution, as done here),
    please consider using :ref:`FeatureValueBarplot` instead.
    When comparing the feature values between two subsets of the data, please use :ref:`FeatureComparison`.


    **Specification arguments:**

    - color_grouping_label (str): The label that is used to color each bar, at each level of the grouping_label.

    - row_grouping_label (str): The label that is used to group bars into different row facets.

    - column_grouping_label (str): The label that is used to group bars into different column facets.

    - mode (str): either 'normal', 'sparse' or 'auto' (default)

    - x_title (str): x-axis label

    - y_title (str): y-axis label


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            reports:
                my_fdistr_report:
                    FeatureDistribution:
                        mode: sparse

    """

    @classmethod
    def build_object(cls, **kwargs):
        return FeatureDistribution(**kwargs)

    def __init__(self, dataset: Dataset = None, result_path: Path = None, color_grouping_label: str = None,
                 row_grouping_label=None, column_grouping_label=None,
                 mode: str = 'auto', x_title: str = None, y_title: str = None, number_of_processes: int = 1, name: str = None):
        super().__init__(dataset=dataset, result_path=result_path, color_grouping_label=color_grouping_label,
                         row_grouping_label=row_grouping_label, column_grouping_label=column_grouping_label,
                         number_of_processes=number_of_processes, name=name)
        self.x_title = x_title if x_title is not None else self.x
        self.y_title = y_title if y_title is not None else "value"
        self.mode = mode
        self.result_name = "feature_distributions"

    def _generate(self):
        result = self._generate_report_result()
        result.info = "Each boxplot represents one feature of the encoded data matrix, and shows the distribution of values for that feature."
        return result

    def _plot(self, data_long_format, mode='sparse') -> ReportOutput:
        sparse_threshold = 0.01

        if self.mode == 'auto':
            if (data_long_format.value == 0).mean() < sparse_threshold:
                self.mode = 'normal'
            else:
                self.mode = 'sparse'

        if self.mode == 'sparse':
            return self._plot_sparse(data_long_format)
        elif self.mode == 'normal':
            return self._plot_normal(data_long_format)

    def _plot_sparse(self, data_long_format) -> ReportOutput:
        columns_to_filter = [self.x, "value"]
        for optional_column in [self.color, self.facet_row, self.facet_column]:
            if optional_column is not None:
                columns_to_filter.append(optional_column)

        data_long_format_filtered = data_long_format.loc[data_long_format.value != 0, columns_to_filter]
        columns_to_filter.remove("value")
        total_counts = data_long_format_filtered.groupby(columns_to_filter, as_index=False).agg(
            {"value": 'sum'})
        data_long_format_filtered = data_long_format_filtered.merge(total_counts,
                                                                    on=self.x,
                                                                    how="left",
                                                                    suffixes=('', '_sum')) \
            .fillna(0) \
            .sort_values(by=self.x) \
            .reset_index(drop=True)

        figure = px.box(data_long_format_filtered, x=self.x, y="value", color=self.color,
                        facet_row=self.facet_row, facet_col=self.facet_column,
                        labels={
                            "value": self.y_title,
                            self.x: self.x_title,
                        }, template='plotly_white',
                        color_discrete_sequence=px.colors.diverging.Tealrose)

        file_path = self.result_path / f"{self.result_name}.html"

        figure.write_html(str(file_path))

        return ReportOutput(path=file_path, name="feature boxplots")

    def _plot_normal(self, data_long_format) -> ReportOutput:

        figure = px.box(data_long_format, x=self.x, y="value", color=self.color,
                        facet_row=self.facet_row, facet_col=self.facet_column,
                        labels={
                            "value": self.y_title,
                            self.x: self.x_title,
                        }, template='plotly_white',
                        color_discrete_sequence=px.colors.diverging.Tealrose)

        file_path = self.result_path / f"{self.result_name}.html"

        figure.write_html(str(file_path))

        return ReportOutput(path=file_path, name="Distributions of feature values")


##
class DimensionalitReductionReport(EncodingReport):
    def _generate(self):
        """
        Get dim reduced data from self.dataset.encoded...
        make a plot
        return ReportResult
        """
        pass