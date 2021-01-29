import warnings
from pathlib import Path

import plotly.express as px

from immuneML.analysis.data_manipulation.DataReshaper import DataReshaper
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.encoding_reports.EncodingReport import EncodingReport
from immuneML.util.PathBuilder import PathBuilder


class FeatureValueBarplot(EncodingReport):
    """
    Plots a barplot of the feature values in a given encoded data matrix, across examples. Can be used in combination
    with any encoding. When the distribution of feature values is of interest (as opposed to showing only the mean
    with user-defined error bar as done in this report), please consider using :ref:`FeatureValueDistplot` instead.

    This report creates a barplot where the height of each bar is the mean value of a feature in a specific group. By
    default, all samples are the group, in which case `grouping_label` is "feature", meaning that each bar is the mean
    value of a given feature, and along the x-axis are the different features. For example, when
    :ref:`KmerFrequency` encoder is used, the features are the k-mers and the feature values are the frequencies per k-mer.

    Optional (metadata) labels can be specified for dividing the bars into groups to make comparisons. Groups
    can be visualized by splitting them across the x-axis, using different colors or different row and column facets.

    Note that if `grouping_label` is specified as something other than "feature", then "feature" must be also specified
    in either `row_grouping_labels` or `column_grouping_labels`, so that each feature is then plotted in a separate
    panel. This prevents the undesired (and often uninterpretable) case where the mean across multiple features
    is plotted.


    Arguments:

        grouping_label (str): The label name used for x-axis grouping of the barplots - defaults to "feature",
        meaning each bar represents one feature.

        color_grouping_label (str): The label that is used to color each bar, at each level of the grouping_label.

        row_grouping_label (str): The label that is used to group bars into different row facets.

        column_grouping_label (str): The label that is used to group bars into different column facets.

        x_title (str): x-axis label

        y_title (str): y-axis label


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_fvb_report:
            FeatureValueBarplot:
                column_grouping_label: timepoint
                row_grouping_label: disease_status
                color_grouping_label: age_group

    """

    @classmethod
    def build_object(cls, **kwargs):
        return FeatureValueBarplot(**kwargs)

    def __init__(self, dataset: RepertoireDataset = None, result_path: Path = None, grouping_label: str = "feature",
                 color_grouping_label: str = None, row_grouping_label=None, column_grouping_label=None,
                 x_title: str = None, y_title: str = None, name: str = None):
        super().__init__(name)
        self.dataset = dataset
        self.result_path = result_path
        self.x = grouping_label
        self.color = color_grouping_label
        self.facet_row = row_grouping_label
        self.facet_column = column_grouping_label
        self.x_title = x_title if x_title is not None else self.x
        self.y_title = y_title if y_title is not None else "value"
        self.result_name = "feature_values"
        self.name = name

    def _generate(self) -> ReportResult:
        PathBuilder.build(self.result_path)
        data_long_format = DataReshaper.reshape(self.dataset)
        table_result = self._write_results_table(data_long_format)
        report_output_fig = self._safe_plot(data_long_format=data_long_format)
        output_figures = None if report_output_fig is None else [report_output_fig]
        return ReportResult(self.name, output_figures, [table_result])

    def _write_results_table(self, data) -> ReportOutput:
        table_path = self.result_path / f"{self.result_name}.csv"
        data.to_csv(table_path, index=False)
        return ReportOutput(table_path, "feature values")

    def std(self, x):
        return x.std(ddof=0)

    def _plot(self, data_long_format) -> ReportOutput:
        groupby_cols = [self.x, self.color, self.facet_row, self.facet_column]
        groupby_cols = [i for i in groupby_cols if i]
        groupby_cols = list(set(groupby_cols))
        plotting_data = data_long_format.groupby(groupby_cols, as_index=False).agg(
            {"value": ['mean', self.std]})

        plotting_data.columns = plotting_data.columns.map(''.join)

        figure = px.bar(plotting_data, x=self.x, y="valuemean", color=self.color, barmode="relative",
                        facet_row=self.facet_row, facet_col=self.facet_column, error_y="valuestd",
                        labels={
                            "valuemean": self.y_title,
                            self.x: self.x_title,
                        }, template='plotly_white',
                        color_discrete_sequence=px.colors.diverging.Tealrose)

        file_path = self.result_path / f"{self.result_name}.html"

        figure.write_html(str(file_path))

        return ReportOutput(path=file_path, name="feature bar plot")

    def check_prerequisites(self):
        location = "FeatureValueBarplot"
        run_report = True

        if self.dataset.encoded_data is None or self.dataset.encoded_data.examples is None:
            warnings.warn(
                f"{location}: this report can only be created for an encoded RepertoireDataset. {location} report will not be created.")
            run_report = False
        else:
            legal_labels = list(self.dataset.encoded_data.labels.keys())
            legal_labels.append("feature")
            legal_labels.append("NULL")

            labels = [self.x, self.color, self.facet_row, self.facet_column]

            for label_param in labels:
                if label_param is not None:
                    if label_param not in legal_labels:
                        warnings.warn(
                            f"{location}: undefined label '{label_param}'. Legal options are: {legal_labels}. {location} report will not be created.")
                        run_report = False

            if "feature" not in labels:
                warnings.warn(
                    f"{location}: `feature` has not been specified in any of `grouping_label`, `row_grouping_labels`, "
                    f"or `column_grouping_labels` - this must be specified so that multiple features are not combined"
                    f"into one, making the plot uninterpretable. {location} report will not be created."
                )
                run_report = False

        return run_report
