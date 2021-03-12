from pathlib import Path

import plotly.express as px

from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.encoding_reports.FeatureReport import FeatureReport


class FeatureValueBarplot(FeatureReport):
    """
    Plots a barplot of the feature values in a given encoded data matrix, averaged across examples. Can be used in combination
    with any encoding and dataset type. Each bar in the barplot represents the mean value of a given feature, and along
    the x-axis are the different features. For example, when :ref:`KmerFrequency` encoder is used, the features are the
    k-mers (AAA, AAC, etc..) and the feature values are the frequencies per k-mer.

    Optional metadata labels can be specified to divide the barplot into groups based on color, row facets or column facets.
    In this case, the average feature values in each group are plotted.
    These labels are specified in the metadata file for repertoire datasets, or as metadata columns for sequence and receptor datasets.

    Alternatively, when the distribution of feature values is of interest (as opposed to showing only the mean, as done here),
    please consider using :ref:`FeatureDistribution` instead.
    When comparing the feature values between two subsets of the data, please use :ref:`FeatureComparison`.

    Arguments:

        color_grouping_label (str): The label that is used to color each bar, at each level of the grouping_label.

        row_grouping_label (str): The label that is used to group bars into different row facets.

        column_grouping_label (str): The label that is used to group bars into different column facets.

        show_error_bar (bool): Whether to show the error bar (standard deviation) for the bars.

        x_title (str): x-axis label

        y_title (str): y-axis label


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_fvb_report:
            FeatureValueBarplot: # timepoint, disease_status and age_group are metadata labels
                column_grouping_label: timepoint
                row_grouping_label: disease_status
                color_grouping_label: age_group

    """

    @classmethod
    def build_object(cls, **kwargs):
        return FeatureValueBarplot(**kwargs)

    def __init__(self, dataset: RepertoireDataset = None, result_path: Path = None,
                 color_grouping_label: str = None, row_grouping_label=None, column_grouping_label=None,
                 x_title: str = None, y_title: str = None, show_error_bar=True, name: str = None):
        super().__init__(dataset=dataset, result_path=result_path, color_grouping_label=color_grouping_label,
                         row_grouping_label=row_grouping_label, column_grouping_label=column_grouping_label, name=name)
        self.show_error_bar = show_error_bar
        self.x_title = x_title if x_title is not None else self.x
        self.y_title = y_title if y_title is not None else "value"
        self.result_name = "feature_value_barplot"
        self.name = name


    def _plot(self, data_long_format) -> ReportOutput:
        groupby_cols = [self.x, self.color, self.facet_row, self.facet_column]
        groupby_cols = [i for i in groupby_cols if i]
        groupby_cols = list(set(groupby_cols))
        plotting_data = data_long_format.groupby(groupby_cols, as_index=False).agg(
            {"value": ['mean', self.std]})

        plotting_data.columns = plotting_data.columns.map(''.join)

        error_y = "valuestd" if self.show_error_bar else None

        figure = px.bar(plotting_data, x=self.x, y="valuemean", color=self.color, barmode="relative",
                        facet_row=self.facet_row, facet_col=self.facet_column, error_y=error_y,
                        labels={
                            "valuemean": self.y_title,
                            self.x: self.x_title,
                        }, template='plotly_white',
                        color_discrete_sequence=px.colors.diverging.Tealrose)

        file_path = self.result_path / f"{self.result_name}.html"

        figure.write_html(str(file_path))

        return ReportOutput(path=file_path, name="Average feature values")
