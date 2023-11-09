import logging
import warnings
from pathlib import Path

import pandas as pd
import plotly.express as px

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.encoding_reports.FeatureReport import FeatureReport
from immuneML.util.ParameterValidator import ParameterValidator


class FeatureComparison(FeatureReport):
    """
    Compares the feature values in a given encoded data matrix across two values for a metadata label.
    These labels are specified in the metadata file for repertoire datasets, or as metadata columns for sequence and receptor datasets.
    Can be used in combination with any encoding and dataset type. This report produces a scatterplot, where each
    point represents one feature, and the values on the x and y axes are the average feature values across two
    subsets of the data. For example, when :ref:`KmerFrequency` encoder is used, and the comparison_label is used to
    represent a disease (true/false), then the features are the k-mers (AAA, AAC, etc..) and their x and y position in the
    scatterplot is determined by their frequency in the subset of the data where disease=true and disease=false.

    Optional metadata labels can be specified to divide the scatterplot into groups based on color, row facets or column facets.

    Alternatively, when the feature values are of interest without comparing them between labelled subgroups of the data, please
    use :ref:`FeatureValueBarplot` or :ref:`FeatureDistribution` instead.

    Specification arguments:

    - comparison_label (str): Mandatory label. This label is used to split the encoded data matrix and define the x and y axes of the plot.
      This label is only allowed to have 2 classes (for example: sick and healthy, binding and non-binding).

    - color_grouping_label (str): Optional label that is used to color the points in the scatterplot. This can not be the same as comparison_label.

    - row_grouping_label (str): Optional label that is used to group scatterplots into different row facets. This can not be the same as comparison_label.

    - column_grouping_label (str): Optional label that is used to group scatterplots into different column facets. This can not be the same as comparison_label.

    - show_error_bar (bool): Whether to show the error bar (standard deviation) for the points, both in the x and y dimension.

    - log_scale (bool): Whether to plot the x and y axes in log10 scale (log_scale = True) or continuous scale (log_scale = False). By default, log_scale is False.

    - keep_fraction (float): The total number of features may be very large and only the features differing significantly across
      comparison labels may be of interest. When the keep_fraction parameter is set below 1, only the fraction of features that
      differs the most across comparison labels is kept for plotting (note that the produced .csv file still contains all data).
      By default, keep_fraction is 1, meaning that all features are plotted.

    - opacity (float): a value between 0 and 1 setting the opacity for data points making it easier to see if there are overlapping points


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_comparison_report:
            FeatureComparison: # compare the different classes defined in the label disease
                comparison_label: disease

    """

    @classmethod
    def build_object(cls, **kwargs):
        comparison_label = kwargs["comparison_label"] if "comparison_label" in kwargs else None
        color_grouping_label = kwargs["color_grouping_label"] if "color_grouping_label" in kwargs else None
        row_grouping_label = kwargs["row_grouping_label"] if "row_grouping_label" in kwargs else None
        column_grouping_label = kwargs["column_grouping_label"] if "column_grouping_label" in kwargs else None
        log_scale = kwargs["log_scale"] if "log_scale" in kwargs else None
        keep_fraction = float(kwargs["keep_fraction"]) if "keep_fraction" in kwargs else 1.0
        ParameterValidator.assert_type_and_value(keep_fraction, float, "FeatureComparison", "keep_fraction", min_inclusive=0, max_inclusive=1)
        ParameterValidator.assert_type_and_value(log_scale, bool, "FeatureComparison", "log_scale")

        assert comparison_label is not None, "FeatureComparison: the parameter 'comparison_label' must be set in order to be able to compare across this label"

        assert comparison_label != color_grouping_label, f"FeatureComparison: comparison label {comparison_label} can not be used as color_grouping_label"
        assert comparison_label != row_grouping_label, f"FeatureComparison: comparison label {comparison_label} can not be used as row_grouping_label"
        assert comparison_label != column_grouping_label, f"FeatureComparison: comparison label {comparison_label} can not be used as column_grouping_label"

        return FeatureComparison(**kwargs)

    def __init__(self, dataset: Dataset = None, result_path: Path = None, comparison_label: str = None,
                 color_grouping_label: str = None, row_grouping_label=None, column_grouping_label=None, opacity: float = 0.7,
                 show_error_bar=True, log_scale: bool = False, keep_fraction: int = 1, number_of_processes: int = 1, name: str = None):
        super().__init__(dataset=dataset, result_path=result_path, color_grouping_label=color_grouping_label,
                         row_grouping_label=row_grouping_label, column_grouping_label=column_grouping_label,
                         number_of_processes=number_of_processes, name=name)
        self.comparison_label = comparison_label
        self.show_error_bar = show_error_bar
        self.log_scale = log_scale
        self.keep_fraction = keep_fraction
        self.opacity = opacity
        self.result_name = "feature_comparison"
        self.name = name

    def _generate(self):
        result = self._generate_report_result()
        result.info = "Compares the feature values in a given encoded data matrix across two values for a metadata label. Each point in the resulting scatterplot represents one feature, and the values on the x and y axes are the average feature values across examples of two different classes. "
        return result

    def _plot(self, data_long_format) -> ReportOutput:
        groupby_cols = [self.comparison_label, self.x, self.color, self.facet_row, self.facet_column]
        groupby_cols = [i for i in groupby_cols if i]
        groupby_cols = list(set(groupby_cols))
        plotting_data = data_long_format.groupby(groupby_cols, as_index=False).agg(
            {"value": ['mean', self.std]})

        plotting_data.columns = plotting_data.columns.map(''.join)

        unique_label_values = plotting_data[self.comparison_label].unique()
        assert len(
            unique_label_values) == 2, f"FeatureComparison: comparison label {self.comparison_label} does not have 2 values; {unique_label_values}"
        class_x, class_y = unique_label_values

        merge_labels = [label for label in ["feature", self.color, self.facet_row, self.facet_column] if label]

        plotting_data = pd.merge(plotting_data.loc[plotting_data[self.comparison_label] == class_x],
                                 plotting_data.loc[plotting_data[self.comparison_label] == class_y],
                                 on=merge_labels)

        if plotting_data.shape[0] == 0:
            logging.warning(f"{FeatureComparison.__name__}: there is no overlap between the (combination of) values of {merge_labels} for "
                            f"different values the {self.comparison_label}.")

        plotting_data = self._filter_keep_fraction(plotting_data) if self.keep_fraction < 1 else plotting_data

        error_x = "valuestd_x" if self.show_error_bar else None
        error_y = "valuestd_y" if self.show_error_bar else None

        figure = px.scatter(plotting_data, x="valuemean_x", y="valuemean_y", error_x=error_x, error_y=error_y,
                            color=self.color, facet_row=self.facet_row, facet_col=self.facet_column, hover_name="feature",
                            log_x=self.log_scale, log_y=self.log_scale, opacity=self.opacity,
                            labels={
                                "valuemean_x": f"Average feature values for {self.comparison_label} = {class_x}",
                                "valuemean_y": f"Average feature values for {self.comparison_label} = {class_y}",
                            }, template='plotly_white',
                            color_discrete_sequence=px.colors.diverging.Tealrose)

        self.add_diagonal(figure)

        file_path = self.result_path / f"{self.result_name}.html"

        figure.write_html(str(file_path))

        return ReportOutput(path=file_path, name=f"Comparison of feature values across {self.comparison_label}")

    def add_diagonal(self, figure):
        figure.update_layout(shapes=[{'type': "line", 'line': dict(color="#B0C2C7", dash="dash"), 'yref': 'paper', 'xref': 'paper', 'y0': 0,
                                      'y1': 1, 'x0': 0, 'x1': 1, 'layer': 'below'}])

    def _filter_keep_fraction(self, plotting_data):
        plotting_data["diff_xy"] = abs(plotting_data["valuemean_x"] - plotting_data["valuemean_y"])
        plotting_data.sort_values(by="diff_xy", inplace=True, ascending=False)
        plotting_data.drop(columns="diff_xy", inplace=True)

        keep_nrows = round(plotting_data.shape[0] * self.keep_fraction)
        return plotting_data.head(keep_nrows)

    def check_prerequisites(self):
        location = self.__class__.__name__
        run_report = True

        if self.dataset.encoded_data is None or self.dataset.encoded_data.examples is None:
            warnings.warn(
                f"{location}: this report can only be created for an encoded dataset. {location} report will not be created.")
            run_report = False
        elif len(self.dataset.encoded_data.examples.shape) != 2:
            warnings.warn(
                f"{location}: this report can only be created for a 2-dimensional encoded dataset. {location} report will not be created.")
            run_report = False
        else:
            legal_labels = list(self.dataset.get_label_names())

            if self.comparison_label not in legal_labels:
                warnings.warn(
                    f"{location}: comparison_label was not defined. {location} report will not be created.")
                run_report = False
            elif len(set(self.dataset.get_metadata([self.comparison_label])[self.comparison_label])) != 2:
                warnings.warn(
                    f"{location}: comparison label {self.comparison_label} does not have 2 values: {set(self.dataset.get_metadata([self.comparison_label])[self.comparison_label])}. {location} report will not be created.")
                run_report = False
            else:
                legal_labels.remove(self.comparison_label)

                for label_param in [self.color, self.facet_row, self.facet_column]:
                    if label_param is not None:
                        if label_param == self.comparison_label:
                            warnings.warn(
                                f"{location}: comparison label '{self.comparison_label}' can not be used in other fields. {location} report will not be created.")
                            run_report = False
                        if label_param not in legal_labels:
                            warnings.warn(
                                f"{location}: undefined label '{label_param}'. Legal options are: {legal_labels}. {location} report will not be created.")
                            run_report = False

        return run_report
