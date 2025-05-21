import logging
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.encoding_reports.FeatureReport import FeatureReport
from immuneML.util.ParameterValidator import ParameterValidator


class FeatureComparison(FeatureReport):
    """
    Encoding a dataset results in a numeric matrix, where the rows are examples (e.g., sequences, receptors, repertoires)
    and the columns are features. For example, when :ref:`KmerFrequency` encoder is used, the features are the
    k-mers (AAA, AAC, etc..) and the feature values are the frequencies per k-mer.

    This report separates the examples based on a binary metadata label, and plots the mean feature value
    of each feature in one example group against the other example group (for example: plot the feature
    value of 'sick' repertoires on the x axis, and 'healthy' repertoires on the y axis to spot consistent differences).
    The plot can be separated into different colors or facets using other metadata labels
    (for example: plot the average feature values of 'cohort1', 'cohort2' and 'cohort3' in different colors to spot biases).

    Alternatively, when plotting features without comparing them across a binary label, see:
    :py:obj:`~immuneML.reports.encoding_reports.FeatureValueBarplot.FeatureValueBarplot` report to plot
    a simple bar chart per feature (average across examples).
    Or :py:obj:`~immuneML.reports.encoding_reports.FeatureDistribution.FeatureDistribution` report to plot
    the distribution of each feature across examples, rather than only showing the mean value in a bar plot.


    Example output:

    .. image:: ../../_static/images/reports/feature_comparison_zoom.png
       :alt: Feature comparison zoomed in plot with VLEQ highlighted
       :width: 650



    **Specification arguments:**

    - comparison_label (str): Mandatory label. This label is used to split the encoded data matrix and define the x
      and y axes of the plot. This label is only allowed to have 2 classes (for example: sick and healthy, binding and
      non-binding).

    - color_grouping_label (str): Optional label that is used to color the points in the scatterplot. This can not be
      the same as comparison_label.

    - row_grouping_label (str): Optional label that is used to group scatterplots into different row facets.
      This can not be the same as comparison_label.

    - column_grouping_label (str): Optional label that is used to group scatterplots into different column facets.
      This can not be the same as comparison_label.

    - show_error_bar (bool): Whether to show the error bar (standard deviation) for the points, both in the x and y
      dimension.

    - log_scale (bool): Whether to plot the x and y axes in log10 scale (log_scale = True) or continuous scale
      (log_scale = False). By default, log_scale is False.

    - keep_fraction (float): The total number of features may be very large and only the features differing
      significantly across comparison labels may be of interest. When the keep_fraction parameter is set below 1, only
      the fraction of features that differs the most across comparison labels is kept for plotting (note that the
      produced .csv file still contains all data). By default, keep_fraction is 1, meaning that all features are
      plotted.

    - opacity (float): a value between 0 and 1 setting the opacity for data points making it easier to see if there are
      overlapping points

    - error_function (str): which error function to use for the error bar. Options are 'std' (standard deviation) or
      'sem' (standard error of the mean). Default: std.



    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            reports:
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
                 color_grouping_label: str = None, row_grouping_label=None, column_grouping_label=None,
                 opacity: float = 0.7, error_function: str = 'std', name: str = None,
                 show_error_bar=True, log_scale: bool = False, keep_fraction: int = 1, number_of_processes: int = 1):
        super().__init__(dataset=dataset, result_path=result_path, color_grouping_label=color_grouping_label,
                         row_grouping_label=row_grouping_label, column_grouping_label=column_grouping_label,
                         number_of_processes=number_of_processes, name=name, error_function=error_function)
        self.comparison_label = comparison_label
        self.show_error_bar = show_error_bar
        self.log_scale = log_scale
        self.keep_fraction = keep_fraction
        self.opacity = opacity
        self.result_name = "feature_comparison"
        self.name = name

    def _generate(self):
        result = self._generate_report_result()
        result.info = ("Compares the feature values in a given encoded data matrix across two values for a metadata "
                       "label. Each point in the resulting scatterplot represents one feature, and the values on the "
                       "x and y axes are the average feature values across examples of two different classes. ")
        return result

    def _plot(self, data_long_format) -> ReportOutput:
        groupby_cols = [self.comparison_label, self.x, self.color, self.facet_row, self.facet_column]
        groupby_cols = [i for i in groupby_cols if i]
        groupby_cols = list(set(groupby_cols))
        plotting_data = data_long_format.groupby(groupby_cols, as_index=False).agg(
            {"value": ['mean', self._get_error_function()]})

        plotting_data.columns = plotting_data.columns.map(''.join)

        unique_label_values = plotting_data[self.comparison_label].unique()
        assert len(unique_label_values) == 2, \
            f"FeatureComparison: comparison label {self.comparison_label} does not have 2 values; {unique_label_values}"
        class_x, class_y = unique_label_values

        merge_labels = [label for label in ["feature", self.color, self.facet_row, self.facet_column] if label]

        plotting_data = pd.merge(plotting_data.loc[plotting_data[self.comparison_label] == class_x],
                                 plotting_data.loc[plotting_data[self.comparison_label] == class_y],
                                 on=merge_labels)

        if plotting_data.shape[0] == 0:
            logging.warning(f"{FeatureComparison.__name__}: there is no overlap between the (combination of) values of {merge_labels} for "
                            f"different values the {self.comparison_label}.")

        plotting_data = self._filter_keep_fraction(plotting_data) if self.keep_fraction < 1 else plotting_data

        error_x = f"value{self.error_function}_x" if self.show_error_bar else None
        error_y = f"value{self.error_function}_y" if self.show_error_bar else None

        figure = px.scatter(plotting_data, x="valuemean_x", y="valuemean_y", error_x=error_x, error_y=error_y,
                            color=self.color, facet_row=self.facet_row, facet_col=self.facet_column, hover_name="feature",
                            log_x=self.log_scale, log_y=self.log_scale, opacity=self.opacity,
                            labels={
                                "valuemean_x": f"Average feature values for {self.comparison_label} = {class_x}",
                                "valuemean_y": f"Average feature values for {self.comparison_label} = {class_y}",
                            }, template='plotly_white',
                            color_discrete_sequence=px.colors.diverging.Tealrose)

        self.add_diagonal(figure, x_range=(plotting_data['valuemean_x'].min(), plotting_data["valuemean_x"].max()),
                          y_range=(plotting_data['valuemean_y'].min(), plotting_data["valuemean_y"].max()))

        file_path = self.result_path / f"{self.result_name}.html"

        figure.write_html(str(file_path))

        return ReportOutput(path=file_path, name=f"Comparison of feature values across {self.comparison_label}")

    def add_diagonal(self, figure, x_range: tuple, y_range: tuple):
        figure.add_trace(go.Scatter(hoverinfo='skip', x=[min(x_range[0], y_range[0]), max(x_range[1], y_range[1])],
                                    y=[min(x_range[0], y_range[0]), max(x_range[1], y_range[1])],
                                    mode='lines', line=dict(color="#B0C2C7", dash='dash'), zorder=1),
                         row='all', col='all')

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
            logging.warning(
                f"{location}: this report can only be created for an encoded dataset. {location} report will not be created.")
            run_report = False
        elif len(self.dataset.encoded_data.examples.shape) != 2:
            logging.warning(
                f"{location}: this report can only be created for a 2-dimensional encoded dataset. {location} report will not be created.")
            run_report = False
        else:
            legal_labels = list(self.dataset.get_label_names())

            if self.comparison_label not in legal_labels:
                logging.warning(
                    f"{location}: comparison_label was not defined. {location} report will not be created.")
                run_report = False
            elif len(set(self.dataset.get_metadata([self.comparison_label])[self.comparison_label])) != 2:
                logging.warning(
                    f"{location}: comparison label {self.comparison_label} does not have 2 values: {set(self.dataset.get_metadata([self.comparison_label])[self.comparison_label])}. {location} report will not be created.")
                run_report = False
            else:
                legal_labels.remove(self.comparison_label)

                for label_param in [self.color, self.facet_row, self.facet_column]:
                    if label_param is not None:
                        if label_param == self.comparison_label:
                            logging.warning(
                                f"{location}: comparison label '{self.comparison_label}' can not be used in other fields. {location} report will not be created.")
                            run_report = False
                        if label_param not in legal_labels:
                            logging.warning(
                                f"{location}: undefined label '{label_param}'. Legal options are: {legal_labels}. {location} report will not be created.")
                            run_report = False

        return run_report
