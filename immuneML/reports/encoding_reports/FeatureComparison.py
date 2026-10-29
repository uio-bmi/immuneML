import logging
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.reports.PlotlyUtil import PlotlyUtil
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
    When comparison_reference is set, the label may have more than two classes, and every other class is compared
    against the reference class in its own panel (for example: the held-out original sequences on the x axis against
    the sequences from each generative model on the y axis).
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
      non-binding), unless comparison_reference is set.

    - comparison_reference (str): Optional class of comparison_label to compare all other classes against. The
      reference class is shown on the x axis, and each other class on the y axis of its own panel (column facet), so
      comparison_label may then have any number of classes. Because the panels take the column facets,
      column_grouping_label cannot be used together with this parameter. By default, this is null, and
      comparison_label must have exactly 2 classes.

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
      produced .csv file still contains all data). With comparison_reference, the fraction is kept per compared class.
      By default, keep_fraction is 1, meaning that all features are plotted.

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
                my_reference_comparison_report:
                    FeatureComparison: # compare each generative model against the held-out original sequences
                        comparison_label: data_origin
                        comparison_reference: original_test

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

        comparison_reference = kwargs.get("comparison_reference", None)
        if comparison_reference is not None:
            assert column_grouping_label is None, \
                f"FeatureComparison: column_grouping_label can not be used together with comparison_reference, since " \
                f"the classes compared against the reference are shown as column facets"

        return FeatureComparison(**kwargs)

    def __init__(self, dataset: Dataset = None, result_path: Path = None, comparison_label: str = None,
                 color_grouping_label: str = None, row_grouping_label=None, column_grouping_label=None,
                 opacity: float = 0.7, error_function: str = 'std', name: str = None,
                 show_error_bar=True, log_scale: bool = False, keep_fraction: int = 1, number_of_processes: int = 1,
                 comparison_reference: str = None):
        super().__init__(dataset=dataset, result_path=result_path, color_grouping_label=color_grouping_label,
                         row_grouping_label=row_grouping_label, column_grouping_label=column_grouping_label,
                         number_of_processes=number_of_processes, name=name, error_function=error_function)
        self.comparison_label = comparison_label
        self.comparison_reference = comparison_reference
        self.show_error_bar = show_error_bar
        self.log_scale = log_scale
        self.keep_fraction = keep_fraction
        self.opacity = opacity
        self.result_name = "feature_comparison"
        self.name = name

    def _generate(self):
        group_cols = list(dict.fromkeys(col for col in [self.comparison_label, self.color, self.facet_row,
                                                        self.facet_column] if col))
        # mean and error per feature and class straight from the (possibly sparse) encoded matrix, so the example by
        # feature matrix is never expanded to long format
        result = self._generate_report_result(self._summarise_feature_values(group_cols))
        result.info = ("Compares the feature values in a given encoded data matrix across two values for a metadata "
                       "label. Each point in the resulting scatterplot represents one feature, and the values on the "
                       "x and y axes are the average feature values across examples of two different classes. ")
        return result

    def _plot(self, data_long_format) -> ReportOutput:
        plotting_data = data_long_format
        unique_label_values = list(plotting_data[self.comparison_label].unique())

        if self.comparison_reference is None:
            assert len(unique_label_values) == 2, \
                f"FeatureComparison: comparison label {self.comparison_label} does not have 2 values; {unique_label_values}"
            class_x, classes_y = unique_label_values[0], unique_label_values[1:]
        else:
            class_x = self.comparison_reference
            classes_y = [value for value in unique_label_values if value != class_x]

        merge_labels = [label for label in ["feature", self.color, self.facet_row, self.facet_column] if label]
        reference_data = plotting_data.loc[plotting_data[self.comparison_label] == class_x]

        compared = []
        for class_y in classes_y:
            pair = pd.merge(reference_data, plotting_data.loc[plotting_data[self.comparison_label] == class_y],
                            on=merge_labels)
            compared.append(self._filter_keep_fraction(pair) if self.keep_fraction < 1 else pair)
        plotting_data = pd.concat(compared, ignore_index=True)

        if plotting_data.shape[0] == 0:
            logging.warning(f"{FeatureComparison.__name__}: there is no overlap between the (combination of) values of {merge_labels} for "
                            f"different values the {self.comparison_label}.")

        error_name = self._get_error_function().__name__
        error_x = f"value{error_name}_x" if self.show_error_bar else None
        error_y = f"value{error_name}_y" if self.show_error_bar else None

        # with a reference, each compared class is a column facet and the y axis is named per facet
        facet_column = f"{self.comparison_label}_y" if self.comparison_reference is not None else self.facet_column
        y_class = classes_y[0] if self.comparison_reference is None else "class in panel title"
        y_title = f"Average feature values for {self.comparison_label} = {y_class}"

        figure = px.scatter(plotting_data, x="valuemean_x", y="valuemean_y", error_x=error_x, error_y=error_y,
                            color=self.color, facet_row=self.facet_row, facet_col=facet_column, hover_name="feature",
                            log_x=self.log_scale, log_y=self.log_scale, opacity=self.opacity,
                            labels={
                                "valuemean_x": f"Average feature values for {self.comparison_label} = {class_x}",
                                "valuemean_y": y_title,
                            }, template='plotly_white',
                            color_discrete_sequence=px.colors.diverging.Tealrose)
        if self.comparison_reference is not None:
            figure.for_each_annotation(lambda a: a.update(text=a.text.replace(f"{facet_column}=", "")))

        self.add_diagonal(figure, x_range=(plotting_data['valuemean_x'].min(), plotting_data["valuemean_x"].max()),
                          y_range=(plotting_data['valuemean_y'].min(), plotting_data["valuemean_y"].max()))

        file_path = self.result_path / f"{self.result_name}.html"

        file_path = PlotlyUtil.write_image_to_file(figure, file_path, plotting_data.shape[0])

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
            elif self.comparison_reference is None and len(set(self.dataset.get_metadata([self.comparison_label])[self.comparison_label])) != 2:
                logging.warning(
                    f"{location}: comparison label {self.comparison_label} does not have 2 values: {set(self.dataset.get_metadata([self.comparison_label])[self.comparison_label])}. {location} report will not be created.")
                run_report = False
            elif self.comparison_reference is not None and self.comparison_reference not in set(self.dataset.get_metadata([self.comparison_label])[self.comparison_label]):
                logging.warning(
                    f"{location}: comparison_reference '{self.comparison_reference}' is not a value of comparison label {self.comparison_label}: {set(self.dataset.get_metadata([self.comparison_label])[self.comparison_label])}. {location} report will not be created.")
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
