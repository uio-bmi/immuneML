import logging
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import plotly.express as px

from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.encoding_reports.FeatureReport import FeatureReport


class FeatureDistribution(FeatureReport):
    """
    Encoding a dataset results in a numeric matrix, where the rows are examples (e.g., sequences, receptors, repertoires)
    and the columns are features. For example, when :ref:`KmerFrequency` encoder is used, the features are the
    k-mers (AAA, AAC, etc..) and the feature values are the frequencies per k-mer.

    This report plots the distribution of feature values.
    For each feature, a violin plot is created to show the distribution of feature values across all examples.
    The violin plots can be separated into different colors or facets using metadata labels
    (for example: plot the feature distributions of 'cohort1', 'cohort2' and 'cohort3' in different colors to spot biases).

    See also: :py:obj:`~immuneML.reports.encoding_reports.FeatureValueBarplot.FeatureValueBarplot` report to plot
    a simple bar chart per feature (average across examples), rather than the entire distribution.
    Or :py:obj:`~immuneML.reports.encoding_reports.FeatureDistribution.FeatureComparison` report to compare
    features across binary metadata labels (e.g., plot the feature value of 'sick' repertoires on the x axis,
    and 'healthy' repertoires on the y axis).


    Example output:

    .. image:: ../../_static/images/reports/feature_distribution.png
       :alt: Feature distribution report example
       :width: 750


    **Specification arguments:**

    - color_grouping_label (str): The label that is used to color each bar, at each level of the grouping_label.

    - row_grouping_label (str): The label that is used to group bars into different row facets.

    - column_grouping_label (str): The label that is used to group bars into different column facets.

    - mode (str): either 'normal', 'sparse' or 'auto' (default). in the 'normal' mode there are normal boxplots
      corresponding to each column of the encoded dataset matrix; in the 'sparse' mode all zero cells are eliminated before
      passing the data to the boxplots. If mode is set to 'auto', then it will automatically
      set to 'sparse' if the density of the matrix is below 0.01

    - x_title (str): x-axis label

    - y_title (str): y-axis label

    - plot_top_n (int): plot n of the largest features on average separately (useful when there are too many features
      to plot at the same time). The n features are chosen based on the average feature values across all examples
      without grouping by labels.

    - plot_bottom_n (int): plot n of the smallest features on average separately (useful when there are too many
      features to plot at the same time). The n features are chosen based on the average feature values across all
      examples without grouping by labels.

    - plot_all_features (bool): whether to plot all (might be slow for large number of features)

    - error_function (str): which error function to use for the error bar. Options are 'std' (standard deviation) or
      'sem' (standard error of the mean). Default: std.


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            reports:
                my_fdistr_report:
                    FeatureDistribution:
                        mode: sparse
                        plot_all_features: True
                        plot_top_n: 10
                        plot_bottom_n: 10

    """

    @classmethod
    def build_object(cls, **kwargs):
        return FeatureDistribution(**kwargs)

    def __init__(self, dataset: Dataset = None, result_path: Path = None, color_grouping_label: str = None,
                 row_grouping_label=None, column_grouping_label=None,
                 mode: str = 'auto', x_title: str = None, y_title: str = None, number_of_processes: int = 1,
                 name: str = None, error_function: str = None,
                 plot_top_n: int = None, plot_bottom_n: int = None, plot_all_features: bool = True):
        super().__init__(dataset=dataset, result_path=result_path, color_grouping_label=color_grouping_label,
                         row_grouping_label=row_grouping_label, column_grouping_label=column_grouping_label,
                         number_of_processes=number_of_processes, name=name, error_function=error_function)
        self.x_title = x_title if x_title is not None else self.x
        self.y_title = y_title if y_title is not None else "value"
        self.mode = mode
        self.result_name = "feature_distributions"
        self.plot_all_features = plot_all_features
        self.plot_top_n = plot_top_n
        self.plot_bottom_n = plot_bottom_n

    def _generate(self):
        result = self._generate_report_result()
        result.info = "Each boxplot represents one feature of the encoded data matrix, and shows the distribution of values for that feature."
        return result

    def _plot(self, data_long_format, mode='sparse') -> Tuple[List[ReportOutput], List[ReportOutput]]:

        plotting_data_dict = self._get_plotting_data_dict(data_long_format)

        output_figures = []
        output_tables = []
        sparse_threshold = 0.01

        for key, data in plotting_data_dict.items():

            if self.mode == 'auto':
                if (data.value == 0).mean() < sparse_threshold:
                    self.mode = 'normal'
                else:
                    self.mode = 'sparse'

            if self.mode == 'sparse':
                output_figures, output_tables = self._plot_sparse(key, data, output_tables, output_figures)
            elif self.mode == 'normal':
                output_figures, output_tables = self._plot_normal(key, data, output_tables, output_figures)

        return output_figures, output_tables

    def _plot_sparse(self, key, plotting_data, output_tables, output_figures) -> Tuple[
        List[ReportOutput], List[ReportOutput]]:
        columns_to_filter = [self.x, "value"]
        for optional_column in [self.color, self.facet_row, self.facet_column]:
            if optional_column is not None:
                columns_to_filter.append(optional_column)

        plotting_data_filtered = plotting_data.loc[plotting_data.value != 0, columns_to_filter]
        columns_to_filter.remove("value")
        total_counts = plotting_data_filtered.groupby(columns_to_filter, as_index=False).agg(
            {"value": 'sum'})
        plotting_data_filtered = plotting_data_filtered.merge(total_counts,
                                                              on=self.x,
                                                              how="left",
                                                              suffixes=('', '_sum')) \
            .fillna(0) \
            .sort_values(by=self.x) \
            .reset_index(drop=True)

        plotting_data_filtered.sort_values([self.color], ascending=[False], inplace=True)

        figure = px.violin(plotting_data_filtered, x=self.x, y="value", color=self.color,
                           facet_row=self.facet_row, facet_col=self.facet_column,
                           labels={
                               "value": self.y_title,
                               self.x: self.x_title,
                           }, template='plotly_white',
                           color_discrete_sequence=px.colors.diverging.Tealrose)
        figure.update_layout(xaxis={'categoryorder': 'total descending'})


        file_path = self.result_path / f"{self.result_name}_{key}.html"
        plotting_data.to_csv(self.result_path / f"{self.result_name}_{key}.csv", index=False)

        figure.write_html(str(file_path))
        output_tables.append(
            ReportOutput(path=self.result_path / f"{self.result_name}_{key}.csv", name=f"{self.result_name} {key}"))

        output_figures.append(ReportOutput(path=file_path,
                                           name=f"Distributions of feature values ({key}, sparse data, zero values filtered)"))

        return output_figures, output_tables

    def _plot_normal(self, key, plotting_data, output_tables, output_figures) -> Tuple[
        List[ReportOutput], List[ReportOutput]]:
        figure = px.violin(plotting_data, x=self.x, y="value", color=self.color,
                           facet_row=self.facet_row, facet_col=self.facet_column,
                           labels={
                               "value": self.y_title,
                               self.x: self.x_title,
                           }, template='plotly_white',
                           color_discrete_sequence=px.colors.diverging.Tealrose)

        figure.update_layout(xaxis={'categoryorder': 'total descending'})

        file_path = self.result_path / f"{self.result_name}_{key}.html"
        plotting_data.to_csv(self.result_path / f"{self.result_name}_{key}.csv", index=False)

        figure.write_html(str(file_path))
        output_tables.append(
            ReportOutput(path=self.result_path / f"{self.result_name}_{key}.csv", name=f"{self.result_name} {key}"))

        output_figures.append(ReportOutput(path=file_path,
                                           name=f"Distributions of feature values ({key})"))

        return output_figures, output_tables

    def _get_plotting_data_dict(self, data_long_format):
        plotting_data_all = data_long_format
        groupby_cols_features = [self.x]
        data_groupedby_features = self._get_grouped_data(data_long_format, groupby_cols_features)

        plotting_data_dict = {'all': plotting_data_all} if self.plot_all_features else {}

        if self.plot_top_n:
            top_n_features = data_groupedby_features.iloc[
                np.argpartition(data_groupedby_features['valuemean'].values, -self.plot_top_n)[-self.plot_top_n:]][
                self.x]
            plotting_data_dict[f'top_{self.plot_top_n}'] = plotting_data_all.loc[
                plotting_data_all[self.x].isin(top_n_features)]

        if self.plot_bottom_n:
            bottom_n_features = data_groupedby_features.iloc[
                np.argpartition(data_groupedby_features['valuemean'].values, self.plot_bottom_n)[:self.plot_bottom_n]][
                self.x]
            plotting_data_dict[f'bottom_{self.plot_bottom_n}'] = plotting_data_all.loc[
                plotting_data_all[self.x].isin(bottom_n_features)]

        return plotting_data_dict
