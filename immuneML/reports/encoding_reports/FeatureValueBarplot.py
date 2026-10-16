from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.sparse import issparse

from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.reports.PlotlyUtil import PlotlyUtil
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.encoding_reports.FeatureReport import FeatureReport


class FeatureValueBarplot(FeatureReport):
    """
    Encoding a dataset results in a numeric matrix, where the rows are examples (e.g., sequences, receptors, repertoires)
    and the columns are features. For example, when :ref:`KmerFrequency` encoder is used, the features are the
    k-mers (AAA, AAC, etc..) and the feature values are the frequencies per k-mer.

    This report plots the mean feature values per feature.
    A bar plot is created where the average feature value across all examples is shown, with optional error bars.
    The bar plots can be separated into different colors or facets using metadata labels
    (for example: plot the average feature values of 'cohort1', 'cohort2' and 'cohort3' in different colors to spot biases).

    See also: :py:obj:`~immuneML.reports.encoding_reports.FeatureDistribution.FeatureDistribution` report to plot
    the distribution of each feature across examples, rather than only showin the mean value in a bar plot.
    Or :py:obj:`~immuneML.reports.encoding_reports.FeatureDistribution.FeatureComparison` report to compare
    features across binary metadata labels (e.g., plot the feature value of 'sick' repertoires on the x axis,
    and 'healthy' repertoires on the y axis.).


    Example output:

    .. image:: ../../_static/images/reports/feature_value_barplot.png
       :alt: Feature value barplot report example
       :width: 750


    **Specification arguments:**

    - color_grouping_label (str): The label that is used to color each bar, at each level of the grouping_label.

    - row_grouping_label (str): The label that is used to group bars into different row facets.

    - column_grouping_label (str): The label that is used to group bars into different column facets.

    - show_error_bar (bool): Whether to show the error bar (standard deviation) for the bars.

    - x_title (str): x-axis label

    - y_title (str): y-axis label

    - plot_top_n (int): plot n of the largest features on average separately (useful when there are too many features
      to plot at the same time). The n features are chosen based on the average feature values across all examples
      without grouping by labels. The plot shows averages per label classes.

    - plot_bottom_n (int): plot n of the smallest features on average separately (useful when there are too many
      features to plot at the same time). The n features are chosen based on the average feature values across all
      examples without grouping by labels. The plot shows averages per label classes.

    - plot_all_features (bool): whether to plot all (might be slow for large number of features)

    - error_function (str): which error function to use for the error bar. Options are 'std' (standard deviation) or
      'sem' (standard error of the mean). Default: std.


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            reports:
                my_fvb_report:
                    FeatureValueBarplot: # timepoint, disease_status and age_group are metadata labels
                        column_grouping_label: timepoint
                        row_grouping_label: disease_status
                        color_grouping_label: age_group
                        plot_all_features: true
                        plot_top_n: 10
                        plot_bottom_n: 5

    """

    @classmethod
    def build_object(cls, **kwargs):
        return FeatureValueBarplot(**kwargs)

    def __init__(self, dataset: RepertoireDataset = None, result_path: Path = None,
                 color_grouping_label: str = None, row_grouping_label=None, column_grouping_label=None,
                 x_title: str = None, y_title: str = None, show_error_bar=True, name: str = None,
                 plot_all_features: bool = True, error_function: str = None,
                 number_of_processes: int = 1, plot_top_n: int = None, plot_bottom_n: int = None):
        super().__init__(dataset=dataset, result_path=result_path, color_grouping_label=color_grouping_label,
                         row_grouping_label=row_grouping_label, column_grouping_label=column_grouping_label,
                         name=name, number_of_processes=number_of_processes, error_function=error_function)
        self.show_error_bar = show_error_bar
        self.x_title = x_title if x_title is not None else self.x
        self.y_title = y_title if y_title is not None else "value"
        self.result_name = "feature_value_barplot"
        self.name = name
        self.plot_all_features = plot_all_features
        self.plot_top_n = plot_top_n
        self.plot_bottom_n = plot_bottom_n
        self._feature_summary = None

    def _generate(self):
        label_cols = self._get_label_cols()
        self._feature_summary = self._summarise_feature_values([])
        result = self._generate_report_result(
            self._summarise_feature_values(label_cols) if label_cols else self._feature_summary)
        result.info = "A barplot of the feature values in a given encoded data matrix, averaged across examples. Each bar in the barplot represents the mean value of a given feature, and along the x-axis are the different features."
        return result

    def _get_label_cols(self) -> List[str]:
        return [col for col in [self.color, self.facet_row, self.facet_column] if col is not None]

    def _summarise_feature_values(self, group_cols: List[str]) -> pd.DataFrame:
        """Mean and error per feature, computed directly from the (possibly sparse) encoded matrix, so that the
        example by feature matrix is never expanded to long format."""
        matrix = self.dataset.encoded_data.examples
        feature_names = self.dataset.encoded_data.feature_names
        feature_names = list(feature_names) if feature_names is not None else list(range(matrix.shape[1]))

        if len(group_cols) == 0:
            groups = {(): np.arange(matrix.shape[0])}
        else:
            metadata = pd.DataFrame(self.dataset.get_metadata(group_cols, return_df=True))
            groups = {key if isinstance(key, tuple) else (key,): np.asarray(indices)
                      for key, indices in metadata.groupby(group_cols, sort=False).indices.items()}

        annotations = self.dataset.encoded_data.feature_annotations
        if annotations is not None:
            annotations = annotations.reset_index(drop=True).drop(columns=[self.x], errors='ignore')

        error_name = self._get_error_function().__name__
        summaries = []
        for group_values, row_indices in groups.items():
            mean, error = self._mean_and_error(matrix[row_indices], error_name)
            summary = pd.DataFrame({self.x: feature_names, "valuemean": mean, f"value{error_name}": error})
            for col, value in zip(group_cols, group_values):
                summary[col] = value
            # annotations are in the same order as the features, so they are added by position: feature names
            # are not necessarily unique and joining on them could duplicate rows
            summaries.append(summary if annotations is None else pd.concat([summary, annotations], axis=1))

        return pd.concat(summaries, ignore_index=True)

    def _mean_and_error(self, matrix, error_name: str) -> Tuple[np.ndarray, np.ndarray]:
        if issparse(matrix):
            mean = np.asarray(matrix.mean(axis=0)).ravel()
            mean_of_squares = np.asarray(matrix.multiply(matrix).mean(axis=0)).ravel()
        else:
            matrix = np.asarray(matrix)
            mean = matrix.mean(axis=0)
            mean_of_squares = np.square(matrix).mean(axis=0)

        # population variance, to match the ddof=0 standard deviation used by FeatureReport
        error = np.sqrt(np.clip(mean_of_squares - np.square(mean), 0, None))
        if error_name == "sem":
            error = error / (matrix.shape[0] ** 0.5)

        return mean, error

    def _plot(self, data_long_format) -> Tuple[List[ReportOutput], List[ReportOutput]]:
        data_groupedby_features = self._feature_summary
        data_groupedby_labels = data_long_format

        plotting_data_dict = {'all': data_groupedby_labels} if self.plot_all_features else {}

        error_y = f"value{self._get_error_function().__name__}" if self.show_error_bar else None
        output_figures = []
        output_tables = []

        if self.plot_top_n:
            top_n_features = data_groupedby_features.iloc[
                np.argpartition(data_groupedby_features['valuemean'].values, -self.plot_top_n)[-self.plot_top_n:]][self.x]
            plotting_data_dict[f'top_{self.plot_top_n}'] = data_groupedby_labels.loc[
                data_groupedby_labels[self.x].isin(top_n_features)]

        if self.plot_bottom_n:
            bottom_n_features = data_groupedby_features.iloc[
                np.argpartition(data_groupedby_features['valuemean'].values, self.plot_bottom_n)[:self.plot_bottom_n]][
                self.x]
            plotting_data_dict[f'bottom_{self.plot_bottom_n}'] = data_groupedby_labels.loc[
                data_groupedby_labels[self.x].isin(bottom_n_features)]

        for key, data in plotting_data_dict.items():
            figure = px.bar(data, x=self.x, y="valuemean", color=self.color, barmode="group",
                            facet_row=self.facet_row, facet_col=self.facet_column, error_y=error_y,
                            labels={
                                "valuemean": self.y_title,
                                self.x: self.x_title,
                            }, template='plotly_white',
                            color_discrete_sequence=px.colors.diverging.Tealrose)
            figure.update_layout(xaxis={'categoryorder': 'total descending'})

            file_path = self.result_path / f"{self.result_name}_{key}.html"

            file_path = PlotlyUtil.write_image_to_file(figure, file_path)
            data.to_csv(self.result_path / f"{self.result_name}_{key}.csv", index=False)
            output_tables.append(ReportOutput(path=self.result_path / f"{self.result_name}_{key}.csv", name=f"{self.result_name} {key}"))

            output_figures.append(ReportOutput(path=file_path, name=f"Average feature values ({key})"))

        return output_figures, output_tables
