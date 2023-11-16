from pathlib import Path
from typing import List, Tuple

import numpy as np
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

    Specification arguments:

    - color_grouping_label (str): The label that is used to color each bar, at each level of the grouping_label.

    - row_grouping_label (str): The label that is used to group bars into different row facets.

    - column_grouping_label (str): The label that is used to group bars into different column facets.

    - show_error_bar (bool): Whether to show the error bar (standard deviation) for the bars.

    - x_title (str): x-axis label

    - y_title (str): y-axis label

    - plot_top_n (int): plot n of the largest features on average separately (useful when there are too many features to plot at the same time)

    - plot_bottom_n (int): plot n of the smallest features on average separately (useful when there are too many features to plot at the same time)

    - plot_all_features (bool): whether to plot all (might be slow for large number of features)


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

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
                 x_title: str = None, y_title: str = None, show_error_bar=True, name: str = None, plot_all_features: bool = True,
                 number_of_processes: int = 1, plot_top_n: int = None, plot_bottom_n: int = None):
        super().__init__(dataset=dataset, result_path=result_path, color_grouping_label=color_grouping_label,
                         row_grouping_label=row_grouping_label, column_grouping_label=column_grouping_label,
                         name=name, number_of_processes=number_of_processes)
        self.show_error_bar = show_error_bar
        self.x_title = x_title if x_title is not None else self.x
        self.y_title = y_title if y_title is not None else "value"
        self.result_name = "feature_value_barplot"
        self.name = name
        self.plot_all_features = plot_all_features
        self.plot_top_n = plot_top_n
        self.plot_bottom_n = plot_bottom_n

    def _generate(self):
        result = self._generate_report_result()
        result.info = "A barplot of the feature values in a given encoded data matrix, averaged across examples. Each bar in the barplot represents the mean value of a given feature, and along the x-axis are the different features."
        return result

    def _plot(self, data_long_format) -> Tuple[List[ReportOutput], List[ReportOutput]]:
        groupby_cols = [self.x, self.color, self.facet_row, self.facet_column]
        groupby_cols = [i for i in groupby_cols if i]
        groupby_cols = list(set(groupby_cols))
        plotting_data = data_long_format.groupby(groupby_cols, as_index=False).agg(
            {"value": ['mean', self.std]})

        plotting_data.columns = plotting_data.columns.map(''.join)
        plotting_data_dict = {'all': plotting_data} if self.plot_all_features else {}

        error_y = "valuestd" if self.show_error_bar else None
        output_figures = []
        output_tables = []

        if self.plot_top_n:
            plotting_data_dict[f'top_{self.plot_top_n}'] = plotting_data.iloc[np.argpartition(plotting_data['valuemean'].values, -self.plot_top_n)[-self.plot_top_n:]]
        if self.plot_bottom_n:
            plotting_data_dict[f'bottom_{self.plot_bottom_n}'] = plotting_data.iloc[np.argpartition(plotting_data['valuemean'].values, self.plot_bottom_n)[:self.plot_bottom_n]]

        for key, data in plotting_data_dict.items():
            figure = px.bar(data, x=self.x, y="valuemean", color=self.color, barmode="relative",
                            facet_row=self.facet_row, facet_col=self.facet_column, error_y=error_y,
                            labels={
                                "valuemean": self.y_title,
                                self.x: self.x_title,
                            }, template='plotly_white',
                            color_discrete_sequence=px.colors.diverging.Tealrose)

            file_path = self.result_path / f"{self.result_name}_{key}.html"

            figure.write_html(str(file_path))
            data.to_csv(self.result_path / f"{self.result_name}_{key}.csv", index=False)
            output_tables.append(ReportOutput(path=self.result_path / f"{self.result_name}_{key}.csv", name=f"{self.result_name} {key}"))

            output_figures.append(ReportOutput(path=file_path, name=f"Average feature values ({key})"))

        return output_figures, output_tables
