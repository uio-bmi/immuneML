import warnings
from pathlib import Path

import plotly.express as px

from immuneML.analysis.data_manipulation.DataReshaper import DataReshaper
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.encoding_reports.EncodingReport import EncodingReport
from immuneML.util.PathBuilder import PathBuilder



class FeatureDistribution(EncodingReport):
    """
    Plots a boxplot for each feature of the encoded dataset in one of the two modes:
    in the 'normal' mode there are normal boxplots corresponding to each column of the 
    encoded dataset matrix; in the 'sparse' mode all zero cells are eliminated before 
    passing the data to the boxplots. If mode is set to 'auto', then it will automatically 
    set to 'sparse' if the density of the matrix is below 0.01

    When only the mean feature values are of interest, please consider using :ref:`FeatureValueBarplot` instead.


    Arguments:

        color_grouping_label (str): The label that is used to color each bar, at each level of the grouping_label.

        mode (str): either 'normal', 'sparse' or 'auto' (default)

        x_title (str): x-axis label

        y_title (str): y-axis label


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_fdistr_report:
            FeatureDistribution:
                mode: sparse

    """

    @classmethod
    def build_object(cls, **kwargs):
        return FeatureDistribution(**kwargs)

    def __init__(self, dataset: RepertoireDataset = None, result_path: Path = None, color_grouping_label: str = None,
                 mode: str = 'auto', x_title: str = None, y_title: str = None, name: str = None):
        super().__init__(name)
        self.dataset = dataset
        self.result_path = result_path
        self.x = "feature"
        self.color = color_grouping_label
        self.mode = mode
        self.x_title = x_title if x_title is not None else self.x
        self.y_title = y_title if y_title is not None else "value"
        self.result_name = "feature_distributions"
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
        columns_to_filter = [self.x, self.color, "value"] if self.color is not None else [self.x, "value"]
        data_long_format_filtered = data_long_format.loc[data_long_format.value != 0, columns_to_filter]
        columns_to_filter.remove("value")
        total_counts = data_long_format_filtered.groupby(columns_to_filter, as_index=False).agg(
            {"value": 'sum'})
        data_long_format_filtered = data_long_format_filtered.merge(total_counts,
                                                                    on=self.x,
                                                                    how="left",
                                                                    suffixes=('', '_sum'))\
                                                             .fillna(0)\
                                                             .sort_values(by=self.x)\
                                                             .reset_index(drop=True)

        
        figure = px.box(data_long_format_filtered, x=self.x, y="value", color=self.color,
                        labels={
                            "valuemean": self.y_title,
                            self.x: self.x_title,
                        }, template='plotly_white',
                        color_discrete_sequence=px.colors.diverging.Tealrose)

        file_path = self.result_path / f"{self.result_name}.html"

        figure.write_html(str(file_path))

        return ReportOutput(path=file_path, name="feature boxplots")
    
    def _plot_normal(self, data_long_format) -> ReportOutput:
        figure = px.box(data_long_format, x=self.x, y="value", color=self.color,
                        labels={
                            "valuemean": self.y_title,
                            self.x: self.x_title,
                        }, template='plotly_white',
                        color_discrete_sequence=px.colors.diverging.Tealrose)

        file_path = self.result_path / f"{self.result_name}.html"

        figure.write_html(str(file_path))

        return ReportOutput(path=file_path, name="feature boxplots")

    
    def check_prerequisites(self):
        location = "FeatureDistribution"
        run_report = True

        if self.dataset.encoded_data is None or self.dataset.encoded_data.examples is None:
            warnings.warn(
                f"{location}: this report can only be created for an encoded RepertoireDataset. {location} report will not be created.")
            run_report = False
        elif len(self.dataset.encoded_data.examples.shape) != 2:
            warnings.warn(
                f"{location}: this report can only be created for a 2-dimensional encoded RepertoireDataset. {location} report will not be created.")
            run_report = False
        else:
            legal_labels = list(self.dataset.encoded_data.labels.keys())
            legal_labels.append("feature")
            legal_labels.append("NULL")

            labels = [self.x]

            for label_param in labels:
                if label_param is not None:
                    if label_param not in legal_labels:
                        warnings.warn(
                            f"{location}: undefined label '{label_param}'. Legal options are: {legal_labels}. {location} report will not be created.")
                        run_report = False

        return run_report
