import warnings

from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import STAP

from scripts.specification_util import update_docs_per_mapping
from source.analysis.data_manipulation.DataReshaper import DataReshaper
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.reports.ReportOutput import ReportOutput
from source.reports.ReportResult import ReportResult
from source.reports.encoding_reports.EncodingReport import EncodingReport
from source.util.ParameterValidator import ParameterValidator
from source.util.PathBuilder import PathBuilder
from source.visualization.ErrorBarMeaning import ErrorBarMeaning


class FeatureValueBarplot(EncodingReport):
    """
    Plots a barplot of the feature values in a given encoded data matrix. Can be used in combination
    with any encoding. When the distribution of feature values is of interest, please consider using
    :py:obj:`~source.reports.encoding_reports.FeatureValueDistplot.FeatureValueDistplot` instead.

    This report creates a barplot where each bar represents one feature, and the length of the bar represents
    the feature value. For example, when :py:obj:`~source.encodings.kmer_frequency.KmerFrequencyEncoder.KmerFrequencyEncoder`
    is used, the features are the k-mers and the feature values are the frequencies per k-mer.

    Optional (metadata) labels can be specified for dividing the bars into groups. Groups
    can be visualized with different colors or different row and column facets.


    Attributes:

        color_grouping_label (str): The label that is used to group bars into different colors.

        row_grouping_label (str): The label that is used to group bars into different row facets.

        column_grouping_label (str): The label that is used to group bars into different column facets.

        errorbar_meaning (:py:obj:`~source.visualization.ErrorBarMeaning.ErrorBarMeaning`): The value that
            the error bar should represent. For options see :py:obj:`~source.visualization.ErrorBarMeaning.ErrorBarMeaning`.


    Specification:

    .. indent with spaces
    .. code-block:: yaml

        my_fvb_report:
            FeatureValueBarplot:
                column_grouping_label: timepoint
                row_grouping_label: disease_status
                color_grouping_label: age_group
                errorbar_meaning: STANDARD_ERROR

    """

    ERRORBAR_CONVERSION = {ErrorBarMeaning.STANDARD_ERROR: "se",
                           ErrorBarMeaning.STANDARD_DEVIATION: "sd",
                           ErrorBarMeaning.CONFIDENCE_INTERVAL: "ci"}

    @classmethod
    def build_object(cls, **kwargs):
        location = "FeatureValueBarplot"

        if "errorbar_meaning" in kwargs:
            ParameterValidator.assert_in_valid_list(kwargs["errorbar_meaning"], [item.name for item in ErrorBarMeaning],
                                                    location, "errorbar_meaning")

        return FeatureValueBarplot(**kwargs)

    def __init__(self, dataset: RepertoireDataset = None, result_path: str = None, color_grouping_label: str = None,
                 row_grouping_label: str = None, column_grouping_label: str = None, errorbar_meaning: str = None, name: str = None):
        self.dataset = dataset
        self.result_path = result_path
        self.errorbar_meaning = ErrorBarMeaning[errorbar_meaning.upper()]
        self.color = color_grouping_label if color_grouping_label is not None else "NULL"
        self.facet_rows = row_grouping_label if row_grouping_label is not None else []
        self.facet_columns = column_grouping_label if column_grouping_label is not None else []
        self.result_name = "feature_values"
        self.name = name

    def generate(self) -> ReportResult:
        PathBuilder.build(self.result_path)
        data_long_format = DataReshaper.reshape(self.dataset)
        table_result = self._write_results_table(data_long_format)
        report_output_fig = self._safe_plot(data_long_format=data_long_format)
        output_figures = None if report_output_fig is None else [report_output_fig]
        return ReportResult(self.name, output_figures, [table_result])

    def _write_results_table(self, data) -> ReportOutput:
        table_path = f"{self.result_path}{self.result_name}.csv"
        data.to_csv(table_path, index=False)
        return ReportOutput(table_path, "feature values")

    def _plot(self, data_long_format) -> ReportOutput:
        pandas2ri.activate()

        with open(EnvironmentSettings.root_path + "source/visualization/Barplot.R") as f:
            string = f.read()

        plot = STAP(string, "plot")

        errorbar_meaning_abbr = FeatureValueBarplot.ERRORBAR_CONVERSION[self.errorbar_meaning]

        plot.plot_barplot(data=data_long_format, x="feature", color=self.color,
                          facet_rows=self.facet_rows, facet_columns=self.facet_columns, facet_type="grid",
                          facet_scales="free", height=6, width=8, result_path=self.result_path,
                          result_name=self.result_name, errorbar_meaning=errorbar_meaning_abbr)

        return ReportOutput(f"{self.result_path}{self.result_name}.pdf", "feature bar plot")

    def check_prerequisites(self):
        location = "FeatureValueBarplot"
        run_report = True

        if self.dataset.encoded_data is None or self.dataset.encoded_data.examples is None:
            warnings.warn(
                f"{location}: this report can only be created for an encoded RepertoireDataset. {location} report will not be created.")
            run_report = False
        else:
            legal_labels = list(self.dataset.encoded_data.labels.keys())

            for label_param in (self.color, self.facet_rows, self.facet_columns):
                if label_param is not None:
                    if label_param not in legal_labels:
                        warnings.warn(
                            f"{location}: undefined label '{label_param}'. Legal options are: {legal_labels}. {location} report will not be created.")
                        run_report = False

        return run_report

    @staticmethod
    def get_documentation():
        doc = str(FeatureValueBarplot.__doc__)
        valid_values = str([option.name for option in ErrorBarMeaning])[1:-1].replace("'", "`")
        mapping = {
            "For options see :py:obj:`~source.visualization.ErrorBarMeaning.ErrorBarMeaning`.": f"Valid values are: {valid_values}."
        }
        doc = update_docs_per_mapping(doc, mapping)
        return doc
