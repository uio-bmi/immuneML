import warnings

from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import STAP

from source.analysis.data_manipulation.DataReshaper import DataReshaper
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.reports.ReportOutput import ReportOutput
from source.reports.ReportResult import ReportResult
from source.reports.encoding_reports.EncodingReport import EncodingReport
from source.util.ParameterValidator import ParameterValidator
from source.util.PathBuilder import PathBuilder
from source.visualization.DistributionPlotType import DistributionPlotType


class FeatureValueDistplot(EncodingReport):
    """
    Plots the distribution of the feature values in a given encoded data matrix. Can be used in combination
    with any encoding. When the exact individual feature values are of interest, please consider using
    :py:obj:`~source.reports.encoding_reports.FeatureValueDistplot.FeatureValueBarplot` instead.

    This report creates distributon plots of the feature values in a group of features. For example, when
    :py:obj:`~source.encodings.kmer_frequency.KmerFrequencyEncoder.KmerFrequencyEncoder`
    is used, the distribution of k-mer frequencies is plotted.

    Optional (metadata) labels can be specified for defining the different groups. One distribution is plotted
    per group. Groups can be visualized with different colors or different row and column facets.


    Attributes:
        distribution_plot_type (:py:obj:`~source.visialization.DistributionPlotType.DistributionPlotType`):
            what type of distribution plot should be used to visualize the data. Possible options are:
            LINE: Plots the feature values per group on a vertical line (strip chart).
            BOX: Creates boxplots of the feature values per group.
            VIOLIN: Creates violin plots of the feature values per group.
            SINA: Plots the feature values per group as a sina plot (strip chart with jitter according to density distribution)
            DENSITY: Creates overlapping density plots of the distributions.
            RIDGE: Creates ridge plots of the distribution.
        grouping_label (str): The label name used for x-axis grouping of the distributions (when using LINE, BOX,
            VIOLIN or SINA distribution types), or used for plotting the different distribution curves
            (when using DENSITY and RIDGE distribution types).
        color_label (str): The label name used to color the data. When plotting distribution curves
            (DENSITY and RIDGE distribution types) the color grouping label is automatically set to the same field
            as the grouping label.
        row_grouping_label (str): The label that is used to group distributions into different row facets.
        column_grouping_label (str): The label that is used to group distributions into different column facets.


    Specification:

        definitions:
            datasets:
                my_data:
                    ...
            encodings:
                my_encoding:
                    ...
            reports:
                my_fvd_report:
                    FeatureValueDistplot:
                        distribution_plot_type: SINA
                        grouping_label: donor
                        column_grouping_label: timepoint
                        row_grouping_label: disease_status
                        color_label: age_group

        instructions:
                instruction_1:
                    type: ExploratoryAnalysis
                    analyses:
                        my_fvb_analysis:
                            dataset: my_data
                            encoding: my_encoding
                            report: my_fvd_report
                            labels:
                                - donor
                                - timepoint
                                - disease_status
                                - age_group
    """

    @classmethod
    def build_object(cls, **kwargs):

        location = "FeatureValueDistplot"

        for required_param in ("distribution_plot_type", "grouping_label"):
            assert required_param in kwargs, f"{location}: missing keyword argument '{required_param}' under {location}. Add missing names."

        ParameterValidator.assert_in_valid_list(kwargs["distribution_plot_type"], [item.name for item in DistributionPlotType],
                                                location, "distribution_plot_type")

        if kwargs["distribution_plot_type"] in ("DENSITY", "RIDGE"):
            if "color_label" in kwargs and kwargs["color_label"] != kwargs["grouping_label"]:
                warnings.warn(f"{location}: illegal color label '{kwargs['color_label']}' has been set to '{kwargs['grouping_label']}'.")
            kwargs["color_label"] = kwargs["grouping_label"]

        return FeatureValueDistplot(**kwargs)

    def __init__(self, dataset: RepertoireDataset = None, result_path: str = None,
                 distribution_plot_type: str = "box", grouping_label: str = None, color_label: str = "NULL",
                 row_grouping_label: list = None, column_grouping_label: list = None, name: str = None):

        self.dataset = dataset
        self.result_path = result_path
        self.type = distribution_plot_type.lower()
        self.grouping_label = grouping_label
        self.color = color_label
        self.facet_rows = row_grouping_label if row_grouping_label is not None else []
        self.facet_columns = column_grouping_label if column_grouping_label is not None else []
        self.result_name = "feature_values"
        self.name = name

    def generate(self) -> ReportResult:
        PathBuilder.build(self.result_path)
        data_long_format = DataReshaper.reshape(self.dataset)
        table_result = self._write_results_table(data_long_format)
        figure_result = self._plot(data_long_format)

        return ReportResult(self.name, [figure_result], [table_result])

    def _write_results_table(self, data):
        table_path = f"{self.result_path}/{self.result_name}.csv"
        data.to_csv(table_path, index=False)
        return ReportOutput(table_path, "feature values")

    def _plot(self, data_long_format):
        pandas2ri.activate()

        with open(EnvironmentSettings.root_path + "source/visualization/Distributions.R") as f:
            string = f.read()

        plot = STAP(string, "plot")

        plot.plot_distributions(data=data_long_format, x=self.grouping_label, color=self.color,
                                facet_rows=self.facet_rows, facet_columns=self.facet_columns,
                                facet_type="wrap", facet_scales="free", height=6, width=8,
                                result_path=self.result_path, result_name=self.result_name, type=self.type)

        return ReportOutput(f"{self.result_path}{self.result_name}", "feature dist plot")

    def check_prerequisites(self):
        location = "FeatureValueDistplot"
        run_report = True

        if self.dataset.encoded_data is None or self.dataset.encoded_data.examples is None:
            warnings.warn(f"{location}: this report can only be created for an encoded RepertoireDataset. {location} report will not be created.")
            run_report = False
        else:
            legal_labels = list(self.dataset.encoded_data.labels.keys())

            for label_param in (self.color, self.facet_rows, self.facet_columns):
                if label_param is not None:
                    if label_param not in legal_labels:
                        warnings.warn(f"{location}: undefined label '{label_param}'. Legal options are: {legal_labels}. {location} report will not be created.")
                        run_report = False

        return run_report
