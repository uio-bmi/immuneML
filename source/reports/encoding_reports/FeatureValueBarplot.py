import warnings

from scripts.specification_util import update_docs_per_mapping
from source.analysis.data_manipulation.DataReshaper import DataReshaper
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.reports.ReportOutput import ReportOutput
from source.reports.ReportResult import ReportResult
from source.reports.encoding_reports.EncodingReport import EncodingReport
from source.util.DocEnumHelper import DocEnumHelper
from source.util.ParameterValidator import ParameterValidator
from source.util.PathBuilder import PathBuilder
from source.visualization.ErrorBarMeaning import ErrorBarMeaning
from source.visualization.PanelAxisScalesType import PanelAxisScalesType
from source.visualization.PanelLabelSwitchType import PanelLabelSwitchType
from source.visualization.PanelLayoutType import PanelLayoutType


class FeatureValueBarplot(EncodingReport):
    """
    Plots a barplot of the feature values in a given encoded data matrix, across examples. Can be used in combination
    with any encoding. When the distribution of feature values is of interest (as opposed to showing only the mean
    with user-defined error bar as done in this report), please consider using
    :py:obj:`~source.reports.encoding_reports.FeatureValueDistplot.FeatureValueDistplot` instead.

    This report creates a barplot where the height of each bar is the mean value of a feature in a specific group. By
    default, all samples are the group, in which case `grouping_label` is "feature", meaning that each bar is the mean
    value of a given feature, and along the x-axis are the different features. For example, when
    :py:obj:`~source.encodings.kmer_frequency.KmerFrequencyEncoder.KmerFrequencyEncoder`
    is used, the features are the k-mers and the feature values are the frequencies per k-mer.

    Optional (metadata) labels can be specified for dividing the bars into groups to make comparisons. Groups
    can be visualized by splitting them across the x-axis, using different colors or different row and column facets.

    Note that if `grouping_label` is specified as something other than "feature", then "feature" must be also specified
    in either `row_grouping_labels` or `column_grouping_labels`, so that each feature is then plotted in a separate
    panel. This prevents the undesired (and often uninterpretable) case where the mean across multiple features
    is plotted.


    Arguments:

        grouping_label (str): The label name used for x-axis grouping of the barplots - defaults to "feature," 
        meaning each bar represents one feature.

        color_grouping_label (str): The label that is used to color each bar, at each level of the grouping_label.

        row_grouping_labels (str or list): The label that is used to group bars into different row facets.

        column_grouping_labels (str or list): The label that is used to group bars into different column facets.

        errorbar_meaning (:py:obj:`~source.visualization.ErrorBarMeaning.ErrorBarMeaning`): The value that
        the error bar should represent. For options see :py:obj:`~source.visualization.ErrorBarMeaning.ErrorBarMeaning`.

        panel_layout_type (:py:obj:`~source.visualization.PanelLayoutType.PanelLayoutType`): Parameter determining how the panels will be
        displayed. For options see :py:obj:`~source.visualization.PanelLayoutType.PanelLayoutType`.

        panel_axis_scales_type (:py:obj:`~source.visualization.PanelAxisScalesType.PanelAxisScalesType`): Parameter determining how the x-
        and y-axis scales should vary across panels. For options see :py:obj:`~source.visualization.PanelAxisScalesType.PanelAxisScalesType`.
            
        panel_label_switch_type (:py:obj:`~source.visualization.PanelLabelSwitchType.PanelLabelSwitchType`): Parameter determining
        placement of labels for each panel in the plot. For options see :py:obj:`~source.visualization.PanelLabelSwitchType.PanelLabelSwitchType`.

        panel_nrow (int): Number of rows in plot (if panel_layout_type is `wrap`)

        panel_ncol (int): Number of columns in plot (if panel_layout_type is `wrap`)

        height (float): Height (in inches) of resulting plot

        width (float): Width (in inches) of resulting plot

        x_title (str): x-axis label

        y_title (str): y-axis label

        color_title (str): label for color


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_fvb_report:
            FeatureValueBarplot:
                column_grouping_labels: timepoint
                row_grouping_labels: disease_status
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
        if "panel_layout_type" in kwargs:
            ParameterValidator.assert_in_valid_list(kwargs["panel_layout_type"], [item.name for item in PanelLayoutType],
                                                    location, "panel_layout_type")
        if "panel_axis_scales_type" in kwargs:
            ParameterValidator.assert_in_valid_list(kwargs["panel_axis_scales_type"], [item.name for item in PanelAxisScalesType],
                                                    location, "panel_axis_scales_type")
        if "panel_label_switch_type" in kwargs:
            ParameterValidator.assert_in_valid_list(kwargs["panel_label_switch_type"], [item.name for item in PanelLabelSwitchType],
                                                    location, "panel_label_switch_type")

        return FeatureValueBarplot(**kwargs)

    def __init__(self, dataset: RepertoireDataset = None, result_path: str = None, grouping_label: str = "feature",
                 color_grouping_label: str = "NULL", errorbar_meaning: str = "se", row_grouping_labels=None, column_grouping_labels=None,
                 panel_layout_type: str = "grid", panel_axis_scales_type: str = "free", panel_label_switch_type: str = "NULL", panel_nrow="NULL",
                 panel_ncol="NULL", height: float = 6, width: float = 8, x_title: str = "NULL", y_title: str = "NULL", color_title: str = "NULL",
                 palette: dict = "NULL", name: str = None):

        super().__init__(name)
        self.dataset = dataset
        self.result_path = result_path
        self.x = grouping_label
        self.color = color_grouping_label
        self.errorbar_meaning = ErrorBarMeaning[errorbar_meaning.upper()]
        if row_grouping_labels is None:
            self.facet_rows = []
        else:
            self.facet_rows = [row_grouping_labels] if isinstance(row_grouping_labels, str) else row_grouping_labels
        if column_grouping_labels is None:
            self.facet_columns = []
        else:
            self.facet_columns = [column_grouping_labels] if isinstance(column_grouping_labels, str) else column_grouping_labels
        self.facet_type = PanelLayoutType[panel_layout_type.upper()].name.lower()
        self.facet_scales = PanelAxisScalesType[panel_axis_scales_type.upper()].name.lower()
        self.facet_switch = PanelLabelSwitchType[panel_label_switch_type.upper()].name.lower()
        self.nrow = panel_nrow
        self.ncol = panel_ncol
        self.height = height
        self.width = width
        self.x_title = x_title
        self.y_title = y_title
        self.color_title = color_title
        self.palette = palette
        self.result_name = "feature_values"
        self.name = name

    def generate(self) -> ReportResult:
        PathBuilder.build(self.result_path)
        data_long_format = DataReshaper.reshape(self.dataset)
        table_result = self._write_results_table(data_long_format)
        report_output_fig = self._safe_plot(data_long_format=table_result.path)
        output_figures = None if report_output_fig is None else [report_output_fig]
        return ReportResult(self.name, output_figures, [table_result])

    def _write_results_table(self, data) -> ReportOutput:
        table_path = f"{self.result_path}{self.result_name}.csv"
        data.to_csv(table_path, index=False)
        return ReportOutput(table_path, "feature values")

    def _plot(self, data_long_format) -> ReportOutput:
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import STAP

        pandas2ri.activate()

        with open(EnvironmentSettings.root_path + "source/visualization/Barplot.R") as f:
            string = f.read()

        plot = STAP(string, "plot")

        errorbar_meaning_abbr = FeatureValueBarplot.ERRORBAR_CONVERSION[self.errorbar_meaning]

        plot.plot_barplot(data=data_long_format,
                          x=self.x,
                          y="value",
                          color=self.color,
                          errorbar_meaning=errorbar_meaning_abbr,
                          facet_rows=self.facet_rows,
                          facet_columns=self.facet_columns,
                          facet_type=self.facet_type,
                          facet_scales=self.facet_scales,
                          facet_switch=self.facet_switch,
                          nrow=self.nrow,
                          ncol=self.ncol,
                          height=self.height,
                          width=self.width,
                          x_lab=self.x_title,
                          y_lab=self.y_title,
                          color_lab=self.color_title,
                          palette=self.palette,
                          result_path=self.result_path,
                          result_name=self.result_name)

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
            legal_labels.append("feature")
            legal_labels.append("NULL")

            labels = [self.x, self.color] + self.facet_rows + self.facet_columns

            for label_param in labels:
                if label_param is not None:
                    if label_param not in legal_labels:
                        warnings.warn(
                            f"{location}: undefined label '{label_param}'. Legal options are: {legal_labels}. {location} report will not be created.")
                        run_report = False

            if "feature" not in labels:
                warnings.warn(
                    f"{location}: `feature` has not been specified in any of `grouping_label`, `row_grouping_labels`, "
                    f"or `column_grouping_labels` - this must be specified so that multiple features are not combined"
                    f"into one, making the plot uninterpretable. {location} report will not be created."
                )
                run_report = False

        return run_report

    @staticmethod
    def get_documentation():
        doc = str(FeatureValueBarplot.__doc__)
        error_bar = DocEnumHelper.get_enum_names(ErrorBarMeaning)
        panel_layout = DocEnumHelper.get_enum_names_and_values(PanelLayoutType)
        panel_axis_scales = DocEnumHelper.get_enum_names_and_values(PanelAxisScalesType)
        panel_label_switch_type = DocEnumHelper.get_enum_names_and_values(PanelLabelSwitchType)
        mapping = {
            "For options see :py:obj:`~source.visualization.ErrorBarMeaning.ErrorBarMeaning`.": f"Valid values are: {error_bar}.",
            "For options see :py:obj:`~source.visualization.PanelLayoutType.PanelLayoutType`.": f"Valid values are: {panel_layout}",
            "For options see :py:obj:`~source.visualization.PanelAxisScalesType.PanelAxisScalesType`.": f"Valid values are: {panel_axis_scales}",
            "For options see :py:obj:`~source.visualization.PanelLabelSwitchType.PanelLabelSwitchType`.": f"Valid values are: {panel_label_switch_type}"
        }
        doc = update_docs_per_mapping(doc, mapping)
        return doc
