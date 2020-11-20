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
from source.visualization.DistributionPlotType import DistributionPlotType
from source.visualization.PanelAxisScalesType import PanelAxisScalesType
from source.visualization.PanelLabelSwitchType import PanelLabelSwitchType
from source.visualization.PanelLayoutType import PanelLayoutType


class FeatureValueDistplot(EncodingReport):
    """
    Plots the distribution of the feature values in a given encoded data matrix. Can be used in combination
    with any encoding. When a summary of the feature values is desired (as opposed to the entire distribution
    across all examples),  please consider using :ref:`FeatureValueBarplot` instead.

    This report creates distribution plots of the feature values in a group of features. For example, when
    :ref:`KmerFrequency` encoder is used, the distribution of the frequency of each k-mer is plotted.

    Optional (metadata) labels can be specified for splitting the distribution according to different groups. Groups
    can be visualized by splitting across the x-axis, different colors, or different row and column facets.

    Note that if `grouping_label` is specified as something other than "feature", then "feature" must be also specified
    in either `row_grouping_labels` or `column_grouping_labels`, so that each feature is then plotted in a separate
    panel. This prevents the undesired (and often uninterpretable) case where points across multiple features
    are plotted in the same overall distribution.


    Arguments:

        distribution_plot_type (:py:obj:`~source.visialization.DistributionPlotType.DistributionPlotType`):
        what type of distribution plot should be used to visualize the data. Possible options are:
        - LINE: Plots the feature values per group on a vertical line (strip chart).
        - BOX: Creates boxplots of the feature values per group.
        - VIOLIN: Creates violin plots of the feature values per group.
        - SINA: Plots the feature values per group as a sina plot (strip chart with jitter according to density distribution)
        - DENSITY: Creates overlapping density plots of the distributions.
        - RIDGE: Creates ridge plots of the distribution.

        grouping_label (str): The label name used for x-axis grouping of the distributions (when using LINE, BOX,
        VIOLIN or SINA distribution types), or used for plotting the different distribution curves
        (when using DENSITY and RIDGE distribution types).

        color_label (str): The label name used to color the data. When plotting distribution curves
        (DENSITY and RIDGE distribution types) the color grouping label is automatically set to the same field
        as the grouping label.

        connection_label (str): The label name used to connect data points, only if `distribution_plot_type` is LINE.
        This is often useful in the case where multiple examples from one patient are available, and change over
        time, for example, is of interest.

        row_grouping_labels (str or list): The label that is used to group distributions into different row facets.

        column_grouping_labels (str or list): The label that is used to group distributions into different column facets.

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

        my_fvd_report:
            FeatureValueDistplot:
                distribution_plot_type: SINA
                grouping_label: subject_id
                column_grouping_labels: timepoint
                row_grouping_labels: disease_status
                color_label: age_group

    """

    @classmethod
    def build_object(cls, **kwargs):

        location = "FeatureValueDistplot"

        required_param = "grouping_label"
        assert required_param in kwargs, f"{location}: missing keyword argument '{required_param}' under {location}. Add missing names."

        ParameterValidator.assert_in_valid_list(kwargs["distribution_plot_type"], [item.name for item in DistributionPlotType],
                                                location, "distribution_plot_type")
        if "panel_layout_type" in kwargs:
            ParameterValidator.assert_in_valid_list(kwargs["panel_layout_type"], [item.name for item in PanelLayoutType],
                                                    location, "panel_layout_type")
        if "panel_axis_scales_type" in kwargs:
            ParameterValidator.assert_in_valid_list(kwargs["panel_axis_scales_type"], [item.name for item in PanelAxisScalesType],
                                                    location, "panel_axis_scales_type")
        if "panel_label_switch_type" in kwargs:
            ParameterValidator.assert_in_valid_list(kwargs["panel_label_switch_type"], [item.name for item in PanelLabelSwitchType],
                                                    location, "panel_label_switch_type")

        if kwargs["distribution_plot_type"] in ("DENSITY", "RIDGE"):
            if "color_label" in kwargs and kwargs["color_label"] != kwargs["grouping_label"]:
                warnings.warn(f"{location}: illegal color label '{kwargs['color_label']}' has been set to '{kwargs['grouping_label']}'.")
            kwargs["color_label"] = kwargs["grouping_label"]

        return FeatureValueDistplot(**kwargs)

    def __init__(self, dataset: RepertoireDataset = None, result_path: str = None, distribution_plot_type: str = "box",
                 grouping_label: str = None, color_label: str = "NULL", connection_label: str = "NULL",
                 row_grouping_labels: list = None, column_grouping_labels: list = None, panel_layout_type: str = "grid",
                 panel_axis_scales_type: str = "free", panel_label_switch_type: str = "NULL", panel_nrow="NULL",
                 panel_ncol="NULL", height: float = 6, width: float = 8, x_title: str = "NULL", y_title: str = "NULL",
                 color_title: str = "NULL", palette: dict = "NULL", name: str = None):

        super().__init__(name)
        self.dataset = dataset
        self.result_path = result_path
        self.type = distribution_plot_type.lower()
        self.grouping_label = grouping_label
        self.color = color_label
        self.group = connection_label
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

    def _write_results_table(self, data):
        table_path = f"{self.result_path}{self.result_name}.csv"
        data.to_csv(table_path, index=False)
        return ReportOutput(table_path, "feature values")

    def _plot(self, data_long_format):
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import STAP

        pandas2ri.activate()

        with open(EnvironmentSettings.root_path + "source/visualization/Distributions.R") as f:
            string = f.read()

        plot = STAP(string, "plot")

        plot.plot_distribution(data=data_long_format,
                               x=self.grouping_label,
                               y="value",
                               color=self.color,
                               group=self.group,
                               type=self.type,
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

        return ReportOutput(f"{self.result_path}{self.result_name}.pdf", "feature dist plot")

    def check_prerequisites(self):
        location = "FeatureValueDistplot"
        run_report = True

        if self.dataset.encoded_data is None or self.dataset.encoded_data.examples is None:
            warnings.warn(f"{location}: this report can only be created for an encoded RepertoireDataset. {location} report will not be created.")
            run_report = False
        else:
            legal_labels = list(self.dataset.encoded_data.labels.keys())
            legal_labels.append("feature")
            legal_labels.append("NULL")

            labels = [self.grouping_label, self.color] + self.facet_rows + self.facet_columns

            for label_param in labels:
                if label_param is not None:
                    if label_param not in legal_labels:
                        warnings.warn(f"{location}: undefined label '{label_param}'. Legal options are: {legal_labels}. {location} report will not be created.")
                        run_report = False

        return run_report

    @staticmethod
    def get_documentation():
        doc = str(FeatureValueDistplot.__doc__)
        panel_layout = DocEnumHelper.get_enum_names_and_values(PanelLayoutType)
        panel_axis_scales = DocEnumHelper.get_enum_names_and_values(PanelAxisScalesType)
        panel_label_switch_type = DocEnumHelper.get_enum_names_and_values(PanelLabelSwitchType)
        mapping = {
            "For options see :py:obj:`~source.visualization.PanelLayoutType.PanelLayoutType`.": f"Valid values are: {panel_layout}",
            "For options see :py:obj:`~source.visualization.PanelAxisScalesType.PanelAxisScalesType`.": f"Valid values are: {panel_axis_scales}",
            "For options see :py:obj:`~source.visualization.PanelLabelSwitchType.PanelLabelSwitchType`.": f"Valid values are: {panel_label_switch_type}"
        }
        doc = update_docs_per_mapping(doc, mapping)
        return doc
