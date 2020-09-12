import warnings

import pandas as pd
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import STAP

from scripts.specification_util import update_docs_per_mapping
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.reports.ReportOutput import ReportOutput
from source.reports.ReportResult import ReportResult
from source.reports.ml_reports.MLReport import MLReport
from source.util.ParameterValidator import ParameterValidator
from source.util.PathBuilder import PathBuilder
from source.visualization.ErrorBarMeaning import ErrorBarMeaning


class BenchmarkHPSettings(MLReport):
    """
    Report for HyperParameterOptimization: plots the performance for each of the setting combinations
    as defined under 'settings' in the assessment (outer validation) loop.
    The performances are grouped by label (horizontal panels) encoding (vertical panels) and ML method (bar color).
    When multiple data splits are used, the average performance over the data splits is shown with an error bar.

    This report can be used only with TrainMLModel instruction under assessment/reports/hyperparameter.

    Attributes:

        errorbar_meaning (:py:obj:`~source.visualization.ErrorBarMeaning.ErrorBarMeaning`): The value that
            the error bar should represent. For options see :py:obj:`~source.visualization.ErrorBarMeaning.ErrorBarMeaning`.


    Specification:

    .. indent with spaces
    .. code-block:: yaml

        my_hp_report:
            BenchmarkHPSettings:
                errorbar_meaning: STANDARD_ERROR

    """

    ERRORBAR_CONVERSION = {ErrorBarMeaning.STANDARD_ERROR: "se",
                           ErrorBarMeaning.STANDARD_DEVIATION: "sd",
                           ErrorBarMeaning.CONFIDENCE_INTERVAL: "ci"}

    @classmethod
    def build_object(cls, **kwargs):
        valid_values = [item.name.lower() for item in ErrorBarMeaning]
        ParameterValidator.assert_in_valid_list(kwargs["errorbar_meaning"].lower(), valid_values, "BenchmarkHPSettings", "errorbar_meaning")
        errorbar_meaning = ErrorBarMeaning[kwargs["errorbar_meaning"].upper()]
        return BenchmarkHPSettings(errorbar_meaning, kwargs["name"] if "name" in kwargs else None)

    def __init__(self, errorbar_meaning: ErrorBarMeaning, name: str = None):
        super(BenchmarkHPSettings, self).__init__()

        self.errorbar_meaning = errorbar_meaning
        self.hp_optimization_state = None
        self.result_path = None
        self.name = name
        self.result_name = "benchmark_result"
        self.vertical_grouping = "encoding"

    def generate(self) -> ReportResult:
        PathBuilder.build(self.result_path)

        plotting_data = self._retrieve_plotting_data()
        result_table = self._write_results_table(plotting_data)
        report_output_fig = self._safe_plot(plotting_data=plotting_data)
        output_figures = [report_output_fig] if report_output_fig is not None else []

        return ReportResult(self.name, output_tables=[result_table], output_figures=output_figures)

    def _get_vertical_grouping(self, assessment_item):
        return assessment_item.hp_setting.encoder_name

    def _get_color_grouping(self, assessment_item):
        return assessment_item.hp_setting.ml_method_name

    def _retrieve_plotting_data(self):
        plotting_data = []

        for assessment_state in self.hp_optimization_state.assessment_states:
            for label_key, label_state in assessment_state.label_states.items():
                for assessment_key, assessment_item in label_state.assessment_items.items():
                    plotting_data.append([assessment_state.split_index,
                                          label_key,
                                          self._get_vertical_grouping(assessment_item),
                                          self._get_color_grouping(assessment_item),
                                          assessment_item.performance])
                    # optional: include assessment_item.hp_setting.preproc_sequence_name. for now ignored.

        plotting_data = pd.DataFrame(plotting_data, columns=["fold", "label", self.vertical_grouping, "ml_method", "performance"])

        return plotting_data

    def _write_results_table(self, plotting_data):
        filepath = f"{self.result_path}{self.result_name}.csv"
        plotting_data.to_csv(filepath, index=False)
        return ReportOutput(filepath)

    def _plot(self, plotting_data):
        pandas2ri.activate()

        with open(EnvironmentSettings.visualization_path + "Barplot.R") as f:
            string = f.read()

        plot = STAP(string, "plot")

        errorbar_meaning_abbr = BenchmarkHPSettings.ERRORBAR_CONVERSION[self.errorbar_meaning]

        plot.plot_barplot(data=plotting_data, x="ml_method", color="ml_method", color_lab="ML method",
                              y="performance", y_lab="Performance (balanced accuracy)", x_lab="Labels",
                              errorbar_meaning=errorbar_meaning_abbr, facet_rows=self.vertical_grouping, facet_columns="label", facet_type="grid",
                              facet_scales="free_y", facet_switch="x", nrow="NULL", height=6,
                              width=8, result_path=self.result_path, result_name=self.result_name, ml_benchmark=True)

        return ReportOutput(f"{self.result_path}{self.result_name}.pdf")


    def check_prerequisites(self):
        run_report = True

        if not hasattr(self, "hp_optimization_state") or self.hp_optimization_state is None:
            warnings.warn(f"{self.__class__.__name__} can only be executed as a hyperparameter report. BenchmarkHPSettings report will not be created.")
            run_report = False

        if not hasattr(self, "result_path") or self.result_path is None:
            warnings.warn(f"{self.__class__.__name__} requires an output 'path' to be set. BenchmarkHPSettings report will not be created.")
            run_report = False

        return run_report

    @staticmethod
    def get_documentation():
        doc = str(BenchmarkHPSettings.__doc__)
        valid_values = str([option.name for option in ErrorBarMeaning])[1:-1].replace("'", "`")
        mapping = {
            "For options see :py:obj:`~source.visualization.ErrorBarMeaning.ErrorBarMeaning`.": f"Valid values are: {valid_values}."
        }
        doc = update_docs_per_mapping(doc, mapping)
        return doc
