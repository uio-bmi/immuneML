import pandas as pd
import warnings
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import STAP

from source.environment.EnvironmentSettings import EnvironmentSettings
from source.reports.ml_reports.MLReport import MLReport
from source.util.PathBuilder import PathBuilder
from source.dsl.report_params_parsers.ErrorBarMeaning import ErrorBarMeaning



class BenchmarkHPSettings(MLReport):
    """
    Creates a report that compares the performance of all combinations defined in 'settings',
    taking into account the encoding and ML method (not preprocessing).
    """

    ERRORBAR_CONVERSION = {ErrorBarMeaning.STANDARD_ERROR: "se",
                           ErrorBarMeaning.STANDARD_DEVIATION: "sd",
                           ErrorBarMeaning.CONFIDENCE_INTERVAL: "ci"}

    def __init__(self, errorbar_meaning):
        self.errorbar_meaning = errorbar_meaning


    def generate(self):
        PathBuilder.build(self.result_path)

        plotting_data = self._retrieve_plotting_data()
        self._write_results_table(plotting_data)
        self._plot(plotting_data)

    def _retrieve_plotting_data(self):
        plotting_data = []

        for assessment_state in self.hp_optimization_state.assessment_states:
            for label_key, label_state in assessment_state.label_states.items():
                for assessment_key, assessment_item in label_state.assessment_items.items():
                    plotting_data.append([assessment_state.split_index,
                                          label_key,
                                          assessment_item.hp_setting.encoder_name,
                                          assessment_item.hp_setting.ml_method_name,
                                          assessment_item.performance])
                    # optional: include assessment_item.hp_setting.preproc_sequence_name. for now ignored.

        plotting_data = pd.DataFrame(plotting_data, columns=["fold", "label", "encoding", "ml_method", "performance"])

        return plotting_data

    def _write_results_table(self, plotting_data):
        plotting_data.to_csv(self.result_path + "benchmark_result.csv", index=False)

    def _plot(self, plotting_data):
        pandas2ri.activate()

        with open(EnvironmentSettings.visualization_path + "Barplot.R") as f:
            string = f.read()

        plot = STAP(string, "plot")

        errorbar_meaning_abbr = BenchmarkHPSettings.ERRORBAR_CONVERSION[self.errorbar_meaning]

        plot.plot_barplot(data=plotting_data, x="ml_method", color="ml_method",
                          y="performance", ylab="Performance (balanced accuracy)", xlab="ML method", errorbar_meaning=errorbar_meaning_abbr,
                          facet_rows="encoding", facet_columns="label", facet_type="grid",
                          facet_scales="free_y", facet_switch="x", nrow="NULL", height=6,
                          width=8, result_path=self.result_path, result_name="benchmark_result")

    def check_prerequisites(self):
        if not hasattr(self, "hp_optimization_state"):
            warnings.warn("BenchmarkSettings can only be executed as a hyperparameter report. BenchmarkSettings report will not be created.")
            return False

        if not hasattr(self, "result_path"):
            warnings.warn("BenchmarkSettings requires an output 'path' to be set. BenchmarkSettings report will not be created.")
            return False

        return True