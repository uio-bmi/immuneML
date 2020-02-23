import warnings

import pandas as pd
import yaml
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import STAP

from source.dsl.report_params_parsers.CoefficientPlottingSetting import CoefficientPlottingSetting
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.ml_methods.RandomForestClassifier import RandomForestClassifier
from source.reports.ml_reports.MLReport import MLReport
from source.util.PathBuilder import PathBuilder


class Coefficients(MLReport):
    """
    A report that outputs the coefficients for a given ML method.

    - For Logistic regression: coefficients
    - For SVM: coefficients
    - For Random Forest: feature importance
    """

    def __init__(self, coefs_to_plot, cutoff, n_largest, label=None, ml_details_path = None, hp_setting=None):
        super(Coefficients, self).__init__()
        self._coefs_to_plot = coefs_to_plot
        self._cutoff = cutoff
        self._n_largest = n_largest
        self.label = label
        self.ml_details_path = ml_details_path
        self.hp_setting = hp_setting

    def generate(self):
        PathBuilder.build(self.result_path)

        self._set_plotting_parameters()

        plot_data = self._retrieve_plot_data()
        plot_data["abs_coefficients"] = abs(plot_data["coefficients"])
        plot_data.sort_values(by="abs_coefficients", inplace=True, ascending=False)

        self._write_results_table(plot_data[["features", "coefficients"]])
        self._write_settings()

        if CoefficientPlottingSetting.ALL in self._coefs_to_plot:
            self._plot(plot_data, "all_coefficients")

        if CoefficientPlottingSetting.NONZERO in self._coefs_to_plot:
            nonzero_data = plot_data[plot_data["coefficients"] != 0]
            self._plot(nonzero_data, "nonzero_coefficients")

        if CoefficientPlottingSetting.CUTOFF in self._coefs_to_plot:
            for cutoff_val in self._cutoff:
                cutoff_data = plot_data[plot_data["abs_coefficients"] >= cutoff_val]
                self._plot(cutoff_data, "cutoff_{}_coefficients".format(cutoff_val))

        if CoefficientPlottingSetting.N_LARGEST in self._coefs_to_plot:
            for n_val in self._n_largest:
                n_largest_data = plot_data.nlargest(n=n_val, columns=["abs_coefficients"])
                self._plot(n_largest_data, "largest_{}_coefficients".format(n_val))

    def _set_plotting_parameters(self):
        if isinstance(self.method, RandomForestClassifier):
            self._param_field = "feature_importances"
            self._y_axis_title = "Feature importance"
        else:
            # SVM, logistic regression, ...
            self._param_field = "coefficients"
            self._y_axis_title = "Coefficient value"

    def _write_settings(self):
        if self.hp_setting is not None:
            with open(self.result_path + "settings.yaml", "w") as file:
                yaml.dump({"preprocessing": self.hp_setting.preproc_sequence_name,
                           "encoder": self.hp_setting.encoder_name,
                           "ml_method": self.hp_setting.ml_method_name},
                          file)

    def _write_results_table(self, plotting_data):
        plotting_data.to_csv(self.result_path + "coefficients.csv", index=False)

    def _retrieve_plot_data(self):
        coefficients = self.method.get_params(self.label)[self._param_field]

        feature_names = self._retrieve_feature_names()

        return pd.DataFrame({"coefficients": coefficients, "features": feature_names})

    def _retrieve_feature_names(self):
        with open(self.ml_details_path, "r") as file:
            params = yaml.load(file)

        return params[self.label]["feature_names"]

    def _plot(self, plotting_data, output_name):

        if plotting_data.empty:
            warnings.warn("Coefficients: empty data subset specified, skipping this plot...")
        else:
            try:
                pandas2ri.activate()

                with open(EnvironmentSettings.visualization_path + "Barplot.R") as f:
                    string = f.read()

                plot = STAP(string, "plot")

                plotting_data.loc[:, "empty_facet"] = ""  # Necessary to remove '(all)' label when not using facets

                plot.plot_barplot(data=plotting_data, x="features", color="NULL", y="coefficients",
                                  ylab=self._y_axis_title, xlab="feature", facet_type="wrap", facet_columns="empty_facet",
                                  facet_scales="free", nrow=1, height=6, sort_by_y=True,
                                  width=8, result_path=self.result_path, result_name=output_name)
            except Exception as e:
                warnings.warn(f"Coefficients: the following exception was thrown when attempting to plot the data:\n{e}")

    def check_prerequisites(self):

        run_report = True

        if not hasattr(self, "method"):
            warnings.warn("Coefficients can only be executed as a model report. Coefficients report will not be created.")
            run_report = False

        if not hasattr(self, "result_path"):
            warnings.warn("Coefficients requires an output 'path' to be set. Coefficients report will not be created.")
            run_report = False

        if not hasattr(self, "ml_details_path"):
            warnings.warn("Coefficients requires an 'ml_details_path' to be set. Coefficients report will not be created.")
            run_report = False

        if not hasattr(self, "label"):
            warnings.warn("Coefficients requires that the relevant 'label' is set. Coefficients report will not be created.")
            run_report = False

        return run_report
