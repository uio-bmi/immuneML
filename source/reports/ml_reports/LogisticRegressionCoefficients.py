import yaml
import warnings
import yaml

import pandas as pd
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import STAP

from source.dsl.report_params_parsers.CoefficientPlottingSetting import CoefficientPlottingSetting
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.reports.ml_reports.MLReport import MLReport
from source.util.PathBuilder import PathBuilder


class LogisticRegressionCoefficients(MLReport):
    """
    A report that outputs the values of the logistic regression coefficients.
    """

    def __init__(self, coefs_to_plot, cutoff, n_largest):
        self._coefs_to_plot = coefs_to_plot
        self._cutoff = cutoff
        self._n_largest = n_largest


    def generate(self):
        PathBuilder.build(self.result_path)

        plot_data = self._retrieve_plot_data()
        self._write_results_table(plot_data)

        if CoefficientPlottingSetting.ALL in self._coefs_to_plot:
            self._plot(plot_data, "all_coefficients")

        if CoefficientPlottingSetting.NONZERO in self._coefs_to_plot:
            nonzero_data = plot_data[plot_data["coefficients"] != 0]
            self._plot(nonzero_data,
                       "nonzero_coefficients")

        plot_data["abs_coefficients"] = abs(plot_data["coefficients"])

        if CoefficientPlottingSetting.CUTOFF in self._coefs_to_plot:
            for cutoff_val in self._cutoff:
                cutoff_data = plot_data[plot_data["abs_coefficients"] >= cutoff_val]
                self._plot(cutoff_data,
                           "cutoff_{}_coefficients".format(cutoff_val))

        if CoefficientPlottingSetting.N_LARGEST in self._coefs_to_plot:
            for n_val in self._n_largest:
                n_largest_data = plot_data.nlargest(n=n_val, columns=["abs_coefficients"])
                self._plot(n_largest_data,
                           "largest_{}_coefficients".format(n_val))


    def _write_results_table(self, plotting_data):
        plotting_data.to_csv(self.result_path + "coefficients.csv", index=False)


    def _retrieve_plot_data(self):
        coefficients = self.method.get_params(self.label)["coefficients"][0]
        feature_names = self._retrieve_feature_names()

        return pd.DataFrame({"coefficients": coefficients, "features": feature_names})


    def _retrieve_feature_names(self):
        with open(self.ml_details_path, "r") as file:
            params = yaml.load(file)

        return params[self.label]["feature_names"]

    def _plot(self, plotting_data, output_name):
        pandas2ri.activate()

        with open(EnvironmentSettings.visualization_path + "Barplot.R") as f:
            string = f.read()

        plot = STAP(string, "plot")

        plotting_data.loc[:, "empty_facet"] = ""  # Necessary to remove '(all)' label when not using facets

        plot.plot_barplot(data=plotting_data, x="features", color="NULL", y="coefficients",
                          ylab="coefficient value", xlab="feature", facet_type="wrap", facet_columns="empty_facet",
                          facet_scales="free", nrow=1, height=6,
                          width=8, result_path=self.result_path, result_name=output_name)



    def check_prerequisites(self):
        if not hasattr(self, "method"):
            warnings.warn("LogisticRegressionCoefficients can only be executed as a model report. LogisticRegressionCoefficients report will not be created.")
            return False

        if not hasattr(self, "result_path"):
            warnings.warn("LogisticRegressionCoefficients requires an output 'path' to be set. LogisticRegressionCoefficients report will not be created.")
            return False

        if not hasattr(self, "ml_details_path"):
            warnings.warn("LogisticRegressionCoefficients requires an 'ml_details_path' to be set. LogisticRegressionCoefficients report will not be created.")
            return False

        if not hasattr(self, "label"):
            warnings.warn("LogisticRegressionCoefficients requires that the relevant 'label' is set. LogisticRegressionCoefficients report will not be created.")
            return False

        return True