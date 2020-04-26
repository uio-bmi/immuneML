import warnings
from numbers import Number

import pandas as pd
import yaml
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import STAP

from source.environment.EnvironmentSettings import EnvironmentSettings
from source.ml_methods.RandomForestClassifier import RandomForestClassifier
from source.reports.ReportOutput import ReportOutput
from source.reports.ReportResult import ReportResult
from source.reports.ml_reports.CoefficientPlottingSetting import CoefficientPlottingSetting
from source.reports.ml_reports.CoefficientPlottingSettingList import CoefficientPlottingSettingList
from source.reports.ml_reports.MLReport import MLReport
from source.util.ParameterValidator import ParameterValidator
from source.util.PathBuilder import PathBuilder


class Coefficients(MLReport):
    """
    A report that plots the coefficients for a given ML method in a barplot. Can be used for Logistic regression,
    SVM and Random Forest. In the case of Random Forest, the feature importances will be plotted.

    When used in HyperParameter optimization, the report can be used for all models and optimal models, both on the
    the selection and assessment levels.

    Which coefficients should be plotted (for example: only nonzero, above a certain threshold, ...) can be specified.
    Multiple options can be specified simultaneously. The full set of coefficients will also be exported as a csv file.


    Attributes:
        coefs_to_plot (list): A list specifying which coefficients should be plotted.
            For options see :py:obj:`~source.reports.ml_reports.CoefficientPlottingSetting.CoefficientPlottingSetting`.
        cutoff (list): If 'cutoff' is specified under 'coefs_to_plot', the cutoff values can be specified here.
            The coefficients which have an absolute value equal to or greater than the cutoff will be plotted.
        n_largest (list): If 'n_largest' is specified under 'coefs_to_plot', the values for n can be specified here.
            These should be integer values. The n largest coefficients are determined based on their absolute values.

    Specification:

        definitions:
            reports:
                my_coef_report:
                    Coefficients:
                        coefs_to_plot:
                            - all
                            - nonzero
                            - cutoff
                            - n_largest
                        cutoff:
                            - 0.1
                            - 0.01
                        n_largest:
                            - 5
                            - 10

        instructions:
            instruction_1:
                type: HPOptimization
                settings:
                      ...
                assessment:
                    reports:
                        models:
                            - my_coef_report
                        optimal_models:
                            - my_coef_report
                    ...
                selection:
                    reports:
                        models:
                            - my_coef_report
                        optimal_models:
                            - my_coef_report
                    ...
                ...

    """

    @classmethod
    def build_object(cls, **kwargs):
        location = "Coefficients"
        coefs_to_plot = kwargs["coefs_to_plot"]
        cutoff = kwargs["cutoff"]
        n_largest = kwargs["n_largest"]
        name = kwargs["name"] if "name" in kwargs else None

        ParameterValidator.assert_all_in_valid_list([coef.upper() for coef in coefs_to_plot],
                                                    [item.name.upper() for item in CoefficientPlottingSetting], location,
                                                    "coefs_to_plot")

        if CoefficientPlottingSetting.CUTOFF in coefs_to_plot:
            ParameterValidator.assert_all_type_and_value(cutoff, Number, location, "cutoff", min_inclusive=1e-15)

        if CoefficientPlottingSetting.N_LARGEST in coefs_to_plot:
            ParameterValidator.assert_all_type_and_value(n_largest, int, location, "n_largest", min_inclusive=1)

        coefs = CoefficientPlottingSettingList()
        for keyword in coefs_to_plot:
            coefs.append(CoefficientPlottingSetting[keyword.upper()])

        return Coefficients(coefs, cutoff, n_largest, name)

    def __init__(self, coefs_to_plot: CoefficientPlottingSettingList, cutoff: list, n_largest: list, name: str = None):
        super(Coefficients, self).__init__()

        self._coefs_to_plot = coefs_to_plot
        self._cutoff = cutoff
        self._n_largest = n_largest
        self.label = None
        self.ml_details_path = None
        self.hp_setting = None
        self.name = name

    def generate(self):
        PathBuilder.build(self.result_path)
        paths = []

        self._set_plotting_parameters()

        plot_data = self._retrieve_plot_data()
        plot_data["abs_coefficients"] = abs(plot_data["coefficients"])
        plot_data.sort_values(by="abs_coefficients", inplace=True, ascending=False)

        result_table_path = self._write_results_table(plot_data[["features", "coefficients"]])
        self._write_settings()

        if CoefficientPlottingSetting.ALL in self._coefs_to_plot:
            plot_path = self._plot(plot_data, "all_coefficients")
            paths.append(plot_path)

        if CoefficientPlottingSetting.NONZERO in self._coefs_to_plot:
            nonzero_data = plot_data[plot_data["coefficients"] != 0]
            plot_path = self._plot(nonzero_data, "nonzero_coefficients")
            paths.append(plot_path)

        if CoefficientPlottingSetting.CUTOFF in self._coefs_to_plot:
            for cutoff_val in self._cutoff:
                cutoff_data = plot_data[plot_data["abs_coefficients"] >= cutoff_val]
                plot_path = self._plot(cutoff_data, "cutoff_{}_coefficients".format(cutoff_val))
                paths.append(plot_path)

        if CoefficientPlottingSetting.N_LARGEST in self._coefs_to_plot:
            for n_val in self._n_largest:
                n_largest_data = plot_data.nlargest(n=n_val, columns=["abs_coefficients"])
                plot_path = self._plot(n_largest_data, "largest_{}_coefficients".format(n_val))
                paths.append(plot_path)

        return ReportResult(self.name, output_tables=[ReportOutput(result_table_path)], output_figures=[ReportOutput(p)
                                                                                                        for p in paths if p is not None])

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
        filepath = self.result_path + "coefficients.csv"
        plotting_data.to_csv(filepath, index=False)
        return filepath

    def _retrieve_plot_data(self):
        coefficients = self.method.get_params(self.label)[self._param_field]

        feature_names = self._retrieve_feature_names()

        return pd.DataFrame({"coefficients": coefficients, "features": feature_names})

    def _retrieve_feature_names(self):
        if self.train_dataset and self.train_dataset.encoded_data:
            return self.train_dataset.encoded_data.feature_names

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
                                  y_lab=self._y_axis_title, x_lab="feature", facet_type="wrap", facet_columns="empty_facet",
                                  facet_scales="free", nrow=1, height=6, sort_by_y=True,
                                  width=8, result_path=self.result_path, result_name=output_name)

                return f"{self.result_path}/{output_name}.pdf"

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
