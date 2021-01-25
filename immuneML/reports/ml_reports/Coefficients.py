import logging
from numbers import Number
from pathlib import Path

import pandas as pd
import plotly.express as px
import yaml

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.hyperparameter_optimization import HPSetting
from immuneML.ml_methods.LogisticRegression import LogisticRegression
from immuneML.ml_methods.MLMethod import MLMethod
from immuneML.ml_methods.RandomForestClassifier import RandomForestClassifier
from immuneML.ml_methods.SVM import SVM
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.ml_reports.CoefficientPlottingSetting import CoefficientPlottingSetting
from immuneML.reports.ml_reports.CoefficientPlottingSettingList import CoefficientPlottingSettingList
from immuneML.reports.ml_reports.MLReport import MLReport
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder
from scripts.specification_util import update_docs_per_mapping


class Coefficients(MLReport):
    """
    A report that plots the coefficients for a given ML method in a barplot. Can be used for :ref:`LogisticRegression`,
    :ref:`SVM` and :ref:`RandomForest`. In the case of RandomForest, the feature importances will be plotted.

    When used in :ref:`TrainMLModel` instruction, the report can be specified under 'models', both on
    the selection and assessment levels.

    Which coefficients should be plotted (for example: only nonzero, above a certain threshold, ...) can be specified.
    Multiple options can be specified simultaneously. By default the 25 largest coefficients are plotted.
    The full set of coefficients will also be exported as a csv file.


    Arguments:

        coefs_to_plot (list): A list specifying which coefficients should be plotted. For options see :py:obj:`~immuneML.reports.ml_reports.CoefficientPlottingSetting.CoefficientPlottingSetting`.

        cutoff (list): If 'cutoff' is specified under 'coefs_to_plot', the cutoff values can be specified here. The coefficients which have an absolute value equal to or greater than the cutoff will be plotted.

        n_largest (list): If 'n_largest' is specified under 'coefs_to_plot', the values for n can be specified here. These should be integer values. The n largest coefficients are determined based on their absolute values.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

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

    """

    @classmethod
    def build_object(cls, **kwargs):
        location = "Coefficients"
        coefs_to_plot = [coef.upper() for coef in kwargs["coefs_to_plot"]]

        name = kwargs["name"] if "name" in kwargs else None

        ParameterValidator.assert_all_in_valid_list(coefs_to_plot, [item.name.upper() for item in CoefficientPlottingSetting], location,
                                                    "coefs_to_plot")

        if CoefficientPlottingSetting.CUTOFF.name in coefs_to_plot:
            cutoff = kwargs["cutoff"]
            ParameterValidator.assert_type_and_value(cutoff, list, location, "cutoff")
            ParameterValidator.assert_all_type_and_value(cutoff, Number, location, "cutoff", min_inclusive=1e-15)
        else:
            cutoff = []

        if CoefficientPlottingSetting.N_LARGEST.name in coefs_to_plot:
            n_largest = kwargs["n_largest"]
            ParameterValidator.assert_type_and_value(n_largest, list, location, "n_largest")
            ParameterValidator.assert_all_type_and_value(n_largest, int, location, "n_largest", min_inclusive=1)
        else:
            n_largest = []

        coefs = CoefficientPlottingSettingList()
        for keyword in coefs_to_plot:
            coefs.append(CoefficientPlottingSetting[keyword.upper()])

        return Coefficients(coefs_to_plot=coefs, cutoff=cutoff, n_largest=n_largest, name=name)

    def __init__(self, coefs_to_plot: CoefficientPlottingSettingList, cutoff: list, n_largest: list, train_dataset: Dataset = None,
                 test_dataset: Dataset = None, method: MLMethod = None, result_path: Path = None, name: str = None, hp_setting: HPSetting = None):
        super().__init__(train_dataset, test_dataset, method, result_path, name, hp_setting)

        self._coefs_to_plot = coefs_to_plot
        self._cutoff = cutoff
        self._n_largest = n_largest
        self.label = None

    def _generate(self):
        PathBuilder.build(self.result_path)
        paths = []

        self._set_plotting_parameters()

        plot_data = self._retrieve_plot_data()
        plot_data["abs_coefficients"] = abs(plot_data["coefficients"])
        plot_data.sort_values(by="abs_coefficients", inplace=True, ascending=False)

        result_table_path = self._write_results_table(plot_data[["features", "coefficients"]])
        self._write_settings()

        if CoefficientPlottingSetting.ALL in self._coefs_to_plot:
            report_output_fig = self._plot(plotting_data=plot_data, output_name="all_coefficients")
            paths.append(report_output_fig)

        if CoefficientPlottingSetting.NONZERO in self._coefs_to_plot:
            nonzero_data = plot_data[plot_data["coefficients"] != 0]
            report_output_fig = self._plot(plotting_data=nonzero_data, output_name="nonzero_coefficients")
            paths.append(report_output_fig)

        if CoefficientPlottingSetting.CUTOFF in self._coefs_to_plot:
            for cutoff_val in self._cutoff:
                cutoff_data = plot_data[plot_data["abs_coefficients"] >= cutoff_val]
                report_output_fig = self._plot(plotting_data=cutoff_data, output_name="cutoff_{}_coefficients".format(cutoff_val))
                paths.append(report_output_fig)

        if CoefficientPlottingSetting.N_LARGEST in self._coefs_to_plot:
            for n_val in self._n_largest:
                n_largest_data = plot_data.nlargest(n=n_val, columns=["abs_coefficients"])
                report_output_fig = self._plot(plotting_data=n_largest_data, output_name="largest_{}_coefficients".format(n_val))
                paths.append(report_output_fig)

        return ReportResult(self.name, output_tables=[ReportOutput(result_table_path, "features and coefficients csv")],
                            output_figures=[p for p in paths if p is not None])

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
            file_path = self.result_path / "settings.yaml"
            with file_path.open("w") as file:
                yaml.dump({"preprocessing": self.hp_setting.preproc_sequence_name,
                           "encoder": self.hp_setting.encoder_name,
                           "ml_method": self.hp_setting.ml_method_name},
                          file)

    def _write_results_table(self, plotting_data):
        filepath = self.result_path / "coefficients.csv"
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
            logging.warning(f"Coefficients: empty data subset specified, skipping {output_name} plot...")
        else:

            filename = self.result_path / f"{output_name}.html"

            figure = px.bar(plotting_data, x='features', y='coefficients', template='plotly_white',
                            title=f"{type(self.method).__name__}{' (' + self.method.name + ') - ' if self.method.name is not None else ' - '}"
                                  f"{' '.join(output_name.split('_'))}")
            figure.update_traces(marker_color=px.colors.sequential.Teal[3])

            with filename.open("w") as file:
                figure.write_html(file)

            return ReportOutput(filename)

    def check_prerequisites(self):

        run_report = True

        if not any([isinstance(self.method, legal_method) for legal_method in (RandomForestClassifier, LogisticRegression, SVM)]):
            logging.warning(f"Coefficients report can only be created for RandomForestClassifier, LogisticRegression or SVM, but got "
                            f"{type(self.method).__name__} instead. Coefficients report will not be created.")
            run_report = False

        return run_report

    @staticmethod
    def get_documentation():
        doc = str(Coefficients.__doc__)
        valid_values = str([option.name for option in CoefficientPlottingSetting])[1:-1].replace("'", "`")
        mapping = {
            "For options see :py:obj:`~immuneML.reports.ml_reports.CoefficientPlottingSetting.CoefficientPlottingSetting`.":
                f"Valid values are: {valid_values}."
        }
        doc = update_docs_per_mapping(doc, mapping)
        return doc
