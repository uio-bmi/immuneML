from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


class RandomDataPlot(DataReport):
    """
    This RandomDataPlot is a placeholder for a real Report.
    It plots some random numbers.

    **Specification arguments:**

    - n_points_to_plot (int): The number of random points to plot.


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            reports:
                my_report:
                    RandomDataPlot:
                        n_points_to_plot: 10

    """

    @classmethod
    def build_object(cls, **kwargs):
        # Here you may check the values of given user parameters
        # This will ensure immuneML will crash early (upon parsing the specification) if incorrect parameters are specified
        ParameterValidator.assert_type_and_value(kwargs['n_points_to_plot'], int, RandomDataPlot.__name__, 'n_points_to_plot', min_inclusive=1)

        return RandomDataPlot(**kwargs)

    def __init__(self, dataset: Dataset = None, result_path: Path = None, number_of_processes: int = 1, name: str = None,
                 n_points_to_plot: int = None):
        super().__init__(dataset=dataset, result_path=result_path, number_of_processes=number_of_processes, name=name)
        self.n_points_to_plot = n_points_to_plot

    def check_prerequisites(self):
        # Here you may check properties of the dataset (e.g. dataset type), or parameter-dataset compatibility
        # and return False if the prerequisites are incorrect.
        # This will generate a user-friendly error message and ensure immuneML does not crash, but instead skips the report.
        # Note: parameters should be checked in 'build_object'
        return True

    def _generate(self) -> ReportResult:
        PathBuilder.build(self.result_path)
        df = self._get_random_data()

        # utility function for writing a dataframe to a csv file
        # and creating a ReportOutput object containing the reference
        report_output_table = self._write_output_table(df, self.result_path / 'random_data.csv', name="Random data file")

        # Calling _safe_plot will internally call _plot, but ensure immuneML does not crash if errors occur
        report_output_fig = self._safe_plot(df=df)

        # Ensure output is either None or a list with item (not an empty list or list containing None)
        output_tables = None if report_output_table is None else [report_output_table]
        output_figures = None if report_output_fig is None else [report_output_fig]

        return ReportResult(name=self.name,
                            info="Some random numbers",
                            output_tables=output_tables,
                            output_figures=output_figures)

    def _get_random_data(self):
        return pd.DataFrame({"random_data_dim1": np.random.rand(self.n_points_to_plot),
                             "random_data_dim2": np.random.rand(self.n_points_to_plot)})

    def _plot(self, df: pd.DataFrame) -> ReportOutput:
        figure = px.scatter(df, x="random_data_dim1", y="random_data_dim2", template="plotly_white")
        figure.update_layout(template="plotly_white")

        file_path = self.result_path / "random_data.html"
        figure.write_html(str(file_path))
        return ReportOutput(path=file_path, name="Random data plot")

