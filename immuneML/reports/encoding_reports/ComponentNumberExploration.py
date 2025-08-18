import copy
import logging
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.metrics import mean_squared_error

from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.dsl.definition_parsers.MLParser import MLParser
from immuneML.ml_methods.dim_reduction.DimRedMethod import DimRedMethod
from immuneML.reports.PlotlyUtil import PlotlyUtil
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.encoding_reports.EncodingReport import EncodingReport
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


class ComponentNumberExploration(EncodingReport):
    """
    This report can be used to choose the number of components for dimensionality reduction methods such as PCA.
    It plots the number  of components vs the reconstruction error (mean squared error) for each component number.

    **Specification arguments:**

    - n_components (list): numbers of components to explore

    - dim_red_method (str): dimensionality reduction method to be used for plotting; if set, in a workflow, this
      dimensionality reduction will be used for plotting instead of any other set in the workflow; if None, it will
      visualize the encoded data of reduced dimensionality if set

    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            reports:
                my_component_number_exp:
                    ComponentNumberExploration:
                        n_components: [2, 3, 4, 5, 6]
                        dim_red_method: PCA

    """

    def __init__(self, dataset: Dataset = None, result_path: Path = None, n_components: List[int] = None,
                 dim_red_method: DimRedMethod = None, number_of_processes: int = 1, name: str = None):
        super().__init__(dataset=dataset, result_path=result_path, number_of_processes=number_of_processes, name=name)
        self.n_components = n_components
        self._dim_red_method = dim_red_method
        self.info = ("This report visualizes how reconstruction error of the data (mean squared error) changes "
                     "across different number of components")

    @classmethod
    def build_object(cls, **kwargs):
        if "dim_red_method" in kwargs and kwargs['dim_red_method'] and kwargs['dim_red_method'] != 'None':
            if isinstance(kwargs['dim_red_method'], str):
                kwargs['dim_red_method'] = {kwargs['dim_red_method']: {}}
            cls_name = list(kwargs['dim_red_method'].keys())[0]
            method = MLParser.parse_any_model("dim_red_method", kwargs['dim_red_method'], cls_name)[0]
        else:
            method = None

        location = f"ComponentNumberExploration ({kwargs['name'] if 'name' in kwargs else ''})"

        ParameterValidator.assert_type_and_value(kwargs['n_components'], list, location, "n_components")
        ParameterValidator.assert_all_type_and_value(kwargs['n_components'], int, location, "n_components", 1)

        return ComponentNumberExploration(**{**kwargs, "dim_red_method": method})

    def _generate(self) -> ReportResult:
        PathBuilder.build(self.result_path)

        df, table_result = self._compute_reconstruction_error()
        figure_result = self._plot(df)

        return ReportResult(self.name,
                            info=f"This report visualizes how reconstruction error of the data (mean squared "
                                 f"error) changes with the number of components of the dimensionality reduction "
                                 f"method ({self._dim_red_method.__class__.__name__}).",
                            output_tables=[table_result], output_figures=[figure_result])

    def _plot(self, df: pd.DataFrame) -> ReportOutput:
        import plotly.express as px

        fig = px.bar(df, x='n_components', y='reconstruction_error')
        fig.update_layout(title='Reconstruction Error vs Number of Components',
                          xaxis_title='Number of Components',
                          yaxis_title='Reconstruction Error (MSE)',
                          template='plotly_white')
        path = PlotlyUtil.write_image_to_file(fig, self.result_path / 'reconstruction_error_plot.html', df.shape[0])
        return ReportOutput(path, 'Reconstruction Error Plot')

    def _compute_reconstruction_error(self) -> Tuple[pd.DataFrame, ReportOutput]:
        reconstruction_errors = []

        for n in self.n_components:
            dim_red_method = copy.deepcopy(self._dim_red_method)

            try:
                dim_red_method.method.n_components = n
                dim_red_method.method.fit_inverse_transform = True
            except AttributeError as e:
                logging.error(f"{self.__class__.__name__}: the dimensionality reduction method does not support "
                              f"setting the number of components or allow fitting inverse transformation, skipping "
                              f"this report.")
                e.message = (f"{self.__class__.__name__}: the dimensionality reduction method does not support "
                             f"setting the number of components or allow fitting inverse transformation, skipping "
                             f"report {self.__class__.__name__} - {self.name}.")
                raise e

            x = dim_red_method.fit_transform(self.dataset)
            x_reconstructed = dim_red_method.method.inverse_transform(x)

            # Mean squared reconstruction error
            mse = mean_squared_error(self.dataset.encoded_data.get_examples_as_np_matrix().flatten(),
                                     x_reconstructed.flatten())
            reconstruction_errors.append(mse)

        df = pd.DataFrame({'n_components': self.n_components, 'reconstruction_error': reconstruction_errors})
        df.to_csv(self.result_path / 'reconstruction_errors.csv', index=False)

        return df, ReportOutput(self.result_path / 'reconstruction_errors.csv', 'reconstruction_errors.csv')

    def check_prerequisites(self):
        if self.dataset.encoded_data is None or self.dataset.encoded_data.examples is None:
            logging.warning(f"{self.__class__.__name__}: the dataset is not encoded, skipping this report...")
            return False
        else:
            return True
