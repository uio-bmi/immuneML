import logging
from pathlib import Path

import pandas as pd
import plotly.express as px

from immuneML.data_model.EncodedData import EncodedData
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.dsl.definition_parsers.MLParser import MLParser
from immuneML.ml_methods.dim_reduction.DimRedMethod import DimRedMethod
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.encoding_reports.EncodingReport import EncodingReport
from immuneML.util.PathBuilder import PathBuilder


class DimensionalityReduction(EncodingReport):
    """
    This report visualizes the data obtained by dimensionality reduction.

    **Specification arguments:**

    - label (str): name of the label to use for highlighting data points; or None

    - dim_red_method (str): name of the dimensionality reduction method defined under ml_methods that will be
      used to transform the data for plotting; if None, it will visualize the encoded data of reduced dimensionality if
      set


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            reports:
                rep1:
                    DimensionalityReduction:
                        label: epitope
                        dim_red_method:
                            PCA:
                                n_components: 2

    """

    @classmethod
    def build_object(cls, **kwargs):
        if "dim_red_method" in kwargs:
            cls_name = list(kwargs['dim_red_method'].keys())[0]
            method = MLParser.parse_any_model("dim_red_method", kwargs['dim_red_method'], cls_name)[0]
        else:
            method = None
        return DimensionalityReduction(**{**kwargs, "dim_red_method": method})

    def __init__(self, dataset: Dataset = None, batch_size: int = 1, result_path: Path = None,
                 name: str = None, label: str = None, dim_red_method: DimRedMethod = None):
        super().__init__(dataset=dataset, result_path=result_path, name=name)
        self._label = label
        self._dim_red_method = dim_red_method
        self.info = (f"This report visualizes the encoded data after applying dimensionality reduction "
                     f"({self._dim_red_method.__class__.__name__}).")

    def check_prerequisites(self):
        return (isinstance(self.dataset.encoded_data, EncodedData) and
                self.dataset.encoded_data.dimensionality_reduced_data is not None)

    def _generate(self) -> ReportResult:
        if self._dim_red_method:
            assert self.dataset.encoded_data.examples is not None, \
                f"{DimensionalityReduction.__name__}: data not encoded, report will not be made."
            dim_reduced_data = self._dim_red_method.fit_transform(self.dataset)
        else:
            assert self.dataset.encoded_data.dimensionality_reduced_data is not None
            dim_reduced_data = self.dataset.encoded_data.dimensionality_reduced_data

        assert dim_reduced_data.shape[1] == 2
        data_labels = None

        try:
            data_labels = self.dataset.get_attribute(self._label).tolist()
        except (AttributeError, TypeError) as e:
            logging.warning(f"Label {self._label} not found in the dataset. Skipping label coloring in the plot.")

        PathBuilder.build(self.result_path)

        df = pd.DataFrame({'example_id': self.dataset.get_example_ids(),
                           "x": dim_reduced_data[:, 0], 'y': dim_reduced_data[:, 1]})
        if self._label:
            df[self._label] = data_labels
        df.to_csv(self.result_path / 'dimensionality_reduced_data.csv', index=False)

        report_output_fig = self._safe_plot(df=df, output_written=True)
        output_figures = None if report_output_fig is None else [report_output_fig]

        return ReportResult(name=self.name, info=self.info,
                            output_figures=output_figures,
                            output_tables=[ReportOutput(self.result_path / 'dimensionality_reduced_data.csv',
                                                        'data after dimensionality reduction')])

    def _plot(self, df: pd.DataFrame) -> ReportOutput:
        figure = px.scatter(df, x="x", y="y", color=self._label)
        figure.update_layout(template="plotly_white")
        PathBuilder.build(self.result_path)

        file_path = self.result_path / "dimensionality_reduction.html"
        figure.write_html(str(file_path))
        return ReportOutput(path=file_path, name="Data visualization after dimensionality reduction")
