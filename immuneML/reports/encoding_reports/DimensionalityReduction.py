import logging
from pathlib import Path
from typing import List

import pandas as pd
import plotly.express as px

from immuneML.data_model.EncodedData import EncodedData
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.dsl.definition_parsers.MLParser import MLParser
from immuneML.ml_methods.dim_reduction.DimRedMethod import DimRedMethod
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.encoding_reports.EncodingReport import EncodingReport
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


class DimensionalityReduction(EncodingReport):
    """
    This report visualizes the data obtained by dimensionality reduction.

    **Specification arguments:**

    - labels (list): names of the label to use for highlighting data points; or None

    - dim_red_method (str): dimensionality reduction method to be used for plotting; if set, in a workflow, this
      dimensionality reduction will be used for plotting instead of any other set in the workflow; if None, it will
      visualize the encoded data of reduced dimensionality if set


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            reports:
                rep1:
                    DimensionalityReduction:
                        labels: [epitope, source]
                        dim_red_method:
                            PCA:
                                n_components: 2

    """

    @classmethod
    def build_object(cls, **kwargs):
        if "dim_red_method" in kwargs and kwargs['dim_red_method'] and kwargs['dim_red_method'] != 'None':
            cls_name = list(kwargs['dim_red_method'].keys())[0]
            method = MLParser.parse_any_model("dim_red_method", kwargs['dim_red_method'], cls_name)[0]
        else:
            method = None

        location = f"DimensionalityReduction ({kwargs['name'] if 'name' in kwargs else ''})"

        # backwards compatibility: to be removed from next major version
        if "label" in kwargs:
            ParameterValidator.warn_deprecated_parameter("label", "labels", location)
            ParameterValidator.assert_type_and_value(kwargs["label"], str, location, "label")
            labels = [kwargs["label"]]
            del kwargs["label"]
        else:
            ParameterValidator.assert_type_and_value(kwargs["labels"], list, location, "labels")
            labels = kwargs["labels"]
            ParameterValidator.assert_all_type_and_value(labels, str, location, "labels")

        return DimensionalityReduction(**{**kwargs, "dim_red_method": method, 'labels': labels})

    def __init__(self, dataset: Dataset = None, batch_size: int = 1, result_path: Path = None,
                 name: str = None, labels: list = None, dim_red_method: DimRedMethod = None):
        super().__init__(dataset=dataset, result_path=result_path, name=name)
        self._labels = labels
        self._dim_red_method = dim_red_method
        self._dimension_names = ['dimension_1',
                                 'dimension_2'] if self._dim_red_method else self._dim_red_method.get_dimension_names()
        self.info = ("This report visualizes the encoded data after applying dimensionality reduction dim_red,"
                     " optionally colored by labels of interest.")

    def check_prerequisites(self):
        return (isinstance(self.dataset.encoded_data, EncodedData) and
                (self.dataset.encoded_data.dimensionality_reduced_data is not None or self._dim_red_method is not None))

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
            data_labels = self.dataset.get_metadata(self._labels, return_df=True)[self._labels]
        except (AttributeError, TypeError) as e:
            logging.warning(f"Labels {self._labels} not found in the dataset. Skipping label coloring in the plot.")

        PathBuilder.build(self.result_path)

        df = pd.DataFrame({'example_id': self.dataset.get_example_ids(),
                           self._dimension_names[0]: dim_reduced_data[:, 0],
                           self._dimension_names[1]: dim_reduced_data[:, 1]})
        if self._labels:
            df[self._labels] = data_labels
        df.to_csv(self.result_path / 'dimensionality_reduced_data.csv', index=False)

        report_output_figures = self._safe_plot(df=df, output_written=True)

        dim_red_text = f" ({self._dim_red_method.__class__.__name__})" if self._dim_red_method else ""

        return ReportResult(name=self.name, info=self.info.replace(" dim_red", dim_red_text),
                            output_figures=report_output_figures,
                            output_tables=[ReportOutput(self.result_path / 'dimensionality_reduced_data.csv',
                                                        'data after dimensionality reduction')])

    def _plot(self, df: pd.DataFrame) -> List[ReportOutput]:
        PathBuilder.build(self.result_path)
        outputs = []
        if self._labels:
            for label in self._labels:
                unique_values = df[label].unique()
                if len(unique_values) <= 3:
                    df[label] = df[label].astype('category')
                    figure = px.scatter(df, x=self._dimension_names[0], y=self._dimension_names[1], color=label,
                                        color_discrete_sequence=px.colors.qualitative.Set1,
                                        hover_data=self._dimension_names + self._labels,
                                        category_orders={label: sorted(unique_values)})
                else:
                    figure = px.scatter(df, x=self._dimension_names[0], y=self._dimension_names[1], color=label,
                                        hover_data=self._dimension_names + self._labels)

                figure.update_layout(template="plotly_white", showlegend=True)
                figure.update_traces(opacity=.6)

                file_path = self.result_path / f"dimensionality_reduction_{label}.html"
                figure.write_html(str(file_path))
                outputs.append(ReportOutput(path=file_path,
                                            name="Data visualization after dimensionality reduction "
                                                 "(highlighted by {})".format(label)))
        else:
            # No label case - just plot points
            figure = px.scatter(df, x=self._dimension_names[0], y=self._dimension_names[1])
            figure.update_layout(template="plotly_white")
            figure.update_traces(opacity=.6)

            file_path = self.result_path / "dimensionality_reduction.html"
            figure.write_html(str(file_path))
            outputs.append(ReportOutput(path=file_path, name="Data visualization after dimensionality reduction"))

        return outputs
