import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly
import plotly.express as px

from immuneML.data_model.EncodedData import EncodedData
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.dsl.definition_parsers.MLParser import MLParser
from immuneML.ml_methods.dim_reduction.DimRedMethod import DimRedMethod
from immuneML.reports.PlotlyUtil import PlotlyUtil
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.encoding_reports.EncodingReport import EncodingReport
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


class DimensionalityReduction(EncodingReport):
    """
    This report visualizes the data obtained by dimensionality reduction. The data points can be highlighted by label of
    interest. It is also possible to specify labels that contain lists of values (e.g., HLA), in which case the data points
    will be duplicated (so that each point refers to one HLA allele) and jittered slightly to improve visibility
    before being highlighted by the concrete HLA allele values.

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
            ParameterValidator.assert_type_and_value(kwargs["labels"], list, location, "labels", nullable=True)
            labels = kwargs["labels"]
            ParameterValidator.assert_all_type_and_value(labels, str, location, "labels")

        return DimensionalityReduction(**{**kwargs, "dim_red_method": method, 'labels': labels})

    def __init__(self, dataset: Dataset = None, batch_size: int = 1, result_path: Path = None,
                 name: str = None, labels: list = None, dim_red_method: DimRedMethod = None):
        super().__init__(dataset=dataset, result_path=result_path, name=name)
        self._labels = labels
        self._dim_red_method = dim_red_method
        self._dimension_names = ['dimension_1', 'dimension_2'] if self._dim_red_method is None \
            else self._dim_red_method.get_dimension_names()
        self.info = ("This report visualizes the encoded data after applying dimensionality reduction dim_red,"
                     " optionally colored by labels of interest.")

    def check_prerequisites(self):
        return (isinstance(self.dataset.encoded_data, EncodedData) and
                (self.dataset.encoded_data.dimensionality_reduced_data is not None or self._dim_red_method is not None))

    def _generate(self) -> ReportResult:

        PathBuilder.build(self.result_path)

        dim_reduced_data = self._get_dim_reduced_data()
        df, report_output_table = self._make_plotting_df(dim_reduced_data)
        report_output_figures = self._safe_plot(df=df, output_written=True)

        dim_red_text = f" ({self._dim_red_method.__class__.__name__})" if self._dim_red_method else ""

        return ReportResult(name=self.name, info=self.info.replace(" dim_red", dim_red_text),
                            output_figures=report_output_figures,
                            output_tables=[report_output_table])

    def _get_dim_reduced_data(self):
        if self._dim_red_method:
            assert self.dataset.encoded_data.examples is not None, \
                f"{DimensionalityReduction.__name__}: data not encoded, report will not be made."
            dim_reduced_data = self._dim_red_method.fit_transform(self.dataset)
        else:
            assert self.dataset.encoded_data.dimensionality_reduced_data is not None
            dim_reduced_data = self.dataset.encoded_data.dimensionality_reduced_data

        assert dim_reduced_data.shape[1] == 2, \
            (f"{DimensionalityReduction.__name__}: {self.name}: dimensionality reduced data is not 2d (got: "
             f"{dim_reduced_data.shape}, so it cannot be plotted.")

        return dim_reduced_data

    def _make_plotting_df(self, dim_reduced_data: np.ndarray) -> Tuple[pd.DataFrame, ReportOutput]:
        df = pd.DataFrame({'example_id': self.dataset.get_example_ids(),
                           self._dimension_names[0]: dim_reduced_data[:, 0],
                           self._dimension_names[1]: dim_reduced_data[:, 1]})

        try:
            if self._labels:
                df[self._labels] = self.dataset.get_metadata(self._labels, return_df=True)[self._labels]
        except (AttributeError, TypeError) as e:
            logging.warning(f"Labels {self._labels} not found in the dataset. Skipping label coloring in the plot.")

        if hasattr(self.dataset, 'get_metadata_fields') and 'subject_id' in self.dataset.get_metadata_fields():
            df['subject_id'] = self.dataset.get_metadata(['subject_id'], return_df=True)['subject_id']

        df.to_csv(self.result_path / 'dimensionality_reduced_data.csv', index=False)
        return df, ReportOutput(self.result_path / 'dimensionality_reduced_data.csv', 'data after dimensionality reduction')

    def _plot(self, df: pd.DataFrame) -> List[ReportOutput]:
        PathBuilder.build(self.result_path)
        outputs = []
        if self._labels:
            for label in self._labels:

                df_copy = self._parse_labels_with_lists(df, label)

                unique_values = df_copy[label].unique()

                hover_data = self._dimension_names + self._labels
                if 'subject_id' in df_copy.columns:
                    hover_data += ['subject_id']
                elif 'example_id' in df_copy.columns:
                    hover_data += ['example_id']

                if len(unique_values) <= 24:
                    color_sequence = px.colors.qualitative.Vivid if len(unique_values) <= 12 else px.colors.qualitative.Dark24
                    df_copy[label] = df_copy[label].astype('category')
                    figure = px.scatter(df_copy, x=self._dimension_names[0], y=self._dimension_names[1], color=label,
                                        color_discrete_sequence=color_sequence,
                                        hover_data=hover_data,
                                        category_orders={label: sorted(unique_values)})
                elif df_copy[label].dtype.name == 'category' or df_copy[label].dtype == object:
                    figure = px.scatter(df_copy, x=self._dimension_names[0], y=self._dimension_names[1], color=label,
                                        hover_data=hover_data,
                                        color_discrete_sequence=plotly.colors.sample_colorscale('Plasma', [i / len(unique_values) for i in range(len(unique_values))]),)
                else:
                    figure = px.scatter(df_copy, x=self._dimension_names[0], y=self._dimension_names[1], color=label,
                                        hover_data=hover_data, color_continuous_scale='Plasma')

                figure.update_layout(template="plotly_white", showlegend=True)
                figure.update_traces(opacity=.6)

                file_path = self.result_path / f"dimensionality_reduction_{label}.html"
                file_path = PlotlyUtil.write_image_to_file(figure, file_path)
                outputs.append(ReportOutput(path=file_path,
                                            name="Data visualization after dimensionality reduction "
                                                 "(highlighted by {})".format(label)))
        else:
            # No label case - just plot points
            figure = px.scatter(df, x=self._dimension_names[0], y=self._dimension_names[1])
            figure.update_layout(template="plotly_white")
            figure.update_traces(opacity=.6)

            file_path = self.result_path / "dimensionality_reduction.html"
            file_path = PlotlyUtil.write_image_to_file(figure, file_path)
            outputs.append(ReportOutput(path=file_path, name="Data visualization after dimensionality reduction"))

        return outputs

    def _parse_labels_with_lists(self, df: pd.DataFrame, label: str) -> pd.DataFrame:
        df_long = df.copy()

        df_long[label] = df_long[label].apply(parse_list_column)

        if any(isinstance(df_long[label].iloc[i], (list, tuple)) for i in range(df_long.shape[0])):
            df_long = df_long.explode(label)

            # Compute jitter based on the range of each axis
            x_range = df_long[self._dimension_names[0]].max() - df_long[self._dimension_names[0]].min()
            y_range = df_long[self._dimension_names[1]].max() - df_long[self._dimension_names[1]].min()
            jitter_strength = 0.005 * min(x_range, y_range)

            # Apply jitter
            df_long[self._dimension_names[0]] += np.random.uniform(-jitter_strength, jitter_strength, size=len(df_long))
            df_long[self._dimension_names[1]] += np.random.uniform(-jitter_strength, jitter_strength, size=len(df_long))

        return df_long


def parse_list_column(value):
    """Parses a string representation of a list or tuple into an actual list."""
    if not value or pd.isna(value):
        return 'unknown'
    if isinstance(value, str):
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (value.startswith('\'') and value.endswith('\'')):
            value = value[1:-1]
            items = [item.strip().replace('\'', '') for item in value.split(',') if item.strip()]
            return items
    return value
