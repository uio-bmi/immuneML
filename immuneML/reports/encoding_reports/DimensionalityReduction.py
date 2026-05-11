import logging
from pathlib import Path
from typing import List, Optional, Tuple

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
    This report visualizes the data obtained by dimensionality reduction. The data points can be highlighted by label
    of interest. It is also possible to specify labels that contain lists of values (e.g., HLA), in which case the
    data points will be duplicated (so that each point refers to one HLA allele) and jittered slightly to improve
    visibility before being highlighted by the concrete HLA allele values.

    When a ``dim_red_method`` is configured, its ``components`` parameter determines which two components are plotted
    and overrides the report-level ``components``. When no ``dim_red_method`` is set (using pre-computed
    dimensionality-reduced data), the report-level ``components`` selects which two columns to plot.
    All computed components are always written to the output CSV.

    For PCA the explained variance per component is exported to a separate CSV and annotated on the axis labels.
    For KernelPCA with ``compute_total_variance: true`` the fraction of total kernel-space variance is shown instead.

    **Specification arguments:**

    - labels (list): names of the label to use for highlighting data points; or None

    - components (list): which two components (1-indexed) to plot when no ``dim_red_method`` is provided.
      When a ``dim_red_method`` is set, use its own ``components`` parameter instead. Default: [1, 2].

    - dim_red_method (str): dimensionality reduction method to be used for plotting; if set, in a workflow, this
      dimensionality reduction will be used for plotting instead of any other set in the workflow; if None, it will
      visualize the encoded data of reduced dimensionality if set


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            reports:
                # Plot PC3 vs PC4 from a 5-component PCA, annotated with explained variance
                rep1:
                    DimensionalityReduction:
                        labels: [epitope, source]
                        dim_red_method:
                            PCA:
                                n_components: 5
                                components: [3, 4]

                # Plot components 1 vs 2 from pre-computed dimensionality-reduced data
                rep2:
                    DimensionalityReduction:
                        labels: [epitope]
                        components: [1, 2]

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

        components = kwargs.get('components', None)
        if components is not None:
            assert isinstance(components, list) and len(components) == 2 \
                   and all(isinstance(c, int) and c >= 1 for c in components), \
                (f"{location}: 'components' must be a list of exactly 2 positive integers (1-indexed), "
                 f"e.g. [1, 2]. Got: {components}.")

        return DimensionalityReduction(**{**kwargs, "dim_red_method": method, 'labels': labels})

    def __init__(self, dataset: Dataset = None, batch_size: int = 1, result_path: Path = None,
                 name: str = None, labels: list = None, dim_red_method: DimRedMethod = None,
                 components: list = None):
        super().__init__(dataset=dataset, result_path=result_path, name=name)
        self._labels = labels
        self._dim_red_method = dim_red_method

        # Method-level components take precedence over report-level components
        if dim_red_method is not None and dim_red_method.components is not None:
            self._components = dim_red_method.components
        else:
            self._components = components

        # Column names for the two plotted components; refreshed after fit if n_components was None
        self._dimension_names = self._resolve_dimension_names()

        self.info = ("This report visualizes the encoded data after applying dimensionality reduction dim_red,"
                     " optionally colored by labels of interest.")

    def _resolve_dimension_names(self) -> List[str]:
        c = self._components
        if self._dim_red_method is not None and c is not None:
            try:
                all_names = self._dim_red_method.get_dimension_names()
                return [all_names[c[0] - 1], all_names[c[1] - 1]]
            except (TypeError, IndexError):
                pass
        return [f"dimension_{c[0]}", f"dimension_{c[1]}"] if c else ['dimension_1', 'dimension_2']

    def check_prerequisites(self):
        valid_encoding = self.dataset.encoded_data.encoding not in ['TCRdistEncoder', 'DistanceEncoder']

        return valid_encoding and (isinstance(self.dataset.encoded_data, EncodedData) and
                (self.dataset.encoded_data.dimensionality_reduced_data is not None or self._dim_red_method is not None))

    def _generate(self) -> ReportResult:

        PathBuilder.build(self.result_path)

        dim_reduced_data = self._get_dim_reduced_data()

        # Refresh dimension names now that fitting is complete (handles n_components=None before fit)
        self._dimension_names = self._resolve_dimension_names()

        output_tables = []
        ev_ratio = self._dim_red_method.get_explained_variance_ratio() if self._dim_red_method else None

        if ev_ratio is not None:
            output_tables.append(self._export_explained_variance(ev_ratio))

        df, report_output_table = self._make_plotting_df(dim_reduced_data)
        output_tables.append(report_output_table)
        report_output_figures = self._safe_plot(df=df, ev_ratio=ev_ratio, output_written=True)

        dim_red_text = f" ({self._dim_red_method.__class__.__name__})" if self._dim_red_method else ""

        return ReportResult(name=self.name, info=self.info.replace(" dim_red", dim_red_text),
                            output_figures=report_output_figures,
                            output_tables=output_tables)

    def _get_dim_reduced_data(self) -> np.ndarray:
        if self._dim_red_method:
            assert self.dataset.encoded_data.examples is not None, \
                f"{DimensionalityReduction.__name__}: data not encoded, report will not be made."
            dim_reduced_data = self._dim_red_method.fit_transform(self.dataset)
        else:
            assert self.dataset.encoded_data.dimensionality_reduced_data is not None
            dim_reduced_data = self.dataset.encoded_data.dimensionality_reduced_data

        assert dim_reduced_data.shape[1] >= 2, \
            (f"{DimensionalityReduction.__name__}: {self.name}: dimensionality reduced data must have at least "
             f"2 components for plotting (got shape {dim_reduced_data.shape}).")

        if self._components is not None:
            assert dim_reduced_data.shape[1] >= max(self._components), \
                (f"{DimensionalityReduction.__name__}: {self.name}: requested components {self._components} but "
                 f"the data only has {dim_reduced_data.shape[1]} components. "
                 f"Ensure n_components >= {max(self._components)}.")

        return dim_reduced_data

    def _export_explained_variance(self, ev_ratio: np.ndarray) -> ReportOutput:
        all_dim_names = self._dim_red_method.get_dimension_names()
        ev_df = pd.DataFrame({
            'component': [all_dim_names[i] for i in range(len(ev_ratio))],
            'explained_variance_ratio': ev_ratio,
            'cumulative_explained_variance_ratio': np.cumsum(ev_ratio)
        })
        path = self.result_path / 'explained_variance.csv'
        ev_df.to_csv(path, index=False)
        return ReportOutput(path, f'Explained variance ratio per component '
                                  f'({self._dim_red_method.__class__.__name__})')

    def _make_plotting_df(self, dim_reduced_data: np.ndarray) -> Tuple[pd.DataFrame, ReportOutput]:
        if self._dim_red_method is not None:
            all_dim_names = self._dim_red_method.get_dimension_names()
        else:
            all_dim_names = [f"dimension_{i + 1}" for i in range(dim_reduced_data.shape[1])]

        component_cols = {all_dim_names[i]: dim_reduced_data[:, i] for i in range(dim_reduced_data.shape[1])}
        df = pd.DataFrame({'example_id': self.dataset.get_example_ids(), **component_cols})

        try:
            if self._labels:
                df[self._labels] = self.dataset.get_metadata(self._labels, return_df=True)[self._labels]
        except (AttributeError, TypeError):
            logging.warning(f"Labels {self._labels} not found in the dataset. Skipping label coloring in the plot.")

        if hasattr(self.dataset, 'get_metadata_fields') and 'subject_id' in self.dataset.get_metadata_fields():
            df['subject_id'] = self.dataset.get_metadata(['subject_id'], return_df=True)['subject_id']

        df.to_csv(self.result_path / 'dimensionality_reduced_data.csv', index=False)
        return df, ReportOutput(self.result_path / 'dimensionality_reduced_data.csv',
                                'data after dimensionality reduction')

    def _build_axis_label_map(self, ev_ratio: Optional[np.ndarray]) -> dict:
        label_map = {}
        for i, col in enumerate(self._dimension_names):
            comp_idx = self._components[i] if self._components else i + 1
            if ev_ratio is not None and comp_idx <= len(ev_ratio):
                label_map[col] = f"{col} ({ev_ratio[comp_idx - 1] * 100:.2f}%)"
            else:
                label_map[col] = col
        return label_map

    def _plot(self, df: pd.DataFrame, ev_ratio: Optional[np.ndarray] = None) -> List[ReportOutput]:
        PathBuilder.build(self.result_path)
        label_map = self._build_axis_label_map(ev_ratio)
        x, y = self._dimension_names[0], self._dimension_names[1]

        outputs = []
        if self._labels:
            for label in self._labels:

                df_copy = self._parse_labels_with_lists(df, label)
                unique_values = df_copy[label].unique()

                hover_data = list(self._dimension_names) + list(self._labels)
                if 'subject_id' in df_copy.columns:
                    hover_data += ['subject_id']
                elif 'example_id' in df_copy.columns:
                    hover_data += ['example_id']

                if len(unique_values) <= 24:
                    color_sequence = px.colors.qualitative.Vivid if len(unique_values) <= 12 else px.colors.qualitative.Dark24
                    df_copy[label] = df_copy[label].astype('category')
                    figure = px.scatter(df_copy, x=x, y=y, color=label,
                                        color_discrete_sequence=color_sequence,
                                        hover_data=hover_data, labels=label_map,
                                        category_orders={label: sorted(unique_values)})
                elif df_copy[label].dtype.name == 'category' or df_copy[label].dtype == object:
                    figure = px.scatter(df_copy, x=x, y=y, color=label,
                                        hover_data=hover_data, labels=label_map,
                                        color_discrete_sequence=plotly.colors.sample_colorscale(
                                            'Plasma', [i / len(unique_values) for i in range(len(unique_values))]))
                else:
                    figure = px.scatter(df_copy, x=x, y=y, color=label,
                                        hover_data=hover_data, labels=label_map,
                                        color_continuous_scale='Plasma')

                figure.update_layout(template="plotly_white", showlegend=True)
                figure.update_traces(opacity=.6)

                file_path = self.result_path / f"dimensionality_reduction_{label}.html"
                file_path = PlotlyUtil.write_image_to_file(figure, file_path)
                outputs.append(ReportOutput(path=file_path,
                                            name=f"Data visualization after dimensionality reduction "
                                                 f"(highlighted by {label})"))
        else:
            figure = px.scatter(df, x=x, y=y, labels=label_map)
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

            x_range = df_long[self._dimension_names[0]].max() - df_long[self._dimension_names[0]].min()
            y_range = df_long[self._dimension_names[1]].max() - df_long[self._dimension_names[1]].min()
            jitter_strength = 0.005 * min(x_range, y_range)

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