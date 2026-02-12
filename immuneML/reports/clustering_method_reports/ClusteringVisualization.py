import logging
from pathlib import Path

import pandas as pd
import plotly
import plotly.express as px

from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.dsl.definition_parsers.MLParser import MLParser
from immuneML.ml_methods.dim_reduction.DimRedMethod import DimRedMethod
from immuneML.reports.PlotlyUtil import PlotlyUtil
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.clustering_method_reports.ClusteringMethodReport import ClusteringMethodReport
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.clustering.clustering_run_model import ClusteringItem


class ClusteringVisualization(ClusteringMethodReport):
    """
    A report that creates low-dimensional visualizations of clustering results using the specified dimensionality reduction method.
    For each dataset and clustering configuration, it creates a scatter plot where points are colored by their cluster assignments.

    Specification arguments:

        - dim_red_method (dict): specification of which dimensionality reduction to perform; valid options are presented
          under :ref:`**Dimensionality reduction methods**` and should be specified with the name of the method and its
          parameters, see the example below; if not specified, the report will use any dimensionality reduced data
          present in the dataset's encoded data; if the dataset does not contain dimensionality reduced data, and the
          encoded data has more than 2 dimensions, the report will be skipped.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        reports:
            my_report_with_pca:
                ClusteringVisualization:
                    dim_red_method:
                        PCA:
                            n_components: 2
            my_report_with_tsne:
                ClusteringVisualization:
                    dim_red_method:
                        TSNE:
                            n_components: 2
                            init: pca
            my_report_existing_dim_red:
                ClusteringVisualization:
                    dim_red_method: null

    """

    def __init__(self, dim_red_method: DimRedMethod = None, name: str = None,
                 result_path: Path = None, clustering_item: ClusteringItem = None):
        super().__init__(name=name, result_path=result_path, clustering_item=clustering_item)
        self.dim_red_method = dim_red_method
        self.result_name = None
        self.desc = "Clustering Visualization"
        self._dimension_names = self.dim_red_method.get_dimension_names() if self.dim_red_method else None

    @classmethod
    def build_object(cls, **kwargs):
        location = "ClusteringVisualization"
        name = kwargs["name"] if "name" in kwargs else None
        result_path = kwargs["result_path"] if "result_path" in kwargs else None

        if "dim_red_method" in kwargs and kwargs["dim_red_method"]:
            method_name = list(kwargs["dim_red_method"].keys())[0]
            dim_red_method = MLParser.parse_any_model("dim_red_method", kwargs["dim_red_method"], method_name)[0]
        else:
            logging.warning(f"{location}: No dimensionality reduction method specified. "
                            "If the encoded dataset includes dimensionality reduction, it will be used.")
            dim_red_method = None

        return cls(dim_red_method=dim_red_method, name=name, result_path=result_path,
                   clustering_item=kwargs['clustering_item'] if 'clustering_item' in kwargs else None,)

    def _generate(self) -> ReportResult:
        PathBuilder.build(self.result_path)
        self.result_name = f"clustering_{self.dim_red_method.__class__.__name__.lower()}_plots"
        result_path = PathBuilder.build(self.result_path / self.result_name)

        plot_path = self._make_plot(result_path)
        report_output = ReportOutput(plot_path,
                                     f"Clustering visualization for {self.item.cl_setting.get_key()}")

        return ReportResult(f"{self.desc} ({self.name})",
                            info=f"Visualizations of clustering results using "
                                 f"{self.dim_red_method.__class__.__name__ if self.dim_red_method else 'encoded data directly'}.",
                            output_figures=[report_output])

    def _make_plot(self, result_path: Path) -> Path:
        if self.dim_red_method is not None:
            transformed_data = self.dim_red_method.fit_transform(dataset=self.item.dataset)
        elif self.item.dataset.encoded_data.dimensionality_reduced_data is not None:
            transformed_data = self.item.dataset.encoded_data.dimensionality_reduced_data
            self._dimension_names = self.item.dataset.encoded_data.dim_names if self.item.dataset.encoded_data.dim_names else ['dim1', 'dim2']
            self.dim_red_method = self.item.dim_red_method
        elif self.item.dataset.encoded_data.examples.shape[1] <= 2:
            transformed_data = self.item.dataset.encoded_data.get_examples_as_np_matrix()
            self._dimension_names = self.item.dataset.encoded_data.feature_names
            self.dim_red_method = None
        else:
            raise ValueError("ClusteringVisualization: No dimensionality reduction method specified, and the dataset "
                             "does not contain dimensionality reduced data. Please specify a dimensionality reduction "
                             "method.")

        df = pd.DataFrame(transformed_data, columns=self._dimension_names)
        df['cluster'] = pd.Series(self.item.predictions).astype(str)
        df['id'] = self.item.dataset.get_example_ids()

        unique_clusters = sorted(df.cluster.astype(int).unique())
        color_palette = self.get_color_palette(len(unique_clusters))
        fig = px.scatter(df, x=self._dimension_names[0], y=self._dimension_names[1], color='cluster',
                         color_discrete_sequence=color_palette,
                         category_orders={'cluster': [str(c) for c in unique_clusters]},
                         hover_data=['id'])

        fig.update_layout(template="plotly_white")

        df.to_csv(result_path / f"clustering_visualization_{self.dim_red_method.name if self.dim_red_method else ''}.csv", index=False)

        plot_path = PlotlyUtil.write_image_to_file(fig,
                                                   result_path / f"clustering_visualization_{self.dim_red_method.name if self.dim_red_method else ''}.html",
                                                   df.shape[0])

        return plot_path

    def get_color_palette(self, n_clusters):
        if n_clusters <= 10:
            return px.colors.qualitative.Vivid
        elif n_clusters <= 24:
            return px.colors.qualitative.Dark24
        else:
            logging.warning(f"ClusteringVisualization: number of clusters is {n_clusters}, which is commonly too many to "
                            f"visualize effectively.")
            return plotly.colors.sample_colorscale('Plasma', [i / n_clusters for i in range(n_clusters)])

    def get_ids(self):
        if isinstance(self.item.dataset, RepertoireDataset):
            metadata = self.item.dataset.get_metadata(['subject_id'], return_df=True)
            if 'subject_id' in metadata.columns:
                return metadata['subject_id'].tolist()
            else:
                return self.item.dataset.get_example_ids()
        else:
            return self.item.dataset.get_example_ids()

    def check_prerequisites(self) -> bool:
        """The results cannot be visualized in this report if the encoded data is precomputed distances"""

        from immuneML.encodings.distance_encoding.DistanceEncoder import DistanceEncoder
        from immuneML.encodings.distance_encoding.TCRdistEncoder import TCRdistEncoder

        return not isinstance(self.item.encoder, TCRdistEncoder) and not isinstance(self.item.encoder, DistanceEncoder)
