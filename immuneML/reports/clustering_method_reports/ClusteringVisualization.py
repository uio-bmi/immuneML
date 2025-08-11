from pathlib import Path

import pandas as pd
import plotly
import plotly.express as px

from immuneML.dsl.definition_parsers.MLParser import MLParser
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.ml_methods.dim_reduction.DimRedMethod import DimRedMethod
from immuneML.reports.PlotlyUtil import PlotlyUtil
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.clustering_method_reports.ClusteringMethodReport import ClusteringMethodReport
from immuneML.reports.clustering_reports.ClusteringReport import ClusteringReport
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.clustering.ClusteringState import ClusteringState
from immuneML.workflows.instructions.clustering.clustering_run_model import ClusteringItem


class ClusteringVisualization(ClusteringMethodReport):
    """
    A report that creates low-dimensional visualizations of clustering results using the specified dimensionality reduction method.
    For each dataset and clustering configuration, it creates a scatter plot where points are colored by their cluster assignments.

    Specification arguments:

        - dim_red_method (dict): specification of which dimensionality reduction to perform; valid options are presented
          under :ref:`**Dimensionality reduction methods**` and should be specified with the name of the method and its
          parameters, see the example below

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
    """

    def __init__(self, dim_red_method: DimRedMethod = None, name: str = None,
                 result_path: Path = None, clustering_item: ClusteringItem = None):
        super().__init__(name=name, result_path=result_path, clustering_item=clustering_item)
        self.dim_red_method = dim_red_method
        self.result_name = None
        self.desc = "Clustering Visualization"
        self._dimension_names = self.dim_red_method.get_dimension_names()

    @classmethod
    def build_object(cls, **kwargs):
        location = "ClusteringVisualization"
        name = kwargs["name"] if "name" in kwargs else None
        result_path = kwargs["result_path"] if "result_path" in kwargs else None

        if "dim_red_method" in kwargs and kwargs["dim_red_method"]:
            method_name = list(kwargs["dim_red_method"].keys())[0]
            dim_red_method = MLParser.parse_any_model("dim_red_method", kwargs["dim_red_method"], method_name)[0]
        else:
            raise ValueError(f"{location}: dim_red_method must be specified.")

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
                            info=f"{self.dim_red_method.__class__.__name__} visualizations of clustering results",
                            output_figures=[report_output])

    def _make_plot(self, result_path: Path) -> Path:
        transformed_data = self.dim_red_method.fit_transform(dataset=self.item.dataset)

        df = pd.DataFrame(transformed_data, columns=self._dimension_names)
        df['cluster'] = pd.Series(self.item.predictions).astype(str)
        df['id'] = self.item.dataset.get_example_ids()

        fig = px.scatter(df, x=self._dimension_names[0], y=self._dimension_names[1], color='cluster',
                         color_discrete_sequence=plotly.colors.qualitative.Set2,
                         category_orders={'cluster': sorted(df.cluster.unique())},
                         hover_data=['id'])

        fig.update_layout(template="plotly_white")

        df.to_csv(result_path / f"clustering_visualization_{self.dim_red_method.name}.csv", index=False)

        plot_path = PlotlyUtil.write_image_to_file(fig,
                                                   result_path / f"clustering_visualization_{self.dim_red_method.name}.html",
                                                   df.shape[0])

        return plot_path
