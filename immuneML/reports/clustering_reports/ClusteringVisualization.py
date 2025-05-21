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
from immuneML.reports.clustering_reports.ClusteringReport import ClusteringReport
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.clustering.ClusteringState import ClusteringState
from immuneML.workflows.instructions.clustering.clustering_run_model import ClusteringItem


class ClusteringVisualization(ClusteringReport):
    """
    A report that creates low-dimensional visualizations of clustering results using the specified dimensionality reduction method.
    For each dataset and clustering configuration, it creates a scatter plot where points are colored by their cluster assignments.

    Specification arguments:
        dim_red_method (dict): specification of which dimensionality reduction to perform; valid options are presented
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
                 result_path: Path = None, number_of_processes: int = 1, state: ClusteringState = None):
        super().__init__(name=name, result_path=result_path, number_of_processes=number_of_processes, state=state)
        self.dim_red_method = dim_red_method
        self.result_name = None
        self.desc = "Clustering Visualization"
        self._dimension_names = self.dim_red_method.get_dimension_names()

    @classmethod
    def build_object(cls, **kwargs):
        location = "ClusteringVisualization"
        name = kwargs["name"] if "name" in kwargs else None
        result_path = kwargs["result_path"] if "result_path" in kwargs else None
        number_of_processes = kwargs["number_of_processes"] if "number_of_processes" in kwargs else None
        state = kwargs["state"] if "state" in kwargs else None

        if "dim_red_method" in kwargs and kwargs["dim_red_method"]:
            method_name = list(kwargs["dim_red_method"].keys())[0]
            dim_red_method = MLParser.parse_any_model("dim_red_method", kwargs["dim_red_method"], method_name)[0]
        else:
            raise ValueError(f"{location}: dim_red_method must be specified.")

        return cls(dim_red_method=dim_red_method, name=name, result_path=result_path,
                   number_of_processes=number_of_processes, state=state)

    def _generate(self) -> ReportResult:
        PathBuilder.build(self.result_path)
        self.result_name = f"clustering_{self.dim_red_method.__class__.__name__.lower()}_plots"
        result_path = self.result_path / self.result_name
        PathBuilder.build(result_path)

        report_outputs = []

        for run_idx, clustering_results in enumerate(self.state.clustering_items):
            for analysis_type in ['discovery', 'method_based_validation', 'result_based_validation']:
                run_results = getattr(clustering_results, analysis_type)
                if run_results is not None:
                    for setting_key, item_result in run_results.items.items():

                        plot_path = self._make_plot(item_result.item, run_idx, analysis_type, setting_key, result_path)
                        report_output = ReportOutput(plot_path,
                                                     f"Clustering visualization for {setting_key} "
                                                     f"({analysis_type} - split {run_idx + 1})")
                        report_outputs.append(report_output)

        return ReportResult(f"{self.desc} ({self.name})",
                            info=f"{self.dim_red_method.__class__.__name__} visualizations of clustering results "
                                 f"across splits and discovery/validation datasets",
                            output_figures=report_outputs)

    def _make_plot(self, cl_item: ClusteringItem, run_idx: int, analysis_type: str, setting_key: str, result_path: Path)\
            -> Path:
        transformed_data = self.dim_red_method.fit_transform(cl_item.dataset)
        ext_label_names = self.state.config.label_config.get_labels_by_name()

        df = pd.DataFrame(transformed_data, columns=self._dimension_names)
        df['cluster'] = pd.Series(cl_item.predictions).astype(str)
        df[ext_label_names] = cl_item.dataset.get_metadata(ext_label_names, return_df=True)

        fig = px.scatter(df, x=self._dimension_names[0], y=self._dimension_names[1], color='cluster',
                         color_discrete_sequence=plotly.colors.qualitative.Set2,
                         category_orders={'cluster': sorted(df.cluster.unique())},
                         hover_data=ext_label_names)

        fig.update_layout(template="plotly_white")

        plot_path = PlotlyUtil.write_image_to_file(fig,
                                                   result_path / f"run_{run_idx}_{analysis_type}_{setting_key}.html",
                                                   df.shape[0])

        return plot_path
