from pathlib import Path
from typing import List

import pandas as pd
import plotly.graph_objects as go

from immuneML.reports.PlotlyUtil import PlotlyUtil
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.clustering_method_reports.ClusteringMethodReport import ClusteringMethodReport
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.clustering.clustering_run_model import ClusteringItem


class ExternalLabelClusterSummary(ClusteringMethodReport):
    """
    This report summarizes the number of examples in a cluster with different values of external labels.
    For each external label, it creates:
    1. A contingency table showing the count of examples for each combination of cluster and label value
    2. A heatmap visualization of these counts

    It can be used in combination with Clustering instruction.

    **Specification arguments:**

    - external_labels (list): the list of metadata columns in the dataset that should be compared against cluster
      assignment

    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        reports:
            my_external_label_cluster_summary:
                ExternalLabelClusterSummary:
                    external_labels: [disease, HLA]

    """

    @classmethod
    def build_object(cls, **kwargs):
        ParameterValidator.assert_keys(list(kwargs.keys()), ['external_labels', 'name'],
                                       ExternalLabelClusterSummary.__name__, ExternalLabelClusterSummary.__name__)
        ParameterValidator.assert_all_type_and_value(kwargs['external_labels'], str,
                                                     ExternalLabelClusterSummary.__name__, 'external_labels')
        return ExternalLabelClusterSummary(**kwargs)

    def __init__(self, external_labels: List[str], name: str = None, item: ClusteringItem = None,
                 result_path: Path = None):
        super().__init__(name=name, result_path=result_path, clustering_item=item)
        self.external_labels = external_labels
        self.desc = "External Label Cluster Summary"

    def _generate(self) -> ReportResult:
        self.result_path = PathBuilder.build(self.result_path / self.name)
        report_outputs = self._process_analysis_results()

        if not report_outputs:
            return ReportResult(
                name=f"{self.desc} ({self.name})",
                info="No results were generated. This could be because no external labels were found in the dataset "
                     "metadata."
            )

        return ReportResult(
            name=f"{self.desc} ({self.name})",
            info="Summary of cluster assignments versus external labels",
            output_tables=[output for output in report_outputs if 'table' in output.name],
            output_figures=[output for output in report_outputs if 'heatmap' in output.name]
        )

    def _process_analysis_results(self) -> List[ReportOutput]:
        outputs = []

        predictions = self.item.predictions
        dataset = self.item.dataset

        # For each external label
        labels = dataset.get_metadata(self.external_labels, return_df=True)
        for label in self.external_labels:
            label_values = labels[label]

            # Create contingency table
            contingency_df = pd.crosstab(
                pd.Series(predictions, name='cluster'),
                pd.Series(label_values, name=label)
            )

            # Save contingency table
            table_path = self.result_path / f"{label}_contingency.csv"
            contingency_df.to_csv(table_path)
            outputs.append(ReportOutput(
                path=table_path,
                name=f"Contingency table for {label} ({self.item.cl_setting.get_key()})"
            ))

            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=contingency_df.values,
                x=contingency_df.columns,
                y=contingency_df.index,
                colorscale='Viridis',
                text=contingency_df.values,
                texttemplate='%{text}',
                hovertemplate='count: %{z}<br>cluster: %{y}<br>' + label + ': %{x}<extra></extra>',
                hoverongaps=False
            ))

            fig.update_layout(
                xaxis_title=label,
                yaxis_title='cluster',
                template='plotly_white'
            )

            fig.update_xaxes(type='category')
            fig.update_yaxes(type='category')

            heatmap_path = self.result_path / f"{label}_heatmap.html"
            plot_path = PlotlyUtil.write_image_to_file(fig, heatmap_path, contingency_df.shape[0])

            outputs.append(ReportOutput(
                path=plot_path,
                name=f"Distribution heatmap for {label} with example counts "
                     f"({dataset.get_example_count()} total examples)"
            ))

        return outputs

    def check_prerequisites(self):
        if self.item is None:
            return False

        if not self.external_labels:
            return False

        return True
