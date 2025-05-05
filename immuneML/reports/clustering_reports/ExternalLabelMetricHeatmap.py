from pathlib import Path
from typing import List

import pandas as pd
import plotly.graph_objects as go

from immuneML.ml_metrics import ClusteringMetric
from immuneML.reports.PlotlyUtil import PlotlyUtil
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.clustering_reports.ClusteringReport import ClusteringReport
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.clustering.ClusteringState import ClusteringState, ClusteringResultPerRun


class ExternalLabelMetricHeatmap(ClusteringReport):
    """
    This report creates heatmaps comparing clustering methods against external labels for each metric.
    For each external label and metric combination, it creates:

    1. A table showing the metric values for each combination of clustering method and external label

    2. A heatmap visualization of these values

    The external labels and metrics are automatically determined from the clustering instruction specification.

    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        reports:
            my_external_label_metric_heatmap: ExternalLabelMetricHeatmap

    """

    @classmethod
    def build_object(cls, **kwargs):
        ParameterValidator.assert_keys(list(kwargs.keys()), ['name'],
                                     ExternalLabelMetricHeatmap.__name__, ExternalLabelMetricHeatmap.__name__)
        return ExternalLabelMetricHeatmap(**kwargs)

    def __init__(self, name: str = None, state: ClusteringState = None,
                 result_path: Path = None, number_of_processes: int = 1):
        super().__init__(name, result_path, number_of_processes, state)
        self.desc = "External Label - Clustering Heatmap"

    def _generate(self) -> ReportResult:
        self.result_path = PathBuilder.build(self.result_path / self.name)
        report_outputs = []
        external_labels = self.state.config.label_config.get_labels_by_name()

        if not external_labels or len(external_labels) == 0:
            return ReportResult(
                name=f"{self.desc} ({self.name})",
                info="No external labels were found in the clustering state's label configuration."
            )

        metrics = [metric for metric in self.state.config.metrics if ClusteringMetric.is_external(metric)]

        for metric in metrics:
            # For each split in the clustering results
            for split_idx, clustering_results in enumerate(self.state.clustering_items):
                # Process discovery results
                if clustering_results.discovery:
                    report_outputs.extend(self._process_analysis_results(
                        clustering_results.discovery,
                        f"discovery_split_{split_idx + 1}",
                        metric
                    ))

                # Process method-based validation results if available
                if clustering_results.method_based_validation:
                    report_outputs.extend(self._process_analysis_results(
                        clustering_results.method_based_validation,
                        f"method_based_validation_split_{split_idx + 1}",
                        metric
                    ))

                # Process result-based validation results if available
                if clustering_results.result_based_validation:
                    report_outputs.extend(self._process_analysis_results(
                        clustering_results.result_based_validation,
                        f"result_based_validation_split_{split_idx + 1}",
                        metric
                    ))

        if not report_outputs:
            return ReportResult(
                name=f"{self.desc} ({self.name})",
                info="No results were generated. This could be because no metrics were computed."
            )

        return ReportResult(
            name=f"{self.desc} ({self.name})",
            info="Heatmaps of metric values for clustering methods versus external labels",
            output_tables=[output for output in report_outputs if output.path.suffix == '.csv'],
            output_figures=[output for output in report_outputs if output.path.suffix in ['.html', '.png']],
        )

    def _process_analysis_results(self, analysis_results: ClusteringResultPerRun, analysis_name: str,
                                  metric: ClusteringMetric) -> List[ReportOutput]:
        outputs = []

        external_labels = self.state.config.label_config.get_labels_by_name()

        # Create a DataFrame with metric values
        df = pd.DataFrame(0, index=list(analysis_results.items.keys()), columns=external_labels)

        for setting_key, item_result in analysis_results.items.items():
            performance_df = item_result.item.external_performance.get_df()
            for label in external_labels:
                df.loc[setting_key, label] = performance_df[performance_df['metric'] == metric][label].values[0]

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=df.values,
            x=df.columns,  # label values
            y=df.index,    # clustering methodsa
            colorscale='Darkmint',
            text=df.values,
            texttemplate='%{text:.3f}',
            hovertemplate=metric + ': %{z:.3f}<br>external label: %{x}<br>clustering setting: %{y}<extra></extra>',
            textfont={"size": 15},
            hoverongaps=False
        ))

        fig.update_layout(
            title=f"{metric.replace('_', ' ')} values by clustering method and external labels",
            template='plotly_white'
        )

        # Save heatmap
        heatmap_path = self.result_path / f"{analysis_name}_{metric}_heatmap.html"
        plot_path = PlotlyUtil.write_image_to_file(fig, heatmap_path, df.shape[0])

        outputs.append(ReportOutput(
            path=plot_path,
            name=f"Heatmap for {metric.replace('_', ' ')} ({analysis_name.replace('_', ' ')})"
        ))

        # Save metric table
        table_path = self.result_path / f"{analysis_name}_{metric}.csv"
        df.reset_index().rename(columns={'index': 'clustering_setting'}).to_csv(table_path, index=False)
        outputs.append(ReportOutput(
            path=table_path,
            name=f"Metric values for {metric} ({analysis_name.replace('_', ' ')})"
        ))

        return outputs

    def check_prerequisites(self):
        if not self.state:
            return False

        if not self.state.clustering_items:
            return False

        if not self.state.config.label_config:
            return False

        return True
