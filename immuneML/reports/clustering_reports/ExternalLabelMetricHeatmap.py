from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from immuneML.ml_metrics import ClusteringMetric
from immuneML.reports.PlotlyUtil import PlotlyUtil
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.clustering_reports.ClusteringReport import ClusteringReport
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.clustering.ClusteringState import ClusteringState


class ExternalLabelMetricHeatmap(ClusteringReport):
    """
    This report creates heatmaps comparing clustering methods against external labels for each metric.
    For each external label and metric combination, it creates:

    1. A table showing the mean and standard deviation of metric values across splits for each
       combination of clustering method and external label

    2. A heatmap visualization where the color represents the mean value and the text shows mean±std

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
            report_outputs.extend(self._process_metric_across_splits(metric, external_labels))

        if not report_outputs:
            return ReportResult(
                name=f"{self.desc} ({self.name})",
                info="No results were generated. This could be because no metrics were computed."
            )

        return ReportResult(
            name=f"{self.desc} ({self.name})",
            info="Heatmaps of metric values (mean±std across splits) for clustering methods versus external labels",
            output_tables=[output for output in report_outputs if output.path.suffix == '.csv'],
            output_figures=[output for output in report_outputs if output.path.suffix in ['.html', '.png']],
        )

    def _process_metric_across_splits(self, metric: str, external_labels: List[str]) -> List[ReportOutput]:
        outputs = []

        # Get setting keys from the first split
        setting_keys = list(self.state.clustering_items[0].items.keys())

        # Collect metric values across all splits
        # Shape: (n_splits, n_settings, n_labels)
        all_values = []

        for clustering_results in self.state.clustering_items:
            split_values = []
            for setting_key in setting_keys:
                item_result = clustering_results.items[setting_key]
                performance_df = item_result.item.external_performance.get_df()
                label_values = []
                for label in external_labels:
                    value = performance_df[performance_df['metric'] == metric][label].values[0]
                    label_values.append(value)
                split_values.append(label_values)
            all_values.append(split_values)

        all_values = np.array(all_values)  # (n_splits, n_settings, n_labels)

        # Calculate mean and std across splits (axis=0)
        mean_values = np.mean(all_values, axis=0)  # (n_settings, n_labels)
        std_values = np.std(all_values, axis=0)    # (n_settings, n_labels)

        # Create text annotations with mean±std format
        text_annotations = np.empty(mean_values.shape, dtype=object)
        for i in range(mean_values.shape[0]):
            for j in range(mean_values.shape[1]):
                text_annotations[i, j] = f"{mean_values[i, j]:.3f}±{std_values[i, j]:.3f}"

        # Wrap long y-axis labels (clustering setting names) for better display
        wrapped_setting_keys = [_wrap_label(key) for key in setting_keys]

        # Calculate dynamic figure dimensions based on data size
        n_settings = len(setting_keys)
        n_labels = len(external_labels)

        # Base dimensions with scaling factors
        row_height = 40  # pixels per row
        col_width = 100  # pixels per column
        min_height, max_height = 400, 1200
        min_width, max_width = 600, 1600

        # Calculate margins based on label lengths
        max_y_label_lines = max(key.count('<br>') + 1 for key in wrapped_setting_keys)
        left_margin = 150 + (max_y_label_lines - 1) * 30

        fig_height = max(min_height, min(max_height, n_settings * row_height + 150))
        fig_width = max(min_width, min(max_width, n_labels * col_width + left_margin + 100))

        # Create heatmap with mean as color and mean±std as text
        # Note on z, x, y mapping: z[i][j] corresponds to (y[i], x[j])
        # mean_values shape is (n_settings, n_labels), so this is correct
        fig = go.Figure(data=go.Heatmap(
            z=mean_values,
            x=external_labels,
            y=wrapped_setting_keys,
            colorscale='Darkmint',
            text=text_annotations,
            texttemplate='%{text}',
            hovertemplate=(metric + ' (mean): %{z:.3f}<br>external label: %{x}<br>'
                          'clustering setting: %{y}<extra></extra>'),
            hoverongaps=False
        ))

        # Adjust text font size based on number of cells
        total_cells = n_settings * n_labels
        if total_cells > 50:
            text_font_size = 9
        elif total_cells > 30:
            text_font_size = 10
        else:
            text_font_size = 11

        fig.update_traces(textfont=dict(size=text_font_size))

        fig.update_layout(
            template='plotly_white',
            width=fig_width,
            height=fig_height,
            margin=dict(l=left_margin, r=50, t=50, b=80),
            yaxis=dict(
                tickfont=dict(size=10),
                automargin=True
            ),
            xaxis=dict(
                tickfont=dict(size=10),
                tickangle=-45 if n_labels > 5 else 0,
                automargin=True
            )
        )

        # Save heatmap
        heatmap_path = self.result_path / f"{metric}_heatmap.html"
        plot_path = PlotlyUtil.write_image_to_file(fig, heatmap_path, len(setting_keys))

        outputs.append(ReportOutput(
            path=plot_path,
            name=f"Heatmap for {metric.replace('_', ' ')} (mean±std across splits)"
        ))

        # Create a combined table with mean±std format (use original keys for CSV)
        df_combined = pd.DataFrame(text_annotations, index=setting_keys, columns=external_labels)
        df_combined = df_combined.reset_index().rename(columns={'index': 'clustering_setting'})

        table_path = self.result_path / f"{metric}_mean_std.csv"
        df_combined.to_csv(table_path, index=False)
        outputs.append(ReportOutput(
            path=table_path,
            name=f"Metric values (mean±std) for {metric}"
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
    
def _wrap_label(label: str, max_chars_per_line: int = 15) -> str:
    """
    Wrap a label by splitting on underscores to create multi-line text.
    Attempts to keep each line under max_chars_per_line characters.
    """
    if len(label) <= max_chars_per_line:
        return label

    parts = label.split('_')
    lines = []
    current_line = ""

    for part in parts:
        if not current_line:
            current_line = part
        elif len(current_line) + 1 + len(part) <= max_chars_per_line:
            current_line += "_" + part
        else:
            lines.append(current_line)
            current_line = part

    if current_line:
        lines.append(current_line)

    return "<br>".join(lines)
