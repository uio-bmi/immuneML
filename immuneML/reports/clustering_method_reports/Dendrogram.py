from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import squareform

from immuneML.ml_methods.clustering.AgglomerativeClustering import AgglomerativeClustering
from immuneML.reports.PlotlyUtil import PlotlyUtil
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.clustering_method_reports.ClusteringMethodReport import ClusteringMethodReport
from immuneML.workflows.instructions.clustering.clustering_run_model import ClusteringItem


class Dendrogram(ClusteringMethodReport):
    """
    This report generates a dendrogram visualization from the AgglomerativeClustering method and shows the external
    labels as annotations.

    **Specification arguments:**

    - labels (list): List of metadata labels to annotate on the dendrogram.

    **YAML specification:**

    .. code-block:: yaml

        reports:
            my_dendrogram_report:
                Dendrogram:
                    labels:
                        - disease_status
                        - age_group

    """

    def __init__(self, labels: list, result_path: Path = None, name: str = None,
                 clustering_item: ClusteringItem = None):
        super().__init__(name=name, result_path=result_path)
        self.item = clustering_item
        self.labels = labels

    @classmethod
    def build_object(cls, **kwargs):
        return cls(**kwargs)

    def check_prerequisites(self) -> bool:
        return (isinstance(self.item.method, AgglomerativeClustering) and
                self.item.method.model.distance_threshold == 0 and self.item.method.model.n_clusters is None)

    def _get_linkage_matrix(self):
        counts = np.zeros(self.item.method.model.children_.shape[0])
        n_samples = len(self.item.method.model.labels_)
        for i, merge in enumerate(self.item.method.model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack(
            [self.item.method.model.children_, self.item.method.model.distances_, counts]
        ).astype(float)

        return linkage_matrix

    def _generate(self) -> ReportResult:

        # Generate output path
        output_path = Path(self.result_path) / "dendrogram.html"

        self._create_full_dendrogram(output_path)

        return ReportResult(output_figures=[ReportOutput(output_path, "Dendrogram visualization")])

    def _save_data(self, linkage_matrix, metadata):
        np.save(f"{self.result_path}/linkage_matrix.npy", linkage_matrix)
        metadata.reset_index().to_csv(f"{self.result_path}/metadata.csv", index=False)
        np.save(f"{self.result_path}/distances.npy", self.item.method.model.distances_)

    def _add_annotations(self, fig, annotation_start, annotation_height, metadata, ordered_ids):
        for i, label in enumerate(self.labels):
            yaxis_name = f'y{i + 3}'
            y_domain = [
                annotation_start + i * annotation_height,
                annotation_start + i * annotation_height + annotation_height
            ]

            annotation_values = [metadata.loc[ind, label] for ind in ordered_ids]
            unique_labels = sorted(metadata[label].unique().tolist())
            label_to_num = {label_val: j for j, label_val in enumerate(unique_labels)}
            annotation_numeric = [label_to_num[v] for v in annotation_values]

            # Add annotation heatmap
            fig.add_trace(go.Heatmap(
                x=fig['layout']['xaxis']['tickvals'],
                y=[label],
                z=np.array(annotation_numeric).reshape(1, -1),
                showlegend=False,
                showscale=False,
                colorscale=px.colors.qualitative.Plotly[1:],
                hovertemplate=label + ': %{text}<br>Example: %{x}<extra></extra>',
                text=[annotation_values],
                yaxis=yaxis_name
            ))

            # Update layout for this axis
            fig.update_layout(**{
                f'yaxis{i + 3}': dict(
                    domain=y_domain,
                    mirror=False,
                    showgrid=False,
                    showline=False,
                    zeroline=False
                )
            })

        return fig

    def _update_layout(self, fig, example_ids):

        fig.update_layout({
            'height': 700 + len(example_ids) / 10 + len(self.labels) * 20,
            # Adjust height based on number of labels and number of examples
            'autosize': True,
            'hovermode': 'closest',
            'template': 'plotly_white',
            'xaxis': {
                'domain': [0.15, 1],
                'mirror': False,
                'showgrid': False,
                'showline': False,
                'zeroline': False,
                'ticks': "",
                'showticklabels': False
            },
            'xaxis2': {
                'domain': [0, 0.145],  # Side dendrogram
                'mirror': False,
                'showgrid': False,
                'showline': False,
                'zeroline': False,
                'ticks': "",
                'showticklabels': False
            },
            'yaxis': {
                'domain': [0, 0.65],  # Reduced height for main heatmap
                'mirror': False,
                'showgrid': False,
                'showline': False,
                'zeroline': False,
                'showticklabels': False,
                'ticks': ""
            },
            'yaxis2': {
                'domain': [0.8, 1],  # Top dendrogram
                'mirror': False,
                'showgrid': False,
                'showline': False,
                'zeroline': False,
                'showticklabels': False,
                'ticks': ""
            }
        })

        return fig

    def _create_full_dendrogram(self, output_path):
        linkage_matrix = self._get_linkage_matrix()
        example_ids = self.item.dataset.get_example_ids()
        metadata = self.item.dataset.get_metadata(self.labels, return_df=True)
        metadata['example_id'] = example_ids
        metadata = metadata.set_index('example_id')

        self._save_data(linkage_matrix, metadata)

        fig, dendro_leaves, dendro_side = self._make_dendrograms(example_ids, linkage_matrix)
        fig, ordered_ids = self._add_distance_heatmap(example_ids, linkage_matrix, dendro_leaves, dendro_side, fig)

        annotation_start, annotation_end = 0.65, 0.8
        annotation_height = (annotation_end - annotation_start) / len(self.labels)

        fig = self._add_annotations(fig, annotation_start, annotation_height, metadata, ordered_ids)
        fig = self._update_layout(fig, example_ids)

        fig_path = PlotlyUtil.write_image_to_file(fig, output_path, len(example_ids))

        return fig_path

    def _add_distance_heatmap(self, example_ids, linkage_matrix, dendro_leaves, dendro_side, fig):
        ordered_ids = [example_ids[idx] for idx in dendro_leaves]
        heat_data = squareform(cophenet(linkage_matrix))
        heat_data = heat_data[dendro_leaves, :]
        heat_data = heat_data[:, dendro_leaves]

        heatmap = go.Heatmap(x=dendro_leaves, y=dendro_leaves, z=heat_data, colorscale=px.colors.sequential.Blues,
                             showlegend=False, showscale=False,
                             hovertemplate='Distance: %{z}<extra></extra>')

        heatmap['x'] = fig['layout']['xaxis']['tickvals']
        heatmap['y'] = dendro_side['layout']['yaxis']['tickvals']

        fig.add_trace(heatmap)

        return fig, ordered_ids


    def _make_dendrograms(self, example_ids, linkage_matrix):

        fig = ff.create_dendrogram(np.zeros(len(example_ids)), orientation='bottom',
                                   colorscale=['#7393B3' for _ in range(8)],
                                   labels=example_ids, linkagefun=lambda x: linkage_matrix,
                                   distfun=lambda x: linkage_matrix[:, 2])

        for i in range(len(fig['data'])):
            fig['data'][i]['yaxis'] = 'y2'
            fig['data'][i]['hovertemplate'] = None
            fig['data'][i]['showlegend'] = False

        dendro_side = ff.create_dendrogram(np.zeros(len(example_ids)), orientation='right',
                                           linkagefun=lambda x: linkage_matrix, distfun=lambda x: linkage_matrix[:, 2],
                                           colorscale=['#7393B3' for _ in range(8)])
        for i in range(len(dendro_side['data'])):
            dendro_side['data'][i]['xaxis'] = 'x2'
            dendro_side['data'][i]['hovertemplate'] = None
            dendro_side['data'][i]['showlegend'] = False

        for data in dendro_side['data']:
            fig.add_trace(data)

        dendro_leaves = dendro_side['layout']['yaxis']['ticktext']
        dendro_leaves = list(map(int, dendro_leaves))

        return fig, dendro_leaves, dendro_side
