import shutil
from pathlib import Path

import numpy as np

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.ml_methods.UnsupervisedMLMethod import UnsupervisedMLMethod
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.ml_reports.UnsupervisedMLReport import UnsupervisedMLReport
from immuneML.util.PathBuilder import PathBuilder

from scipy.sparse import csr_matrix

import plotly.graph_objs as go

from immuneML.IO.dataset_export.AIRRExporter import AIRRExporter


class ClusteringReport(UnsupervisedMLReport):
    @classmethod
    def build_object(cls, **kwargs):
        name = kwargs["name"] if "name" in kwargs else "ClusteringReport"
        labels = kwargs["labels"] if "labels" in kwargs else None
        return ClusteringReport(name=name, labels=labels)

    def __init__(self, dataset: Dataset = None, method: UnsupervisedMLMethod = None, result_path: Path = None, name: str = None, labels: [str] = None, number_of_processes: int = 1):
        super().__init__(dataset=dataset, method=method, result_path=result_path,
                         name=name, number_of_processes=number_of_processes)
        self.labels = labels

    def _generate(self) -> ReportResult:
        # Prepare the result path
        PathBuilder.build(self.result_path)

        # Initialize figure and table paths
        fig_paths = []
        table_paths = []

        # Generate 2D or 3D plots based on dataset
        fig_paths.extend(self._generate_plots())

        # Compare labels
        fig_paths.extend(self._compare_labels())

        # Export dataset with cluster id
        table_paths.extend(self._export_dataset_with_cluster_id())

        # Filter out empty paths and return the report result
        return ReportResult(self.name,
                            output_figures=[p for p in fig_paths if p is not None],
                            output_tables=[p for p in table_paths if p is not None])

    def _generate_plots(self):
        plots = []
        data = self.dataset.encoded_data.examples

        if isinstance(data, csr_matrix):
            data = data.toarray()

        if self.dataset.encoded_data.examples.shape[1] == 2:
            plots.append(self._2dplot(data, f'2d_{self.name}'))
        elif self.dataset.encoded_data.examples.shape[1] == 3:
            plots.append(self._3dplot(data, f'3d_{self.name}'))

        return plots

    def _export_dataset_with_cluster_id(self):
        dataset_path = PathBuilder.build(f'{self.result_path}/{self.dataset.name}_cluster_id')
        AIRRExporter.export(self.dataset, dataset_path)

        shutil.make_archive(dataset_path, "zip", dataset_path)
        return [ReportOutput(self.result_path / f"{self.dataset.name}_cluster_id.zip", f"dataset with cluster id")]

    def _2dplot(self, plotting_data, output_name):
        traces = self._prepare_traces_2d(plotting_data)
        layout = self._prepare_plot_layout("Clustering scatter plot")
        figure = go.Figure(data=traces, layout=layout)

        filename = self.result_path / f"{output_name}.html"
        with filename.open("w") as file:
            figure.write_html(file)

        return ReportOutput(path=filename, name="2d scatter plot")

    def _3dplot(self, plotting_data, output_name):
        traces = self._prepare_traces_3d(plotting_data)
        layout = go.Layout(title="Clustering scatter plot")
        figure = go.Figure(data=traces, layout=layout)

        filename = self.result_path / f"{output_name}.html"
        with filename.open("w") as file:
            figure.write_html(file)

        return ReportOutput(path=filename, name="3d scatter plot")

    def _prepare_traces_2d(self, plotting_data):
        data_grouped_by_cluster = self._group_data_by_cluster(plotting_data)
        traces = []

        for cluster_id in data_grouped_by_cluster:
            data = np.array(list(data_grouped_by_cluster[cluster_id].values()))
            marker_text = self._generate_marker_text(data_grouped_by_cluster, cluster_id)
            trace = go.Scatter(x=data[:, 0],
                               y=data[:, 1],
                               text=marker_text,
                               name=str(cluster_id),
                               mode='markers',
                               marker=go.scatter.Marker(opacity=1,
                                                        color=cluster_id),
                               showlegend=True
                               )
            traces.append(trace)

        return traces

    def _prepare_traces_3d(self, plotting_data):
        data_grouped_by_cluster = self._group_data_by_cluster(plotting_data)
        traces = []

        for cluster_id in data_grouped_by_cluster:
            data = np.array(list(data_grouped_by_cluster[cluster_id].values()))
            marker_text = self._generate_marker_text(data_grouped_by_cluster, cluster_id)
            trace = go.Scatter3d(x=data[:, 0],
                                 y=data[:, 1],
                                 z=data[:, 2],
                                 text=marker_text,
                                 name=str(cluster_id),
                                 mode='markers',
                                 marker=dict(opacity=1,
                                             color=cluster_id),
                                 showlegend=True
                                 )
            traces.append(trace)

        return traces

    def _group_data_by_cluster(self, plotting_data):
        data_grouped_by_cluster = {}

        for index, data in enumerate(plotting_data):
            if self.method.model.labels_[index] not in data_grouped_by_cluster.keys():
                data_grouped_by_cluster.update({self.method.model.labels_[index]: {}})
            data_grouped_by_cluster[self.method.model.labels_[index]].update({self.dataset.encoded_data.example_ids[index]: data})

        return data_grouped_by_cluster

    def _generate_marker_text(self, data_grouped_by_cluster, cluster_id):
        return [f"Cluster id: {cluster_id}<br>Datapoint id: {list(data_grouped_by_cluster[cluster_id].keys())[i]}"
                for i in range(len(data_grouped_by_cluster[cluster_id]))]

    def _prepare_plot_layout(self, title):
        return go.Layout(
            xaxis=go.layout.XAxis(showgrid=False,
                                  zeroline=False,
                                  showline=True,
                                  mirror=True,
                                  linewidth=1,
                                  linecolor='gray',
                                  showticklabels=False),
            yaxis=go.layout.YAxis(showgrid=False,
                                  zeroline=False,
                                  showline=True,
                                  mirror=True,
                                  linewidth=1,
                                  linecolor='black',
                                  showticklabels=False),
            legend_title_text="Cluster Id",
            hovermode='closest',
            template="ggplot2",
            title=title
        )

    def _compare_labels(self):
        fig_paths = []

        for label in self.labels:
            fig = self.compare_labels(label, f'compare_labels{label}')
            fig_paths.append(fig)

        return fig_paths

    def compare_labels(self, label, output_name):
        clusters, total = self._prepare_cluster_and_total_counts(label)
        percentage_data, fig_text = self._prepare_percentage_data_and_fig_text(clusters, total, label)

        fig = self._create_heatmap_figure(label, percentage_data, fig_text)

        filename_fig = self.result_path / f"{output_name}.html"
        with filename_fig.open("w") as file:
            fig.write_html(file)

        return ReportOutput(path=filename_fig, name=f"{label} to cluster_id label comparison")

    def _create_heatmap_figure(self, label, percentage_data, fig_text):
        return go.Figure(
            data=go.Heatmap(
                x=list(self.dataset.labels[label]),
                y=[f'Cluster {str(id)}' for id in self.dataset.labels["cluster_id"]],
                z=percentage_data,
                text=fig_text
            ),
            layout=go.Layout(
                title=f"{label} to cluster_id label comparison",
                xaxis_title=f"{label}"
            )
        )

    def _prepare_cluster_and_total_counts(self, label):
        clusters = {}
        total = {}

        for item in list(self.dataset.get_data()):
            if type(item).__name__ == 'ReceptorSequence':
                label_value = item.metadata.get_attribute(label)
                cluster_id = item.metadata.get_attribute("cluster_id")
            else:
                label_value = item.metadata[label]
                cluster_id = item.metadata["cluster_id"]

            if label_value not in total.keys():
                total[label_value] = 0
            total[label_value] += 1

            if cluster_id not in clusters.keys():
                clusters[cluster_id] = {}
            if label_value in clusters[cluster_id].keys():
                clusters[cluster_id][label_value] += 1
            else:
                clusters[cluster_id][label_value] = 1

        return clusters, total

    def _prepare_percentage_data_and_fig_text(self, clusters, total, label):
        percentage_data = []
        fig_text = []

        for c_id in self.dataset.labels["cluster_id"]:
            percentages = []
            cluster_text = []
            for l in self.dataset.labels[label]:
                if l in clusters[str(c_id)].keys():
                    percentage = clusters[str(c_id)][l] / total[l]
                    percentages.append(percentage)

                    txt = f'{clusters[str(c_id)][l]}/{total[l]}'
                    cluster_text.append(txt)
                else:
                    percentages.append(0)
                    cluster_text.append(f'{str(0)}/{total[l]}')
            percentage_data.append(percentages)
            fig_text.append(cluster_text)

        return percentage_data, fig_text
