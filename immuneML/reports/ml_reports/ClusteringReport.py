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
        PathBuilder.build(self.result_path)
        fig_paths = []
        table_paths = []
        data = self.dataset.encoded_data.examples

        if isinstance(data, csr_matrix):
            data = data.toarray()
        if self.dataset.encoded_data.examples.shape[1] == 2:
            fig_paths.append(self._2dplot(data, f'2d_{self.name}'))
        elif self.dataset.encoded_data.examples.shape[1] == 3:
            fig_paths.append(self._3dplot(data, f'3d_{self.name}'))

        for label in self.labels:
            f, t = self._label_comparison(label, f'label_comaprison_{label}')
            fig_paths.append(f)
            # table_paths.append(t)

        datasetPath = PathBuilder.build(f'{self.result_path}/{self.dataset.name}_cluster_id')
        AIRRExporter.export(self.dataset, datasetPath)

        shutil.make_archive(datasetPath, "zip", datasetPath)
        table_paths.append(ReportOutput(self.result_path / f"{self.dataset.name}_cluster_id.zip", f"dataset with cluster id"))

        return ReportResult(self.name,
                            output_figures=[p for p in fig_paths if p is not None],
                            output_tables=[p for p in table_paths if p is not None])

    def _2dplot(self, plotting_data, output_name):
        traces = []
        filename = self.result_path / f"{output_name}.html"

        data_grouped_by_cluster = {}
        for index, data in enumerate(plotting_data):
            if self.method.model.labels_[index] not in data_grouped_by_cluster.keys():
                data_grouped_by_cluster.update({self.method.model.labels_[index]: {}})
            data_grouped_by_cluster[self.method.model.labels_[index]].update({self.dataset.encoded_data.example_ids[index]: data})

        for cluster_id in data_grouped_by_cluster:
            data = np.array(list(data_grouped_by_cluster[cluster_id].values()))
            markerText = list("Cluster id: {}<br>Datapoint id: {}".format(cluster_id, list(data_grouped_by_cluster[cluster_id].keys())[i])
                              for i in range(len(data)))
            trace = go.Scatter(x=data[:, 0],
                               y=data[:, 1],
                               text=markerText,
                               name=str(cluster_id),
                               mode='markers',
                               marker=go.scatter.Marker(opacity=1,
                                                        color=cluster_id),
                               showlegend=True
                               )
            traces.append(trace)

        layout = go.Layout(
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
            title=f"Clustering scatter plot"
        )
        figure = go.Figure(
            data=traces,
            layout=layout
        )

        with filename.open("w") as file:
            figure.write_html(file)

        return ReportOutput(path=filename, name="2d scatter plot")

    def _3dplot(self, plotting_data, output_name):
        traces = []
        filename = self.result_path / f"{output_name}.html"

        data_grouped_by_cluster = {}
        for index, data in enumerate(plotting_data):
            if self.method.model.labels_[index] not in data_grouped_by_cluster.keys():
                data_grouped_by_cluster.update({self.method.model.labels_[index]: {}})
            data_grouped_by_cluster[self.method.model.labels_[index]].update({self.dataset.encoded_data.example_ids[index]: data})

        for cluster_id in data_grouped_by_cluster:
            data = np.array(list(data_grouped_by_cluster[cluster_id].values()))
            markerText = list("Cluster id: {}<br>Datapoint id: {}".format(cluster_id, list(data_grouped_by_cluster[cluster_id].keys())[i])
                              for i in range(len(data)))
            trace = go.Scatter3d(x=data[:, 0],
                                 y=data[:, 1],
                                 z=data[:, 2],
                                 text=markerText,
                                 name=str(cluster_id),
                                 mode='markers',
                                 marker=dict(opacity=1,
                                             color=cluster_id),
                                 showlegend=True
                                 )
            traces.append(trace)

        figure = go.Figure(
            data=traces,
            layout=go.Layout(title=f"Clustering scatter plot")
        )

        with filename.open("w") as file:
            figure.write_html(file)

        return ReportOutput(path=filename, name="3d scatter plot")

    def _label_comparison(self, label, output_name):
        filenameFig = self.result_path / f"{output_name}.html"
        filenameTbl = self.result_path / f"{output_name}.csv"

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

        fig = go.Figure(
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

        with filenameFig.open("w") as file:
            fig.write_html(file)

        # with filenameTbl.open("w") as file:
        #     tbl.write_html(file)

        return ReportOutput(path=filenameFig, name=f"{label} to cluster_id label comparison"), ReportOutput(filenameTbl)
