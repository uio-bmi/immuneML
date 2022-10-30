import shutil
from pathlib import Path

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.ml_methods.UnsupervisedMLMethod import UnsupervisedMLMethod
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.ml_reports.UnsupervisedMLReport import UnsupervisedMLReport
from immuneML.util.PathBuilder import PathBuilder

from scipy.sparse import csr_matrix

import plotly.graph_objs as go
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from immuneML.IO.dataset_export.ImmuneMLExporter import ImmuneMLExporter


class ClusteringReport(UnsupervisedMLReport):
    @classmethod
    def build_object(cls, **kwargs):
        name = kwargs["name"] if "name" in kwargs else "ClusteringReport"
        return ClusteringReport(name=name)

    def __init__(self, dataset: Dataset = None, method: UnsupervisedMLMethod = None, result_path: Path = None, name: str = None, number_of_processes: int = 1):
        super().__init__(dataset=dataset, method=method, result_path=result_path,
                         name=name, number_of_processes=number_of_processes)

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

        datasetPath = PathBuilder.build(f'{self.result_path}/{self.dataset.name}_clusterId')
        ImmuneMLExporter.export(self.dataset, datasetPath, False)

        shutil.make_archive(datasetPath, "zip", datasetPath)
        table_paths.append(ReportOutput(self.result_path / f"{self.dataset.name}_clusterId.zip", f"{self.dataset.name} with cluster id"))

        return ReportResult(self.name,
                            output_figures=[p for p in fig_paths if p is not None],
                            output_tables=[p for p in table_paths if p is not None])

    def _2dplot(self, plotting_data, output_name):
        traces = []
        filename = self.result_path / f"{output_name}.html"

        markerText = list(
            "Cluster id: {}<br>Repertoire id: {}".format(self.method.model.labels_[i], self.dataset.encoded_data.example_ids[i]) for i in range(len(self.dataset.encoded_data.example_ids)))
        trace0 = go.Scatter(x=plotting_data[:, 0],
                            y=plotting_data[:, 1],
                            name='Data points',
                            text=markerText,
                            mode='markers',
                            marker=go.scatter.Marker(opacity=1,
                                                     color=self.method.model.labels_),
                            showlegend=True
                            )
        traces.append(trace0)
        if hasattr(self.method.model, "cluster_centers_"):
            trace1 = go.Scatter(x=self.method.model.cluster_centers_[:, 0],
                                y=self.method.model.cluster_centers_[:, 1],
                                name='Cluster centers',
                                text=list("Cluster id: '%s'" % i for i in range(self.method.model.cluster_centers_.shape[0])),
                                mode='markers',
                                marker=go.scatter.Marker(symbol='x',
                                                         size=16,
                                                         line=dict(
                                                             color='DarkSlateGrey',
                                                             width=2
                                                         ),
                                                         color=list(
                                                             range(self.method.model.cluster_centers_.shape[0]))),
                                showlegend=True
                                )
            traces.append(trace1)
        layout = go.Layout(xaxis=go.layout.XAxis(showgrid=False,
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
                           hovermode='closest',
                           template="ggplot2"
                           )
        figure = go.Figure(data=traces, layout=layout)

        with filename.open("w") as file:
            figure.write_html(file)

        return ReportOutput(filename)

    def _3dplot(self, plotting_data, output_name):
        traces = []
        filename = self.result_path / f"{output_name}.html"

        markerText = list(
            "Cluster id: {}<br>Repertoire id: {}".format(self.method.model.labels_[i], self.dataset.encoded_data.example_ids[i]) for i in range(len(self.dataset.encoded_data.example_ids)))
        trace0 = go.Scatter3d(x=plotting_data[:, 0],
                              y=plotting_data[:, 1],
                              z=plotting_data[:, 2],
                              name='Data points',
                              text=markerText,
                              mode='markers',
                              marker=dict(opacity=1,
                                          color=self.method.model.labels_),
                              showlegend=True
                              )
        traces.append(trace0)
        if hasattr(self.method.model, "cluster_centers_"):
            trace1 = go.Scatter3d(x=self.method.model.cluster_centers_[:, 0],
                                  y=self.method.model.cluster_centers_[:, 1],
                                  z=self.method.model.cluster_centers_[:, 2],
                                  name='Cluster centers',
                                  text=list(
                                      "Cluster id: '%s'" % i for i in
                                      range(self.method.model.cluster_centers_.shape[0])),
                                  mode='markers',
                                  marker=dict(symbol='x',
                                              size=12,
                                              line=dict(
                                                  color='DarkSlateGrey',
                                                  width=8
                                              ),
                                              color=list(range(self.method.model.cluster_centers_.shape[0]))),
                                  showlegend=True
                                  )
            traces.append(trace1)

        figure = go.Figure(data=traces)

        with filename.open("w") as file:
            figure.write_html(file)

        return ReportOutput(filename)
