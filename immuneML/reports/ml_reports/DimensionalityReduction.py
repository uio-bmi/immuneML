from pathlib import Path
from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.ml_methods.UnsupervisedMLMethod import UnsupervisedMLMethod
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.ml_reports.UnsupervisedMLReport import UnsupervisedMLReport
from immuneML.util.PathBuilder import PathBuilder

from scipy.sparse import csr_matrix
import numpy as np
import plotly.graph_objs as go
import plotly as plt
from sklearn.preprocessing import StandardScaler

class DimensionalityReduction(UnsupervisedMLReport):
    """
    This is a report that visualizes the results of applying dimensionality reduction techniques like PCA or t-SNE to your dataset.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        reports:
          my_dimensionality_reduction_report: DimensionalityReduction
              name: my_report_name  # user-defined name of the report
              label: epitope  # label of the data

    Attributes:

    - name: a user-defined name of the report. It will also be the name of the HTML file.
    - label: the label of the data on which dimensionality reduction was performed.
    - method: an instance of UnsupervisedMLMethod class. It must have been trained before running the report.
    - result_path: the path where the HTML file will be stored.
    - dataset: the dataset on which the method was trained.

    Returns:

    - An HTML file containing plots of the dimensionality-reduced data. If the data was reduced to two dimensions, a 2D scatter plot will be created.
      If the method used has an explained variance, an additional explained variance plot will also be created. The plots are interactive and the data points in the scatter plot can be hovered over to display more information.

    """
    @classmethod
    def build_object(cls, **kwargs):
        name = kwargs["name"] if "name" in kwargs else "DimensionalityReduction"
        label = kwargs["label"] if "label" in kwargs else "epitope"
        return DimensionalityReduction(name=name, label=label)

    def __init__(self, dataset: Dataset = None, method: UnsupervisedMLMethod = None, result_path: Path = None, label: [str] = None, name: str = None, number_of_processes: int = 1):
        super().__init__(dataset=dataset, method=method, result_path=result_path,
                         name=name, number_of_processes=number_of_processes)
        self.label = label

    def _generate(self) -> ReportResult:
        PathBuilder.build(self.result_path)
        paths = []
        data = self.dataset.encoded_data.examples

        if isinstance(data, csr_matrix):
            data = data.toarray()
        if self.dataset.encoded_data.examples.shape[1] == 2:
            paths.append(self._2dplot(data, f'2d_{self.name}'))
        if type(self.method.model).__name__ is not "TSNE" or None:
            paths.append(self.explained_varience_plot())
        infoText = f"Dimensionality Reduction with {type(self.method.model).__name__}"

        return ReportResult(self.name,
                            info=infoText,
                            output_figures=[p for p in paths if p is not None])
    def explained_varience_plot(self):
        filename = self.result_path / "explained_variance.html"

        exp_var = self.method.model.explained_variance_ratio_
        x = list(range(0, len(exp_var)))
        cum_sum_eigenvalues = np.cumsum(exp_var)
        traces = []
        bar0 = go.Bar(x=x, y=exp_var, name="Individual explained variance")
        traces.append(bar0)
        bar1 = go.Scatter(x=x, y=cum_sum_eigenvalues, line_shape="hvh", mode="lines", name="Cumulative explained variance")
        traces.append(bar1)
        layout = go.Layout(xaxis=go.layout.XAxis(showgrid=False,
                                                 zeroline=False,
                                                 showline=True,
                                                 mirror=True,
                                                 title="Principal component index",
                                                 linewidth=1,
                                                 linecolor='gray'),
                           yaxis=go.layout.YAxis(showgrid=True,
                                                 title="Explained variance ratio",
                                                 zeroline=False,
                                                 showline=True,
                                                 mirror=True,
                                                 linewidth=1,
                                                 linecolor='black'),
                           hovermode='closest'
                           )
        figure = go.Figure(data=traces, layout=layout)

        with filename.open("w", encoding="utf-8") as file:
            figure.write_html(file)

        return ReportOutput(filename)
    def _2dplot(self, plotting_data, output_name):
        traces = []
        filename = self.result_path / f"{output_name}.html"

        data_grouped_by_label = {}
        for index, data in enumerate(list(self.dataset.get_data())):
            if data.metadata[self.label] not in data_grouped_by_label.keys():
                data_grouped_by_label.update({data.metadata[self.label]: {}})
            data_grouped_by_label[data.metadata[self.label]].update({self.dataset.encoded_data.example_ids[index]: plotting_data[index]})

        for label_value in data_grouped_by_label:
            data = np.array(list(data_grouped_by_label[label_value].values()))
            markerText = list("{}: {}<br>Datapoint id: {}".format(self.label, label_value, list(data_grouped_by_label[label_value].keys())[i])
                              for i in range(len(data)))
            trace = go.Scatter(x=data[:, 0],
                               y=data[:, 1],
                               text=markerText,
                               name=str(label_value),
                               mode='markers',
                               marker=go.scatter.Marker(opacity=1,
                                                        color=list(self.dataset.labels[self.label]).index(label_value)),
                               showlegend=True
                               )
            traces.append(trace)

        layout = go.Layout(xaxis=go.layout.XAxis(showgrid=False,
                                                 zeroline=False,
                                                 showline=True,
                                                 mirror=True,
                                                 linewidth=1,
                                                 linecolor='gray',
                                                 showticklabels=True),
                           yaxis=go.layout.YAxis(showgrid=False,
                                                 zeroline=False,
                                                 showline=True,
                                                 mirror=True,
                                                 linewidth=1,
                                                 linecolor='black',
                                                 showticklabels=True),
                           hovermode='closest',
                           template="ggplot2"
                           )
        figure = go.Figure(data=traces, layout=layout)

        with filename.open("w", encoding="utf-8") as file:
            figure.write_html(file)

        return ReportOutput(filename)
