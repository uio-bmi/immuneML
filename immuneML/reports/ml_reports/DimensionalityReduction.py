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
    @classmethod
    def build_object(cls, **kwargs):
        name = kwargs["name"] if "name" in kwargs else "DimensionalityReduction"
        return DimensionalityReduction(name=name)

    def __init__(self, dataset: Dataset = None, method: UnsupervisedMLMethod = None, result_path: Path = None, name: str = None, number_of_processes: int = 1):
        super().__init__(dataset=dataset, method=method, result_path=result_path,
                         name=name, number_of_processes=number_of_processes)

    def _generate(self) -> ReportResult:
        PathBuilder.build(self.result_path)
        paths = []
        data = self.dataset.encoded_data.examples

        if isinstance(data, csr_matrix):
            data = data.toarray()
        if self.dataset.encoded_data.examples.shape[1] == 2:
            paths.append(self._2dplot(data, f'2d_{self.name}'))
        if self.method is not None:
            paths.append(self.explained_varience_plot())
        infoText = f"Dimensionality Reduction with {type(self.method.model).__name__}"

        return ReportResult(self.name,
                            info=infoText,
                            output_figures=[p for p in paths if p is not None])
    def explained_varience_plot(self):
        filename = self.result_path / "explained_variance.html"

        exp_var_pca = self.method.model.explained_variance_ratio_
        x = list(range(0, len(exp_var_pca)))
        cum_sum_eigenvalues = np.cumsum(exp_var_pca)
        traces = []
        bar0 = go.Bar(x=x, y=exp_var_pca, name="Individual explained variance")
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

        with filename.open("w") as file:
            figure.write_html(file)

        return ReportOutput(filename)
    def _2dplot(self, plotting_data, output_name):
        traces = []
        filename = self.result_path / f"{output_name}.html"

        trace0 = go.Scatter(x=plotting_data[:, 0],
                            y=plotting_data[:, 1],
                            name='Data points',
                            mode='markers',
                            marker=go.scatter.Marker(opacity=1),
                            showlegend=True
                            )
        traces.append(trace0)
        layout = go.Layout(xaxis=go.layout.XAxis(showgrid=False,
                                                 zeroline=False,
                                                 showline=True,
                                                 mirror=True,
                                                 title="PC1",
                                                 linewidth=1,
                                                 linecolor='gray',
                                                 showticklabels=True),
                           yaxis=go.layout.YAxis(showgrid=False,
                                                 title="PC2",
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



        with filename.open("w") as file:
            figure.write_html(file)

        return ReportOutput(filename)
