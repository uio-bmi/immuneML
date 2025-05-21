import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA

from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.dsl.definition_parsers.MLParser import MLParser
from immuneML.ml_methods.dim_reduction.DimRedMethod import DimRedMethod
from immuneML.ml_methods.generative_models.GenerativeModel import GenerativeModel
from immuneML.ml_methods.generative_models.SimpleVAE import SimpleVAE
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.gen_model_reports.GenModelReport import GenModelReport
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


class VAESummary(GenModelReport):
    """
    This report provides the summary of the train SimpleVAE and shows the following:

    - plots of the latent space after applying a dimensionality reduction method to reduce the data to 2 dimensions,
      highlighted by V and J gene
    - plots the histogram for each latent dimension
    - plots loss per epoch

    **Specification arguments:**

    - dim_dist_cols (int): how many columns to use to plot the histograms of latent dimensions (either this or
      dim_dist_rows has to be set, or both)

    - dim_dist_rows (int): how many rows to use to plot the histogram of latent dimensions (either this or
      dim_dist_cols has to be set, or both)

    - dim_red_method (dict): which dimensionality reduction method to use along with its parameters; the method has to
      return 2 dimensions; see :ref:`**Dimensionality reduction methods**` for available options;
      default is PCA with 2 components

    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            reports:
                my_vae_summary:
                    VAESummary:
                        dim_dist_cols: 4
                        dim_dist_rows: None
                        dim_red_method:
                            PCA:
                                n_components: 2

    """
    @classmethod
    def build_object(cls, **kwargs):
        name = kwargs["name"] if "name" in kwargs else "VAESummary"

        ParameterValidator.assert_keys_present(list(kwargs.keys()), ['dim_dist_cols', 'dim_dist_rows', 'dim_red_method'],
                                               VAESummary.__name__, 'parameters')

        cls_name = list(kwargs['dim_red_method'].keys())[0]
        method = MLParser.parse_any_model("dim_red_method", kwargs['dim_red_method'], cls_name)[0]
        assert len(method.get_dimension_names()) == 2, \
            (f"{VAESummary.__name__}: dimensionality reduction method has to return 2 dimensions, got "
             f"{len(method.get_dimension_names())} instead.")

        assert ((kwargs['dim_dist_cols'] is not None and isinstance(kwargs['dim_dist_cols'], int)) or
                (kwargs['dim_dist_rows'] is not None and isinstance(kwargs['dim_dist_rows'], int))), \
            f"{cls.__name__}: at least one of dim_dist_cols or dim_dist_rows has to be set."

        return VAESummary(name=name, dim_dist_cols=kwargs['dim_dist_cols'], dim_dist_rows=kwargs['dim_dist_rows'],
                          dim_red_method=method)

    def __init__(self, dim_dist_rows: int, dim_dist_cols: int, dataset: Dataset = None, model: SimpleVAE = None,
                 result_path: Path = None, name: str = None, dim_red_method: DimRedMethod = None):
        super().__init__(dataset, model, result_path, name)
        self.dim_dist_rows = dim_dist_rows
        self.dim_dist_cols = dim_dist_cols
        self.dim_red_method = dim_red_method

    @property
    def dim_red_method_name(self) -> str:
        return self.dim_red_method.__class__.__name__

    def _generate(self) -> ReportResult:
        PathBuilder.build(self.result_path)

        embeddings = self.get_embeddings()

        latent_space_table_out = self._prepare_latent_space(embeddings)
        train_progress_table_out = self._prepare_training_progress()

        latent_space_fig_out = self._safe_plot(output_written=True, plot_callable='_plot_latent_space',
                                               latent_space_table=latent_space_table_out)
        latent_dim_dist_fig_out = self._safe_plot(output_written=True, embeddings=embeddings,
                                                  plot_callable='_plot_latent_dimension_distributions')
        train_progress_fig_out = self._safe_plot(output_written=True, plot_callable='_plot_training_progress',
                                                 training_progress_table=train_progress_table_out)

        output_figures = [latent_space_fig_out, latent_dim_dist_fig_out, train_progress_fig_out]
        output_figures = [el for el in output_figures if el is not None]

        result = ReportResult(name=self.name, info='Summary of the fitted VAE model.', output_figures=output_figures,
                              output_tables=[latent_space_table_out, train_progress_table_out])
        return result

    def get_embeddings(self):

        data_loader = self.model.encode_dataset(self.dataset, None, self.dataset.get_example_count(), False)

        for data in data_loader:
            with torch.no_grad():
                cdr3_input, v_gene_input, j_gene_input = data
                embeddings = self.model.model.encoding_func(cdr3_input, v_gene_input, j_gene_input)

        return embeddings

    def _prepare_latent_space(self, embeddings) -> ReportOutput:
        embeddings = self.dim_red_method.fit_transform(design_matrix=embeddings.numpy())

        df = pd.DataFrame(data=embeddings, columns=self.dim_red_method.get_dimension_names())
        df['v_gene'] = [v_call.split("*")[0] for v_call in self.dataset.data.v_call.tolist()]
        df['j_gene'] = [j_call.split("*")[0] for j_call in self.dataset.data.j_call.tolist()]

        path = self.result_path / f'latent_space_{self.dim_red_method_name}.csv'

        df.to_csv(str(path), index=False)

        return ReportOutput(path, f'principal component analysis on the data embedded into '
                                  f'{self.model.latent_dim} dimensional space')

    def _prepare_training_progress(self) -> ReportOutput:
        path = self.result_path / 'training_losses.csv'
        shutil.copyfile(str(self.model.loss_path), str(path))
        return ReportOutput(path, 'Loss per epoch')

    def _plot_latent_space(self, latent_space_table: ReportOutput) -> ReportOutput:

        encoded_data = pd.read_csv(latent_space_table.path)
        dim_names = self.dim_red_method.get_dimension_names()

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, shared_yaxes=True, subplot_titles=['V genes', 'J genes'],
                            x_title=dim_names[0], y_title=dim_names[1])

        for gene in self.model.unique_v_genes:
            tmp_df = encoded_data[encoded_data['v_gene'] == gene]
            fig.add_trace(
                go.Scatter(x=tmp_df[dim_names[0]].values, y=tmp_df[dim_names[1]].values, mode='markers', opacity=0.5,
                           name=gene,
                           marker=dict(colorscale='Viridis'), legendgroup=1,
                           legendgrouptitle_text='V genes'), row=1, col=1)
        for gene in self.model.unique_j_genes:
            tmp_df = encoded_data[encoded_data['j_gene'] == gene]
            fig.add_trace(
                go.Scatter(x=tmp_df[dim_names[0]].values, y=tmp_df[dim_names[1]].values, mode='markers', opacity=0.5,
                           name=gene,
                           marker=dict(colorscale='Viridis'),
                           legendgroup=2, legendgrouptitle_text='J genes'), row=2, col=1)

        fig.update_layout(template='plotly_white', legend_tracegroupgap=30)
        fig.write_html(self.result_path / f'latent_space_{self.dim_red_method_name}.html')

        return ReportOutput(self.result_path / f'latent_space_{self.dim_red_method_name}.html',
                            f'principal component analysis on the data embedded into {self.model.latent_dim} '
                            f'dimensional space')

    def _plot_latent_dimension_distributions(self, embeddings) -> ReportOutput:

        if self.dim_dist_rows is None:
            self.dim_dist_rows = int(self.model.latent_dim / self.dim_dist_cols)
        elif self.dim_dist_cols is None:
            self.dim_dist_cols = int(self.model.latent_dim / self.dim_dist_rows)

        assert self.dim_dist_rows * self.dim_dist_cols == self.model.latent_dim, \
            (f"{VAESummary.__name__}: cannot plot latent dimension distribution since dim_dist_cols and dim_dist_rows "
             f"parameters do not match the latent dimension of the VAE.")

        fig = make_subplots(rows=self.dim_dist_rows, cols=self.dim_dist_cols, shared_xaxes=True, shared_yaxes=True)

        i = 0
        for row in range(self.dim_dist_rows):
            for col in range(self.dim_dist_cols):
                fig.add_trace(go.Histogram(x=embeddings[:, i].numpy(), name=f'dim {i + 1}'),
                              row=row + 1, col=col + 1)
                i += 1

        fig.update_layout(template='plotly_white')
        fig.write_html(self.result_path / 'latent_dim_dist.html')

        return ReportOutput(path=self.result_path / 'latent_dim_dist.html',
                            name=f'latent dimension distribution for examples from dataset {self.dataset.name}')

    def _plot_training_progress(self, training_progress_table: ReportOutput) -> ReportOutput:
        df = pd.read_csv(str(training_progress_table.path))
        fig = px.line(df, x='epoch', y='loss', markers=True)
        fig.update_layout(template='plotly_white')
        fig.write_html(self.result_path / 'loss_per_epoch.html')

        return ReportOutput(path=self.result_path / 'loss_per_epoch.html',
                            name='loss per epoch')

    def check_prerequisites(self) -> bool:
        return isinstance(self.model, SimpleVAE)
