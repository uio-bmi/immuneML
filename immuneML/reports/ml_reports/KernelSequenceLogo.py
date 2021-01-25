import logging
from copy import copy

import logomaker
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.ml_methods.ReceptorCNN import ReceptorCNN
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.ml_reports.MLReport import MLReport
from immuneML.util.PathBuilder import PathBuilder


class KernelSequenceLogo(MLReport):
    """
    A report that plots kernels of a CNN model as sequence logos. It works only with trained ReceptorCNN models which has kernels already normalized
    to represent information gain matrices. Additionally, it also plots the weights in the final fully-connected layer of the network associated with
    kernel outputs. For more information on how the model works, see :ref:`ReceptorCNN`.

    The kernels are visualized using Logomaker. Original publication: Tareen A, Kinney JB. Logomaker: beautiful sequence logos in Python.
    Bioinformatics. 2020; 36(7):2272-2274. `doi:10.1093/bioinformatics/btz921 <https://academic.oup.com/bioinformatics/article/36/7/2272/5671693>`_.


    Arguments: this report does not take any arguments as input.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_kernel_seq_logo: KernelSequenceLogo

    """

    @classmethod
    def build_object(cls, **kwargs):
        return KernelSequenceLogo(**kwargs)

    def _generate(self) -> ReportResult:
        PathBuilder.build(self.result_path)
        report_result = ReportResult()
        sequence_alphabet = EnvironmentSettings.get_sequence_alphabet(self.method.sequence_type)
        for kernel_name in self.method.CNN.conv_chain_1 + self.method.CNN.conv_chain_2:
            figure_outputs, table_outputs = self._plot_kernels(kernel_name, sequence_alphabet)
            report_result.output_figures.extend(figure_outputs)
            report_result.output_tables.extend(table_outputs)

        figure_output, table_output = self._plot_fc_layer()
        report_result.output_figures.append(figure_output)
        report_result.output_tables.append(table_output)

        return report_result

    def _plot_kernels(self, kernel_name, sequence_alphabet):
        figure_outputs = []
        table_outputs = []
        friendly_kernel_name = copy(kernel_name).replace("chain_1", self.method.chain_names[0]).replace("chain_2", self.method.chain_names[1])

        for i in range(self.method.kernel_count):
            kernel = getattr(self.method.CNN, kernel_name)
            kernel_df = pd.DataFrame(kernel.weight[i].detach().numpy().T[:, :len(sequence_alphabet)], columns=sequence_alphabet)
            kernel_csv_path = self.result_path / f"{friendly_kernel_name}_{i + 1}.csv"
            kernel_df.to_csv(kernel_csv_path, index=False)
            table_outputs.append(ReportOutput(kernel_csv_path, friendly_kernel_name + f"_{i + 1}"))

            logo = logomaker.Logo(kernel_df, shade_below=0.5, fade_below=0.5, font_name='Arial Rounded MT Bold', vpad=0.05, vsep=0.01)
            logo_path = self.result_path / f"{friendly_kernel_name}_{i + 1}.png"

            logo.style_spines(visible=False)
            logo.style_spines(spines=('left', 'bottom'), visible=True)
            logo.style_xticks(fmt='%d', anchor=0)

            logo.fig.savefig(str(logo_path))
            plt.close(logo.fig)
            figure_outputs.append(ReportOutput(logo_path, f"{friendly_kernel_name}_{i + 1}"))

        return figure_outputs, table_outputs

    def _plot_fc_figure(self, df, bias):
        fig = make_subplots(rows=1, cols=2, column_widths=[0.8, 0.2], specs=[[{"type": "bar"}, {'type': "table"}]])
        fig.add_trace(go.Bar(x=df["names"], y=df["weights"], name="weights", hovertemplate='Weight for %{x}: %{y:.4f}<extra></extra>',
                             hoverlabel={"font_color": "white"}, marker_color=px.colors.diverging.Tealrose[0]), row=1, col=1)
        table = go.Table(header={"values": ["bias"]}, cells={"values": bias})
        table.cells.format = [[None], ['.3f']]
        fig.add_trace(table, row=1, col=2)
        fig.update_layout(template="plotly_white")
        fig.write_html(str(self.result_path / "fully_connected_layer_weights.html"))

        return ReportOutput(self.result_path / "fully_connected_layer_weights.html", "fully-connected layer weights")

    def _store_fc_table(self, df, bias):
        df.append({"weights": bias, "names": "bias"}, ignore_index=True)
        df.to_csv(self.result_path / "fully_connected_layer_weights.csv", index=False)

        return ReportOutput(self.result_path / "fully_connected_layer_weights.csv", "fully-connected layer weights")

    def _plot_fc_layer(self):
        weights = self.method.CNN.fully_connected.weight.detach().numpy()[0]
        tmp_bias = self.method.CNN.fully_connected.bias.detach().numpy()[0]
        bias = [str(round(tmp_bias, 5))]

        names = []
        for kernel_name in self.method.CNN.conv_chain_1 + self.method.CNN.conv_chain_2:
            friendly_kernel_name = copy(kernel_name).replace("chain_1", self.method.chain_names[0]).replace("chain_2", self.method.chain_names[1])
            names.extend([f"{friendly_kernel_name}_{i + 1}" for i in range(self.method.kernel_count)])

        df = pd.DataFrame({"weights": weights, "names": names})

        figure_output = self._plot_fc_figure(df, bias)
        table_output = self._store_fc_table(df, bias)

        return figure_output, table_output

    def check_prerequisites(self):

        run_report = True

        if self.method is None:
            logging.warning("KernelSequenceLogo: ML method is None, skipping report.")
            run_report = False
        elif not isinstance(self.method, ReceptorCNN):
            logging.info(f"KernelSequenceLogo: ML method is not instance of ReceptorCNN class, but of {type(self.method).__name__}, "
                         f"skipping report.")
            run_report = False

        return run_report
