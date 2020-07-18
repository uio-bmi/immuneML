import logging

import logomaker
import matplotlib.pyplot as plt
import pandas as pd

from source.environment.EnvironmentSettings import EnvironmentSettings
from source.ml_methods.ReceptorCNN import ReceptorCNN
from source.reports.ReportOutput import ReportOutput
from source.reports.ReportResult import ReportResult
from source.reports.ml_reports.MLReport import MLReport
from source.util.PathBuilder import PathBuilder


class KernelSequenceLogo(MLReport):
    """
    A report that plots kernels of a CNN model as sequence logos. It works only with trained ReceptorCNN models which has kernels already normalized
    to represent information gain matrices. For more information on how the model works, see :py:obj:`~source.ml_methods.ReceptorCNN.ReceptorCNN`.

    The kernels are visualized using Logomaker. Original publication: Tareen, A. & Kinney, J. B. Logomaker: beautiful sequence logos in Python.
    Bioinformatics 36, 2272–2274 (2020).

    Arguments: this report does not take any arguments as input.

    Specification:

    .. indent with spaces
    .. code-block:: yaml

        my_kernel_seq_logo: KernelSequenceLogo

    """

    @classmethod
    def build_object(cls, **kwargs):
        return KernelSequenceLogo(**kwargs)

    def generate(self) -> ReportResult:
        PathBuilder.build(self.result_path)
        report_result = ReportResult()
        sequence_alphabet = EnvironmentSettings.get_sequence_alphabet(self.method.sequence_type)
        for kernel_name in self.method.CNN.conv_chain_1 + self.method.CNN.conv_chain_2:
            figure_outputs, table_outputs = self._plot_kernels(kernel_name, sequence_alphabet)
            report_result.output_figures.extend(figure_outputs)
            report_result.output_tables.extend(table_outputs)

        return report_result

    def _plot_kernels(self, kernel_name, sequence_alphabet):
        figure_outputs = []
        table_outputs = []

        for i in range(self.method.kernel_count):

            kernel = getattr(self.method.CNN, kernel_name)
            kernel_df = pd.DataFrame(kernel.weight[i].detach().numpy().T[:, :len(sequence_alphabet)], columns=sequence_alphabet)
            kernel_csv_path = self.result_path + kernel_name + f"_{i+1}.csv"
            kernel_df.to_csv(kernel_csv_path, index=False)
            table_outputs.append(ReportOutput(kernel_csv_path, kernel_name + f"_{i+1}"))

            logo = logomaker.Logo(kernel_df, shade_below=0.5, fade_below=0.5, font_name='Arial Rounded MT Bold', vpad=0.05, vsep=0.01)
            logo_path = self.result_path + kernel_name + f"_{i+1}.png"

            logo.style_spines(visible=False)
            logo.style_spines(spines=('left', 'bottom'), visible=True)
            logo.style_xticks(fmt='%d', anchor=0)

            logo.fig.savefig(logo_path)
            plt.close(logo.fig)
            figure_outputs.append(ReportOutput(logo_path, kernel_name + f"_{i+1}"))

        return figure_outputs, table_outputs

    def check_prerequisites(self):

        run_report = True

        if self.method is None:
            logging.warning("KernelSequenceLogo: ML method is None, skipping report.")
            run_report = False
        elif not isinstance(self.method, ReceptorCNN):
            logging.warning(f"KernelSequenceLogo: ML method is not instance of ReceptorCNN class, but of {type(self.method).__name__}, "
                            f"skipping report.")
            run_report = False

        return run_report
