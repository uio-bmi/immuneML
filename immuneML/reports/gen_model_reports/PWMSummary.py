from pathlib import Path
from typing import Tuple, List

import pandas as pd
import plotly.express as px

from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.ml_methods.generative_models.GenerativeModel import GenerativeModel
from immuneML.ml_methods.generative_models.PWM import PWM
from immuneML.reports.PlotlyUtil import PlotlyUtil
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.gen_model_reports.GenModelReport import GenModelReport
from immuneML.util.PathBuilder import PathBuilder


class PWMSummary(GenModelReport):
    """
    This report provides the summary of the baseline PWM and shows the following:

    - probabilities of generated sequences having different lengths
    - PWMs for each length with positive probability

    This report takes no input arguments.

    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        my_pwm_summary: PWMSummary

    """

    @classmethod
    def build_object(cls, **kwargs):
        name = kwargs["name"] if "name" in kwargs else "PWMSummary"
        return PWMSummary(name=name)

    def __init__(self, dataset: Dataset = None, model: GenerativeModel = None,
                 result_path: Path = None, name: str = None):
        super().__init__(dataset, model, result_path, name)
        self.info = ("This reports provides a summary of baseline PWM model and shows the probabilities of generated "
                     "sequences having different lengths, as well as logo plots for each length with positive "
                     "probability.")

    def _generate(self) -> ReportResult:
        PathBuilder.build(self.result_path)
        len_figure, len_table = self._report_length_probs()
        pwm_figures, pwm_tables = self._report_pwms()
        return ReportResult(name=self.name, info=self.info, output_figures=[len_figure] + pwm_figures,
                            output_tables=[len_table] + pwm_tables)

    def _report_length_probs(self) -> Tuple[ReportOutput, ReportOutput]:
        df = pd.DataFrame({'sequence_length': self.model.length_probs.keys(), 'probability': self.model.length_probs.values()})
        df.to_csv(str(self.result_path / 'length_probs.csv'), index=False)
        len_table = ReportOutput(path=self.result_path / 'length_probs.csv', name='Sequence length probabilities')

        fig = px.bar(df, x="sequence_length", y="probability")
        fig.update_layout(xaxis=dict(tickmode='array', tickvals=df["sequence_length"]),
                          template="plotly_white")
        fig.update_traces(marker_color=px.colors.diverging.Tealrose[0])
        fig.write_html(self.result_path / 'length_probs.html')
        len_figure = ReportOutput(path=self.result_path / 'length_probs.html', name="Sequence length probabilities")

        return len_figure, len_table

    def _report_pwms(self) -> Tuple[List[ReportOutput], List[ReportOutput]]:
        figure_outputs = []
        table_outputs = []
        for seq_len in self.model.pwm_matrix:
            df = pd.DataFrame(data=self.model.pwm_matrix[seq_len],
                              index=EnvironmentSettings.get_sequence_alphabet(self.model.sequence_type))
            df = pd.melt(df.reset_index().rename(columns={'index': 'letter'}), id_vars=['letter'], var_name='position',
                         value_name='probability')
            figure = px.bar(df, x="position", y="probability", color="letter", text="letter",
                            color_discrete_map=PlotlyUtil.get_amino_acid_color_map(), template="plotly_white")
            figure.update_layout(showlegend=False)

            fig_path = self.result_path / f"pwm_len_{seq_len}.html"
            figure.write_html(str(fig_path))
            figure_outputs.append(ReportOutput(fig_path, f"PWM (sequence length = {seq_len})"))

            table_path = self.result_path / f"pwm_len_{seq_len}.csv"
            df.to_csv(str(table_path), index=False)
            table_outputs.append(ReportOutput(table_path, f"PWM (sequence length = {seq_len})"))

        return figure_outputs, table_outputs

    def check_prerequisites(self) -> bool:
        return isinstance(self.model, PWM)
