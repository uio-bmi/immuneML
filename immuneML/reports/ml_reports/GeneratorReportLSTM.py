
from pathlib import Path
from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.ml_methods.LSTM import LSTM
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.ml_reports.GeneratorReport import GeneratorReport

import plotly.express as px


class NeuralNetGeneratorReport(GeneratorReport):
    """
    A subclass of the GeneratorReport that works on neural nets that produce loss values.
    Initially runs the regular GeneratorReport and then appends the loss value graph.

    Arguments:

        name: Name of the report

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_neural_net_report:
            NeuralNetGeneratorReport:
                name: nnReport
    """

    @classmethod
    def build_object(cls, **kwargs):
        name = kwargs["name"] if "name" in kwargs else "NeuralNetGeneratorReport"
        return NeuralNetGeneratorReport(name=name)

    def __init__(self, dataset: Dataset = None, method: LSTM = None, result_path: Path = None,
                 name: str = None, number_of_processes: int = 1):
        super().__init__(dataset=dataset, method=method, result_path=result_path,
                         name=name, number_of_processes=number_of_processes)

    def _generate(self) -> ReportResult:

        report_result = self._make_report()

        loss_over_time = self.result_path / f"{self.name}loss.html"

        if self.method.historydf is not None:
            fig = px.line(self.method.historydf['data'][0])
            with loss_over_time.open("w", encoding="utf-8") as file:
                fig.write_html(file)
            report_result.output_figures.append(ReportOutput(loss_over_time, name="Loss"))

        return report_result
