
from pathlib import Path
from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.ml_methods.PWM import PWM
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.ml_reports.GeneratorReport import GeneratorReport
from immuneML.util.PathBuilder import PathBuilder
import pandas as pd


class GeneratorReportPWM(GeneratorReport):
    @classmethod
    def build_object(cls, **kwargs):
        name = kwargs["name"] if "name" in kwargs else "GeneratorReport"
        return GeneratorReportPWM(name=name)

    def __init__(self, dataset: Dataset = None, method: PWM = None, result_path: Path = None,
                 name: str = None, number_of_processes: int = 1):
        super().__init__(dataset=dataset, method=method, result_path=result_path,
                         name=name, number_of_processes=number_of_processes)
        self.dataset = dataset

    def _generate(self) -> ReportResult:

        PathBuilder.build(self.result_path)
        output_figures = []

        generated_sequences = self.result_path / f"{self.name}GeneratedSequences.csv"
        logo_path = self.result_path / f"{self.name}Logo.png"
        data_logo_path = self.result_path / f"{self.name}Data_Logo.png"

        data = pd.DataFrame(enumerate(self.sequences), columns=["id", "sequence_aas"])
        data.to_csv(generated_sequences, index=False)

        self._make_generated_logo(logo_path)
        output_figures.append(ReportOutput(logo_path, name="Generated Logo"))

        if self.dataset:
            self._make_dataset_logo(data_logo_path)
            output_figures.append(ReportOutput(data_logo_path, name="Dataset Logo"))

        sequences_to_output = ReportOutput(generated_sequences, name="Generated Sequences")

        return ReportResult(self.name, output_figures=output_figures, output_tables=[sequences_to_output])
