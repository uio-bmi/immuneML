from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from immuneML.IO.dataset_export.AIRRExporter import AIRRExporter
from immuneML.environment.SequenceType import SequenceType
from immuneML.ml_methods.generative_models.GenerativeModel import GenerativeModel
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.reports.ml_reports.MLReport import MLReport
from immuneML.util.Logger import print_log
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.Instruction import Instruction


@dataclass
class ApplyGenModelState:
    result_path: Path
    name: str
    gen_examples_count: int
    sequence_examples: list = None
    model_path: Path = None
    report_results: Dict[str, List[ReportResult]] = field(
        default_factory=lambda: {'data_reports': [], 'ml_reports': []})


class ApplyGenModelInstruction(Instruction):
    def __init__(self, method: GenerativeModel = None, reports: list = None, result_path: Path = None,
                 name: str = None, gen_examples_count: int = None):
        self.generated_dataset = None
        self.method = method
        self.reports = reports
        self.state = ApplyGenModelState(result_path, name, gen_examples_count)

    def run(self, result_path: Path) -> ApplyGenModelState:
        self.state.result_path = PathBuilder.build(result_path / self.state.name)
        self._gen_data()
        self._run_reports()

        return self.state

    def _gen_data(self):
        dataset = self.method.generate_sequences(self.state.gen_examples_count, 1,
                                                 self.state.result_path / 'generated_sequences',
                                                 SequenceType.AMINO_ACID, False)

        print_log(f"{self.state.name}: generated {self.state.gen_examples_count} examples from the fitted model", True)
        self.generated_dataset = dataset

        AIRRExporter.export(dataset, self.state.result_path)

    def _run_reports(self):
        report_path = PathBuilder.build(self.state.result_path / 'reports')
        for report in self.reports:
            report.result_path = report_path
            if isinstance(report, DataReport):
                report.dataset = self.generated_dataset
                self.state.report_results['data_reports'].append(report.generate_report())
            elif isinstance(report, MLReport):
                report.method = self.method
                self.state.report_results['ml_reports'].append(report.generate_report())

        if len(self.reports) > 0:
            gen_rep_count = len(self.state.report_results['ml_reports']) + len(
                self.state.report_results['data_reports'])
            print_log(f"{self.state.name}: generated {gen_rep_count} reports.", True)
