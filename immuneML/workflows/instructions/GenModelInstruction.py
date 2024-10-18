import copy
from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from immuneML.IO.dataset_export.AIRRExporter import AIRRExporter
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.environment.SequenceType import SequenceType
from immuneML.ml_methods.generative_models import GenerativeModel
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.reports.ml_reports.MLReport import MLReport
from immuneML.util.Logger import print_log
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.Instruction import Instruction


@dataclass
class GenModelState:
    result_path: Path
    name: str
    gen_examples_count: int
    model_path: Path = None
    generated_dataset: Dataset = None
    exported_datasets: Dict[str, Path] = field(default_factory=dict)
    report_results: Dict[str, List[ReportResult]] = field(
        default_factory=lambda: {'data_reports': [], 'ml_reports': [], 'instruction_reports': []})


class GenModelInstruction(Instruction, ABC):

    def __init__(self, state=None, method: GenerativeModel = None, reports: list = None):
        self.generated_dataset = None
        self.method = method
        self.state = state
        self.reports = reports

    def _gen_data(self):
        dataset = self.method.generate_sequences(self.state.gen_examples_count, 1,
                                                 self.state.result_path / 'generated_sequences',
                                                 SequenceType.AMINO_ACID, False)

        print_log(f"{self.state.name}: generated {self.state.gen_examples_count} examples from the fitted model", True)
        self.generated_dataset = dataset
        self.state.generated_dataset = self.generated_dataset

    def _export_generated_dataset(self):
        AIRRExporter.export(self.generated_dataset, self.state.result_path / 'exported_gen_dataset')
        self.state.exported_datasets['generated_dataset'] = self.state.result_path / 'exported_gen_dataset'

    def _run_reports(self):
        report_path = self._get_reports_path()
        for report in self.reports:
            report.result_path = report_path
            if isinstance(report, DataReport):
                rep = copy.deepcopy(report)
                rep.dataset = self.generated_dataset
                rep.name = rep.name + " (generated dataset)"
                self.state.report_results['data_reports'].append(rep.generate_report())
            elif isinstance(report, MLReport):
                report.method = self.method
                self.state.report_results['ml_reports'].append(report.generate_report())

    def _print_report_summary_log(self):
        if len(self.reports) > 0:
            gen_rep_count = len(self.state.report_results['ml_reports']) + len(
                self.state.report_results['data_reports'])
            print_log(f"{self.state.name}: generated {gen_rep_count} reports.", True)

    def _get_reports_path(self) -> Path:
        return PathBuilder.build(self.state.result_path / 'reports')

    def _set_path(self, result_path):
        self.state.result_path = PathBuilder.build(result_path / self.state.name)
