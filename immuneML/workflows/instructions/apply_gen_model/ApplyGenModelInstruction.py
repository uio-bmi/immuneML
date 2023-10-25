from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from immuneML.IO.dataset_export.AIRRExporter import AIRRExporter
from immuneML.environment.SequenceType import SequenceType
from immuneML.ml_methods.generative_models.GenerativeModel import GenerativeModel
from immuneML.reports.ReportResult import ReportResult
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
    def __init__(self, result_path: Path = None, name: str = None, method: GenerativeModel = None,
                 gen_examples_count: int = None, reports: list = None):
        self.state = ApplyGenModelState(result_path, name, gen_examples_count)
        self.method = method
        self.reports = reports

    def run(self, result_path: Path) -> ApplyGenModelState:
        self.state.result_path = PathBuilder.build(result_path / self.state.name)
        dataset = self.method.generate_sequences(self.state.gen_examples_count, 1,
                                                 self.state.result_path / 'generated_sequences',
                                                 SequenceType.AMINO_ACID, False)
        #self._gen_data()

        return self.state

    def _load_model(self):
        pass

    def _gen_data(self):
        dataset = self.method.generate_sequences(self.state.gen_examples_count, 1,
                                                 self.state.result_path / 'generated_sequences',
                                                 SequenceType.AMINO_ACID, False)

        print_log(f"{self.state.name}: generated {self.state.gen_examples_count} examples from the fitted model", True)
        self.generated_dataset = dataset

        AIRRExporter.export(dataset, self.state.result_path)

    def _run_reports(self):
        pass
