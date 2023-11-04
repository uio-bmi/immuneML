from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict

from immuneML.IO.dataset_export.AIRRExporter import AIRRExporter
from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.environment.SequenceType import SequenceType
from immuneML.ml_methods.generative_models.GenerativeModel import GenerativeModel
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.reports.ml_reports.MLReport import MLReport
from immuneML.util.Logger import print_log
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.Instruction import Instruction
from immuneML.workflows.instructions.dimensionality_reduction.DimensionalityReductionState import \
    DimensionalityReductionState


class DimensionalityReductionInstruction(Instruction):
    def __init__(self, dataset: Dataset = None, method: str = "", name: str = None,
                 result_path: Path = None, reports: list = None):
        self.dataset = dataset
        self.generated_dataset = None
        self.state = DimensionalityReductionState(name, result_path)
        self.reports = reports

    def run(self, result_path: Path) -> DimensionalityReductionState:
        self.state.name = "Test123"
        return self.state
