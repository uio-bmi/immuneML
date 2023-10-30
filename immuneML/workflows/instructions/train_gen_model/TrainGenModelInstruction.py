from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict

from immuneML.IO.dataset_export.AIRRExporter import AIRRExporter
from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.environment.SequenceType import SequenceType
from immuneML.ml_methods.generative_models.GenerativeModel import GenerativeModel
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.reports.gen_model_reports.GenModelReport import GenModelReport
from immuneML.reports.ml_reports.MLReport import MLReport
from immuneML.util.Logger import print_log
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.Instruction import Instruction


@dataclass
class TrainGenModelState:
    result_path: Path
    name: str
    gen_examples_count: int
    sequence_examples: list = None
    model_path: Path = None
    report_results: Dict[str, List[ReportResult]] = field(default_factory=lambda: {'data_reports': [], 'ml_reports': []})


class TrainGenModelInstruction(Instruction):
    """
    TrainGenModel instruction implements training generative AIRR models on both repertoire and receptor level
    depending on the parameters and the chosen model. Models that can be trained for sequence generation are listed
    under Generative Models section.

    This instruction takes a dataset as input which will be used to train a model, the model itself, and the number of
    sequences to generate to illustrate the applicability of the model. It can also produce reports of the fitted model
    or generated sequences.

    To use the generative model previously trained with immuneML, see ApplyGenModel instruction.

    Arguments:

        dataset: dataset to use for fitting the generative model; it has to be defined under definitions/datasets

        method: which model to fit (defined previously under definitions/ml_methods)

        number_of_processes (int): how many processes to use for fitting the model

        gen_examples_count (int): how many examples (sequences, repertoires) to generate from the fitted model

        reports (list): list of report ids (defined under definitions/reports) to apply after fitting a generative model and generating gen_examples_count examples; these can be data reports (to be run on generated examples), ML reports (to be run on the fitted model)

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_train_gen_model_inst: # user-defined instruction name
            type: TrainGenModel
            dataset: d1 # defined previously under definitions/datasets
            model: model1 # defined previously under definitions/ml_methods
            gen_examples_count: 100
            number_of_processes: 4
            reports: [data_rep1, ml_rep2]

    """

    MAX_ELEMENT_COUNT_TO_SHOW = 10

    def __init__(self, dataset: Dataset = None, method: GenerativeModel = None, number_of_processes: int = 1,
                 gen_examples_count: int = 100, result_path: Path = None, name: str = None, reports: list = None):
        self.dataset = dataset
        self.generated_dataset = None
        self.number_of_processes = number_of_processes
        self.method = method
        self.state = TrainGenModelState(result_path, name, gen_examples_count)
        self.reports = reports

    def run(self, result_path: Path) -> TrainGenModelState:
        self._set_path(result_path)
        self._fit_model()
        self._save_model()
        self._gen_data()
        self._evaluate_model()
        self._run_reports()

        return self.state

    def _fit_model(self):
        print_log(f"{self.state.name}: starting to fit the model", True)
        self.method.fit(self.dataset, self.state.result_path)
        print_log(f"{self.state.name}: fitted the model", True)

    def _save_model(self):
        self.state.model_path = self.method.save_model(self.state.result_path / 'trained_model/')

    def _gen_data(self):
        dataset = self.method.generate_sequences(self.state.gen_examples_count, 1,
                                                 self.state.result_path / 'generated_sequences',
                                                 SequenceType.AMINO_ACID, False)

        print_log(f"{self.state.name}: generated {self.state.gen_examples_count} examples from the fitted model", True)
        self.generated_dataset = dataset

        AIRRExporter.export(dataset, self.state.result_path)

    def _evaluate_model(self):
        print("Evaluation is not implemented yet!")

    def _run_reports(self):
        report_path = PathBuilder.build(self.state.result_path / 'reports')
        for report in self.reports:
            report.result_path = report_path
            if isinstance(report, DataReport):
                report.dataset = self.generated_dataset
                self.state.report_results['data_reports'].append(report.generate_report())
            elif isinstance(report, GenModelReport):
                report.model = self.method
                report.dataset = self.dataset
                self.state.report_results['ml_reports'].append(report.generate_report())

        if len(self.reports) > 0:
            gen_rep_count = len(self.state.report_results['ml_reports']) + len(self.state.report_results['data_reports'])
            print_log(f"{self.state.name}: generated {gen_rep_count} reports.", True)

    def _set_path(self, result_path):
        self.state.result_path = PathBuilder.build(result_path / self.state.name)
