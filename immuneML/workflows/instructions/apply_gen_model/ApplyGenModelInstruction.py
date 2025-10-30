import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

from immuneML.IO.dataset_export.AIRRExporter import AIRRExporter
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.environment.SequenceType import SequenceType
from immuneML.ml_methods.generative_models import GenerativeModel
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
    model_path: Path = None
    generated_dataset: Dataset = None
    exported_datasets: Dict[str, Path] = field(default_factory=dict)
    report_results: dict = field(default_factory=lambda: {'data_reports': [], 'ml_reports': []})


class ApplyGenModelInstruction(Instruction):
    """

    ApplyGenModel instruction implements applying generative AIRR models on the sequence level.

    This instruction takes as input a trained model (trained in the :ref:`TrainGenModel` instruction)
    which will be used for generating data and the number of sequences to be generated.
    It can also produce reports of the applied model and reports of generated sequences.


    **Specification arguments:**

    - gen_examples_count (int): how many examples (sequences, repertoires) to generate from the applied model

    - reports (list): list of report ids (defined under definitions/reports) to apply after generating
      gen_examples_count examples; these can be data reports (to be run on generated examples), ML reports (to be run
      on the fitted model)

    - ml_config_path (str): path to the trained model in zip format (as provided by TrainGenModel instruction)

    **YAML specification:**

    .. highlight:: yaml
    .. code-block:: yaml

        instructions:
            my_apply_gen_model_inst: # user-defined instruction name
                type: ApplyGenModel
                gen_examples_count: 100
                ml_config_path: ./config.zip
                reports: [data_rep1, ml_rep2]

    """

    def __init__(self, method: GenerativeModel = None, reports: list = None, result_path: Path = None,
                 name: str = None, gen_examples_count: int = None):
        self.state = ApplyGenModelState(result_path, name, gen_examples_count)
        self.method = method
        self.reports = reports
        self.generated_dataset = None

    def run(self, result_path: Path) -> ApplyGenModelState:
        self._set_path(result_path)
        self._gen_data()
        self._export_generated_dataset()
        self._run_reports()

        return self.state

    def _gen_data(self):
        dataset = self.method.generate_sequences(self.state.gen_examples_count, 1,
                                                 self.state.result_path / 'generated_sequences',
                                                 SequenceType.AMINO_ACID, False)

        self.generated_dataset = dataset
        print_log(f"{self.state.name}: generated {self.state.gen_examples_count} examples from the fitted model",
                  True)

    def _export_generated_dataset(self):
        AIRRExporter.export(self.state.generated_dataset, self.state.result_path / f'exported_gen_dataset')
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
                rep = copy.deepcopy(report)
                rep.method = self.method
                rep.name = rep.name
                self.state.report_results['ml_reports'].append(rep.generate_report())

        self._print_report_summary_log()

    def _print_report_summary_log(self):
        if len(self.reports) > 0:
            gen_rep_count = len(self.state.report_results['ml_reports']) + len(
                self.state.report_results['data_reports'])
            print_log(f"{self.state.name}: generated {gen_rep_count} reports.", True)

    def _get_reports_path(self) -> Path:
        return PathBuilder.build(self.state.result_path / 'reports')

    def _set_path(self, result_path):
        self.state.result_path = PathBuilder.build(result_path / self.state.name)
