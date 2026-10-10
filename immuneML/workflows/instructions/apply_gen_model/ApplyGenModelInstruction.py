import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np

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

    Optionally, one or more datasets can be provided (p_gen_datasets) to compute the generation probabilities of their
    sequences under the trained model. The sequences of each dataset are then stored along with their generation
    probabilities (column p_gen) in a separate tsv file in the datasets_with_p_gens folder. This is only available for
    models that can compute generation probabilities.


    **Specification arguments:**

    - gen_examples_count (int): how many examples (sequences, repertoires) to generate from the applied model; it can
      be set to 0 to skip generation if only the generation probabilities of the provided datasets are needed

    - reports (list): list of report ids (defined under definitions/reports) to apply after generating
      gen_examples_count examples; these can be data reports (to be run on generated examples), ML reports (to be run
      on the fitted model)

    - ml_config_path (str): path to the trained model in zip format (as provided by TrainGenModel instruction)

    - p_gen_datasets (list): optional; names of the sequence datasets (defined under definitions/datasets) for which
      the generation probabilities under the trained model should be computed; if not specified, no generation
      probabilities are computed

    **YAML specification:**

    .. highlight:: yaml
    .. code-block:: yaml

        instructions:
            my_apply_gen_model_inst: # user-defined instruction name
                type: ApplyGenModel
                gen_examples_count: 100
                ml_config_path: ./config.zip
                reports: [data_rep1, ml_rep2]
                p_gen_datasets: [my_dataset1, my_dataset2] # optional

    """

    def __init__(self, method: GenerativeModel = None, reports: list = None, result_path: Path = None,
                 name: str = None, gen_examples_count: int = None, p_gen_datasets: Dict[str, Dataset] = None):
        self.state = ApplyGenModelState(result_path, name, gen_examples_count)
        self.method = method
        self.reports = reports
        self.generated_dataset = None
        self.p_gen_datasets = p_gen_datasets

    def run(self, result_path: Path) -> ApplyGenModelState:
        self._set_path(result_path)
        self._gen_data()
        self._export_generated_dataset()
        self._compute_p_gens()
        self._run_reports()

        return self.state

    def _gen_data(self):
        if self.state.gen_examples_count == 0:
            return

        dataset = self.method.generate_sequences(self.state.gen_examples_count, 1,
                                                 self.state.result_path / 'generated_sequences',
                                                 SequenceType.AMINO_ACID, False)

        self.generated_dataset = dataset
        print_log(f"{self.state.name}: generated {self.state.gen_examples_count} examples from the fitted model",
                  True)

    def _export_generated_dataset(self):
        if self.state.gen_examples_count == 0:
            return

        AIRRExporter.export(self.state.generated_dataset, self.state.result_path / f'exported_gen_dataset')
        self.state.exported_datasets['generated_dataset'] = self.state.result_path / 'exported_gen_dataset'

    def _compute_p_gens(self):
        if not self.p_gen_datasets:
            return

        path = PathBuilder.build(self.state.result_path / 'datasets_with_p_gens')

        for name, dataset in self.p_gen_datasets.items():
            p_gens = self.method.compute_p_gens(dataset.data, SequenceType.AMINO_ACID)
            df = dataset.data.topandas()
            df['p_gen'] = np.asarray(p_gens, dtype=float)
            df.to_csv(path / f'{name}_p_gens.tsv', sep='\t', index=False)

            print_log(f"{self.state.name}: computed generation probabilities for {df.shape[0]} sequences from "
                      f"dataset {name}", True)

        self.state.exported_datasets['datasets_with_p_gens'] = path

    def _run_reports(self):
        report_path = self._get_reports_path()
        for report in self.reports:
            report.result_path = report_path
            if isinstance(report, DataReport):
                if self.generated_dataset is None:
                    print_log(f"{self.state.name}: no examples were generated, skipping data report {report.name}.",
                              True)
                    continue
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
