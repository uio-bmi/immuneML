import copy
import logging
from pathlib import Path

import numpy as np

from immuneML.IO.dataset_export.AIRRExporter import AIRRExporter
from immuneML.data_model.bnp_util import merge_dataclass_objects, bnp_write_to_file, get_type_dict_from_bnp_object
from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.ml_methods.generative_models.GenerativeModel import GenerativeModel
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.reports.train_gen_model_reports.TrainGenModelReport import TrainGenModelReport
from immuneML.util.Logger import print_log
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.GenModelInstruction import GenModelState, GenModelInstruction


class TrainGenModelState(GenModelState):
    combined_dataset: Dataset = None


class TrainGenModelInstruction(GenModelInstruction):
    """
    TrainGenModel instruction implements training generative AIRR models on receptor level. Models that can be trained
    for sequence generation are listed under Generative Models section.

    This instruction takes a dataset as input which will be used to train a model, the model itself, and the number of
    sequences to generate to illustrate the applicability of the model. It can also produce reports of the fitted model
    and reports of original and generated sequences.

    To use the generative model previously trained with immuneML, see ApplyGenModel instruction.

    .. note::

        This is an experimental feature in version 3.0.0a1.

    Specification arguments:

    - dataset: dataset to use for fitting the generative model; it has to be defined under definitions/datasets

    - method: which model to fit (defined previously under definitions/ml_methods)

    - number_of_processes (int): how many processes to use for fitting the model

    - gen_examples_count (int): how many examples (sequences, repertoires) to generate from the fitted model

    - reports (list): list of report ids (defined under definitions/reports) to apply after fitting a generative model
      and generating gen_examples_count examples; these can be data reports (to be run on generated examples), ML
      reports (to be run on the fitted model)

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
                 gen_examples_count: int = 100, result_path: Path = None, name: str = None, reports: list = None,
                 export_combined_dataset: bool = False):
        super().__init__(TrainGenModelState(result_path, name, gen_examples_count), method, reports)
        self.dataset = dataset
        self.number_of_processes = number_of_processes
        self.export_combined_dataset = export_combined_dataset

    def run(self, result_path: Path) -> TrainGenModelState:
        self._set_path(result_path)
        self._fit_model()
        self._save_model()
        self._gen_data()
        self._run_reports()

        return self.state

    def _fit_model(self):
        print_log(f"{self.state.name}: starting to fit the model", True)
        self.method.fit(self.dataset, self.state.result_path)
        print_log(f"{self.state.name}: fitted the model", True)

    def _gen_data(self):
        super()._gen_data()
        self._make_and_export_combined_dataset()

    def _make_combined_dataset(self):
        path = PathBuilder.build(self.state.result_path / 'combined_dataset')
        data = merge_dataclass_objects(list(self.dataset.get_data(batch_size=self.dataset.get_example_count(),
                                                                  return_objects=False)))
        org_data = data.add_fields({'from_gen_model': np.zeros(len(data))}, {'from_gen_model': bool})

        data = merge_dataclass_objects(list(
            self.generated_dataset.get_data(batch_size=self.generated_dataset.get_example_count(),
                                            return_objects=False)))
        gen_data = data.add_fields({'from_gen_model': np.ones(len(data))}, {'from_gen_model': bool})

        combined_data = merge_dataclass_objects([org_data, gen_data], fill_unmatched=True)
        bnp_write_to_file(path / 'batch1.tsv', combined_data)

        self.state.combined_dataset = SequenceDataset.build(
            dataset_file=path / f'combined_{self.state.name}_dataset.yaml',
            types=get_type_dict_from_bnp_object(combined_data),
            filenames=[path / 'batch1.tsv'],
            element_class_name='ReceptorSequence')

    def _make_and_export_combined_dataset(self):
        if self.export_combined_dataset and isinstance(self.dataset, SequenceDataset):
            self._make_combined_dataset()
            export_path = PathBuilder.build(self.state.result_path / 'exported_combined_dataset')
            try:
                AIRRExporter.export(self.state.combined_dataset, export_path)
                self.state.exported_datasets['combined_dataset'] = export_path
            except AssertionError as e:
                logging.warning(f"{TrainGenModelInstruction.__name__}: {self.state.name}: combined dataset could not "
                                f"be exported due to the following error: {e}.")
        else:
            logging.warning(f"{TrainGenModelInstruction.__name__}: {self.state.name}: export_combined_dataset is only "
                            f"supported for sequence datasets at this point.")

    def _run_reports(self):
        super()._run_reports()

        report_path = self._get_reports_path()
        for report in self.reports:
            report.result_path = report_path
            if isinstance(report, TrainGenModelReport):
                report.generated_dataset = self.generated_dataset
                report.original_dataset = self.dataset
                report.model = self.method
                self.state.report_results['instruction_reports'].append(report.generate_report())
            elif isinstance(report, DataReport):
                rep = copy.deepcopy(report)
                rep.result_path = PathBuilder.build(rep.result_path.parent / f"{rep.result_path.name}_original_dataset")
                rep.dataset = self.dataset
                rep.name = rep.name + " (original dataset)"
                self.state.report_results['data_reports'].append(rep.generate_report())

        super()._print_report_summary_log()

    def _save_model(self):
        self.state.model_path = self.method.save_model(self.state.result_path / 'trained_model/')
