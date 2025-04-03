import copy
import logging
from dataclasses import field, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from immuneML.IO.dataset_export.AIRRExporter import AIRRExporter
from immuneML.data_model.bnp_util import merge_dataclass_objects, bnp_write_to_file, write_dataset_yaml
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.data_model.datasets.ElementDataset import SequenceDataset
from immuneML.hyperparameter_optimization.config.SplitType import SplitType
from immuneML.ml_methods.generative_models.GenerativeModel import GenerativeModel
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.reports.train_gen_model_reports.TrainGenModelReport import TrainGenModelReport
from immuneML.util.Logger import print_log
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.GenModelInstruction import GenModelState, GenModelInstruction
from immuneML.workflows.steps.data_splitter.DataSplitter import DataSplitter
from immuneML.workflows.steps.data_splitter.DataSplitterParams import DataSplitterParams


@dataclass
class TrainGenModelState:
    result_path: Path = None
    name: str = None
    gen_examples_count: int = None
    model_path: Path = None
    generated_dataset: Dataset = None
    exported_datasets: Dict[str, Path] = field(default_factory=dict)
    report_results: Dict[str, List[ReportResult]] = field(
        default_factory=lambda: {'data_reports': [], 'ml_reports': [], 'instruction_reports': []})
    combined_dataset: Dataset = None
    train_dataset: Dataset = None
    test_dataset: Dataset = None
    training_percentage: float = None


class TrainGenModelInstruction(GenModelInstruction):
    """

    TrainGenModel instruction implements training generative AIRR models on receptor level. Models that can be trained
    for sequence generation are listed under Generative Models section.

    This instruction takes a dataset as input which will be used to train a model, the model itself, and the number of
    sequences to generate to illustrate the applicability of the model. It can also produce reports of the fitted model
    and reports of original and generated sequences.

    To use the generative model previously trained with immuneML, see :ref:`ApplyGenModel` instruction.


    **Specification arguments:**

    - dataset: dataset to use for fitting the generative model; it has to be defined under definitions/datasets

    - method: which model to fit (defined previously under definitions/ml_methods)

    - number_of_processes (int): how many processes to use for fitting the model

    - gen_examples_count (int): how many examples (sequences, repertoires) to generate from the fitted model

    - reports (list): list of report ids (defined under definitions/reports) to apply after fitting a generative model
      and generating gen_examples_count examples; these can be data reports (to be run on generated examples), ML
      reports (to be run on the fitted model)

    - training_percentage (float): percentage of the dataset to use for training the generative model. If set to 1, the
      full dataset will be used for training and the test dataset will be the same as the training dataset. Default
      value is 0.7. When export_combined_dataset is set to True, the splitting of sequences into train, test, and
      generated will be shown in column dataset_split.

    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        instructions:
            my_train_gen_model_inst: # user-defined instruction name
                type: TrainGenModel
                dataset: d1 # defined previously under definitions/datasets
                method: model1 # defined previously under definitions/ml_methods
                gen_examples_count: 100
                number_of_processes: 4
                training_percentage: 0.7
                export_generated_dataset: True
                export_combined_dataset: False
                reports: [data_rep1, ml_rep2]

    """

    MAX_ELEMENT_COUNT_TO_SHOW = 10

    def __init__(self, dataset: Dataset = None, method: GenerativeModel = None,
                 number_of_processes: int = 1, gen_examples_count: int = 100, result_path: Path = None,
                 name: str = None, reports: list = None, export_generated_dataset: bool = True,
                 export_combined_dataset: bool = False, training_percentage: float = None):
        super().__init__(TrainGenModelState(result_path, name, gen_examples_count), method, reports)
        self.dataset = dataset
        self.number_of_processes = number_of_processes
        self.export_generated_dataset = export_generated_dataset
        self.export_combined_dataset = export_combined_dataset
        self.state.training_percentage = training_percentage

    def run(self, result_path: Path) -> TrainGenModelState:
        self._set_path(result_path)
        self._split_dataset()
        self._fit_model()
        self._save_model()
        self._gen_data()
        if self.export_generated_dataset:
            self._export_generated_dataset()
        self._make_and_export_combined_dataset()
        self._run_reports()

        return self.state

    def _split_dataset(self):
        if self.state.training_percentage != 1:
            split_params = DataSplitterParams(dataset=self.dataset, split_strategy=SplitType.RANDOM, split_count=1,
                                              training_percentage=self.state.training_percentage,
                                              paths=[self.state.result_path])
            train_datasets, test_datasets = DataSplitter.run(split_params)
            self.state.train_dataset = train_datasets[0]
            self.state.test_dataset = test_datasets[0]
        else:
            logging.info(f"{TrainGenModelInstruction.__name__}: training_percentage was set to 1 meaning that the full "
                         f"dataset will be used for fitting the generative model. All resulting comparison reports "
                         f"will then use the full original dataset as opposed to independent test dataset if the "
                         f"training percentage was less than 1.")
            self.state.train_dataset = self.dataset
            self.state.test_dataset = self.dataset

    def _fit_model(self):
        print_log(f"{self.state.name}: starting to fit the model", True)
        self.method.fit(self.state.train_dataset, self.state.result_path)
        print_log(f"{self.state.name}: fitted the model", True)

    def _make_combined_dataset(self):
        path = PathBuilder.build(self.state.result_path / 'combined_dataset')

        gen_data = self._get_dataclass_object_from_dataset(self.generated_dataset,
                                                           np.ones(self.state.gen_examples_count),
                                                           np.zeros(self.state.gen_examples_count),
                                                           ['generated'] * self.state.gen_examples_count)

        if self.state.training_percentage < 1:

            org_data = self._get_dataclass_object_from_dataset(self.state.train_dataset,
                                                               np.zeros(self.state.train_dataset.get_example_count()),
                                                               np.ones(self.state.train_dataset.get_example_count()),
                                                               ['train'] * self.state.train_dataset.get_example_count())
            test_data = self._get_dataclass_object_from_dataset(self.state.test_dataset,
                                                                np.zeros(self.state.test_dataset.get_example_count()),
                                                                np.zeros(self.state.test_dataset.get_example_count()),
                                                                ['test'] * self.state.test_dataset.get_example_count())

            combined_data = merge_dataclass_objects([org_data, test_data, gen_data], fill_unmatched=True)
        else:
            org_data = self._get_dataclass_object_from_dataset(self.dataset, np.zeros(self.dataset.get_example_count()),
                                                               np.ones(self.dataset.get_example_count()),
                                                               ['train'] * self.dataset.get_example_count())
            combined_data = merge_dataclass_objects([org_data, gen_data], fill_unmatched=True)

        bnp_write_to_file(path / f'combined_{self.state.name}_dataset.tsv', combined_data)

        metadata_yaml = SequenceDataset.create_metadata_dict(dataset_class=SequenceDataset.__name__,
                                                             filename=f'combined_{self.state.name}_dataset.tsv',
                                                             type_dict=type(combined_data).get_field_type_dict(
                                                                 all_fields=False),
                                                             name=f'combined_{self.state.name}_dataset',
                                                             labels={'gen_model_name': [self.method.name, ''],
                                                                     "from_gen_model": [True, False],
                                                                     "dataset_split": ['train', 'test', 'generated']},)

        write_dataset_yaml(path / f'combined_{self.state.name}_dataset.yaml', metadata_yaml)

        self.state.combined_dataset = SequenceDataset.build(
            metadata_filename=path / f'combined_{self.state.name}_dataset.yaml',
            filename=path / f'combined_{self.state.name}_dataset.tsv', name=f'combined_{self.state.name}_dataset')

    def _get_dataclass_object_from_dataset(self, dataset: Dataset, from_gen_model_vals: np.ndarray,
                                           used_for_training_vals: np.ndarray, dataset_split: list):
        return dataset.data.add_fields(
            {'from_gen_model': np.where(from_gen_model_vals, 'T', 'F'),
             'used_for_training': np.where(used_for_training_vals, 'T', 'F'),
             'dataset_split': dataset_split},
            {'from_gen_model': str, 'used_for_training': str, 'dataset_split': str})

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
        elif self.export_combined_dataset:
            logging.warning(f"{TrainGenModelInstruction.__name__}: {self.state.name}: export_combined_dataset is only "
                            f"supported for sequence datasets at this point.")

    def _run_reports(self):
        super()._run_reports()

        report_path = self._get_reports_path()
        for report in self.reports:
            original_dataset = self.state.train_dataset if self.state.training_percentage != 1 else self.dataset
            report.result_path = report_path
            if isinstance(report, TrainGenModelReport):
                report.generated_dataset = self.generated_dataset
                report.original_dataset = original_dataset
                report.model = self.method
                self.state.report_results['instruction_reports'].append(report.generate_report())
            elif isinstance(report, DataReport):
                rep = copy.deepcopy(report)
                rep.result_path = PathBuilder.build(rep.result_path.parent / f"{rep.result_path.name}_original_dataset")
                rep.dataset = original_dataset
                rep.name = rep.name + " (original dataset)"
                self.state.report_results['data_reports'].append(rep.generate_report())

        self._print_report_summary_log()

    def _save_model(self):
        self.state.model_path = self.method.save_model(self.state.result_path / 'trained_model/')
