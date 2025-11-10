import copy
import logging
from dataclasses import field, dataclass
from pathlib import Path
from typing import Dict, List

from immuneML.IO.dataset_export.AIRRExporter import AIRRExporter
from immuneML.data_model.bnp_util import merge_dataclass_objects, bnp_write_to_file, write_dataset_yaml
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.data_model.datasets.ElementDataset import SequenceDataset
from immuneML.environment.SequenceType import SequenceType
from immuneML.hyperparameter_optimization.config.ManualSplitConfig import ManualSplitConfig
from immuneML.hyperparameter_optimization.config.SplitConfig import SplitConfig
from immuneML.hyperparameter_optimization.config.SplitType import SplitType
from immuneML.ml_methods.generative_models.GenerativeModel import GenerativeModel
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.reports.ml_reports.MLReport import MLReport
from immuneML.reports.train_gen_model_reports.TrainGenModelReport import TrainGenModelReport
from immuneML.reports.gen_model_reports.GenModelReport import GenModelReport
from immuneML.util.Logger import print_log
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.Instruction import Instruction
from immuneML.workflows.steps.data_splitter.DataSplitter import DataSplitter
from immuneML.workflows.steps.data_splitter.DataSplitterParams import DataSplitterParams


@dataclass
class TrainGenModelState:
    result_path: Path = None
    name: str = None
    gen_examples_count: int = None
    model_paths: Dict[str, Path] = field(default_factory=dict)
    generated_dataset: Dataset = None
    exported_datasets: Dict[str, Path] = field(default_factory=dict)
    report_results: Dict[str, List[ReportResult]] = field(
        default_factory=lambda: {'data_reports': [], 'ml_reports': [], 'gen_ml_reports': [], 'instruction_reports': []})
    combined_dataset: Dataset = None
    train_dataset: Dataset = None
    test_dataset: Dataset = None
    training_percentage: float = None
    split_strategy: SplitType = None
    split_config: ManualSplitConfig = None


class TrainGenModelInstruction(Instruction):
    """

    TrainGenModel instruction implements training generative AIRR models on receptor level. Models that can be trained
    for sequence generation are listed under Generative Models section.

    This instruction takes a dataset as input which will be used to train a model, the model itself, and the number of
    sequences to generate to illustrate the applicability of the model. It can also produce reports of the fitted model
    and reports of original and generated sequences.

    To use the generative model previously trained with immuneML, see :ref:`ApplyGenModel` instruction.


    **Specification arguments:**

    - dataset: dataset to use for fitting the generative model; it has to be defined under definitions/datasets

    - methods: which methods to fit (defined previously under definitions/ml_methods); for compatibility with previous
      versions 'method' with a single method can also be used, but the single method option will be removed in the
      future.

    - number_of_processes (int): how many processes to use for fitting the model

    - gen_examples_count (int): how many examples (sequences, repertoires) to generate from the fitted model

    - reports (list): list of report ids (defined under definitions/reports) to apply after fitting a generative model
      and generating gen_examples_count examples; these can be data reports (to be run on generated examples), ML
      reports (to be run on the fitted model)

    - split_strategy (str): strategy to use for splitting the dataset into training and test datasets; valid options are
      RANDOM and MANUAL (in which case train and test set are fixed); default is RANDOM

    - training_percentage (float): percentage of the dataset to use for training the generative model if split_strategy
      parameter is RANDOM. If set to 1, the
      full dataset will be used for training and the test dataset will be the same as the training dataset. Default
      value is 0.7. When export_combined_dataset is set to True, the splitting of sequences into train, test, and
      generated will be shown in column dataset_split.

    - manual_config (dict): if split_strategy is set to MANUAL, this parameter can be used to specify the ids of examples
      that should be in train and test sets; the paths to csv files with ids for train and test data should be provided
      under keys 'train_metadata_path' and 'test_metadata_path'

    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        instructions:
            my_train_gen_model_inst: # user-defined instruction name
                type: TrainGenModel
                dataset: d1 # defined previously under definitions/datasets
                methods: [model1] # defined previously under definitions/ml_methods
                gen_examples_count: 100
                number_of_processes: 4
                training_percentage: 0.7
                split_strategy: RANDOM # optional, default is RANDOM
                export_generated_dataset: True
                export_combined_dataset: False
                reports: [data_rep1, ml_rep2]

            my_train_gen_model_with_manual_split: # another instruction example
                type: TrainGenModel
                dataset: d1 # defined previously under definitions/datasets
                methods: [m1, m2]
                gen_examples_count: 100
                split_strategy: MANUAL
                split_config:
                    train_metadata_path: path/to/train_metadata.csv # path to csv file with ids of examples in train set
                    test_metadata_path: path/to/test_metadata.csv # path to csv file with ids of examples in test set
                export_generated_dataset: True
                export_combined_dataset: False
                reports: [data_rep1, ml_rep2]

    """

    MAX_ELEMENT_COUNT_TO_SHOW = 10

    def __init__(self, dataset: Dataset = None, methods: List[GenerativeModel] = None,
                 number_of_processes: int = 1, gen_examples_count: int = 100, result_path: Path = None,
                 name: str = None, reports: list = None, export_generated_dataset: bool = True,
                 export_combined_dataset: bool = False, training_percentage: float = None,
                 split_strategy: SplitType = SplitType.RANDOM, split_config: ManualSplitConfig = None):
        self.state = TrainGenModelState(result_path, name, gen_examples_count, training_percentage=training_percentage,
                                        split_config= split_config, split_strategy=split_strategy)
        self.methods = methods
        self.reports = reports
        self.dataset = dataset
        self.number_of_processes = number_of_processes
        self.export_generated_dataset = export_generated_dataset
        self.export_combined_dataset = export_combined_dataset
        self.generated_datasets = {}

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
        if self.state.split_strategy == SplitType.RANDOM:
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
        elif self.state.split_strategy == SplitType.MANUAL:
            split_params = DataSplitterParams(self.dataset, self.state.split_strategy, 1,
                                              paths=[self.state.result_path],
                                              split_config=SplitConfig(self.state.split_strategy, split_count=1,
                                                                       manual_config=self.state.split_config))
            train_datasets, test_datasets = DataSplitter.manual_split(split_params)
            self.state.train_dataset = train_datasets[0]
            self.state.test_dataset = test_datasets[0]
        else:
            raise ValueError(f"{TrainGenModelInstruction.__name__}: {self.state.name}: "
                             f"split_strategy {self.state.split_strategy} is not supported for TrainGenModel instruction.")

    def _fit_model(self):
        for ind, method in enumerate(self.methods):
            print_log(f"{self.state.name}: starting to fit the model {method.name} ({ind + 1}/{len(self.methods)})",
                      True)
            method.fit(self.state.train_dataset, self.state.result_path)
            print_log(f"{self.state.name}: fitted the model {method.name} ({ind + 1}/{len(self.methods)})", True)

    def _make_combined_dataset(self):
        path = PathBuilder.build(self.state.result_path / 'combined_dataset')

        gen_data_list = []
        data_origin_list = []

        for model_name, gen_dataset in self.generated_datasets.items():
            gen_data = self._get_dataclass_object_from_dataset(gen_dataset,
                                                               [model_name] * self.state.gen_examples_count)
            gen_data_list.append(gen_data)
            data_origin_list.append(model_name)

        if self.state.training_percentage < 1:
            data_origin_list.extend(['original_train', 'original_test'])

            org_data = self._get_dataclass_object_from_dataset(self.state.train_dataset,
                                                               ['original_train'] * self.state.train_dataset.get_example_count())
            test_data = self._get_dataclass_object_from_dataset(self.state.test_dataset,
                                                                ['original_test'] * self.state.test_dataset.get_example_count())

            combined_data = merge_dataclass_objects([org_data, test_data] + gen_data_list, fill_unmatched=True)
        else:
            data_origin_list.extend(['original_train'])
            org_data = self._get_dataclass_object_from_dataset(self.dataset,
                                                               ['original_train'] * self.dataset.get_example_count())
            combined_data = merge_dataclass_objects([org_data] + gen_data_list, fill_unmatched=True)

        labels = {'gen_model_name': [''] + [method.name for method in self.methods],
                  "from_gen_model": [True, False], 'data_origin': data_origin_list}

        self.state.combined_dataset = SequenceDataset.build_from_dataclass_object(combined_data, path=path,
                                                                                  name=f'combined_{self.state.name}_dataset',
                                                                                  labels=labels)

    def _get_dataclass_object_from_dataset(self, dataset: Dataset, data_origin: list):
        return dataset.data.add_fields({'data_origin': data_origin}, {'data_origin': str})

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
        self._run_reports_main()

        report_path = self._get_reports_path()
        for report in self.reports:
            original_dataset = self.state.train_dataset if self.state.training_percentage != 1 else self.dataset
            report.result_path = report_path
            if isinstance(report, TrainGenModelReport):
                for method in self.methods:
                    report.generated_dataset = self.generated_datasets[method.name]
                    report.original_dataset = original_dataset
                    report.model = method
                    self.state.report_results['instruction_reports'].append(report.generate_report())
            elif isinstance(report, DataReport):
                rep = copy.deepcopy(report)
                rep.result_path = PathBuilder.build(rep.result_path.parent / f"{rep.result_path.name}_original_dataset")
                rep.dataset = original_dataset
                rep.name = rep.name + " (original dataset)"
                self.state.report_results['data_reports'].append(rep.generate_report())
            elif isinstance(report, GenModelReport):
                for method in self.methods:
                    rep = copy.deepcopy(report)
                    rep.result_path = PathBuilder.build(rep.result_path.parent / f"{rep.result_path.name}_original_dataset")
                    rep.dataset = original_dataset
                    rep.model = method
                    rep.name = rep.name + " (original dataset)"
                    self.state.report_results['gen_ml_reports'].append(rep.generate_report())

        self._print_report_summary_log()

    def _save_model(self):
        for method in self.methods:
            self.state.model_paths[method.name] = method.save_model(
                self.state.result_path / f'trained_model_{method.name}/')

    def _gen_data(self):
        for method in self.methods:
            dataset = method.generate_sequences(self.state.gen_examples_count, 1,
                                                self.state.result_path / 'generated_sequences' / method.name,
                                                SequenceType.AMINO_ACID, False)

            self.generated_datasets[method.name] = dataset
            print_log(f"{self.state.name}: generated {self.state.gen_examples_count} examples from the fitted model",
                      True)

        self.state.generated_dataset = self.merge_datasets(list(self.generated_datasets.values()),
                                                           self.state.result_path / 'combined_generated_dataset')

    def merge_datasets(self, datasets: List[SequenceDataset], result_path: Path) -> SequenceDataset:
        if len(datasets) == 1:
            return datasets[0]
        else:
            merged_data = merge_dataclass_objects([dataset.data for dataset in datasets])
            merged_dataset = SequenceDataset.build_from_dataclass_object(merged_data, PathBuilder.build(result_path))
            return merged_dataset

    def _export_generated_dataset(self):
        AIRRExporter.export(self.state.generated_dataset, self.state.result_path / f'exported_gen_dataset')
        self.state.exported_datasets['generated_dataset'] = self.state.result_path / 'exported_gen_dataset'

    def _run_reports_main(self):
        report_path = self._get_reports_path()
        for report in self.reports:
            report.result_path = report_path
            if isinstance(report, DataReport):
                for method in self.methods:
                    rep = copy.deepcopy(report)
                    rep.dataset = self.generated_datasets[method.name]
                    rep.result_path = PathBuilder.build(
                        rep.result_path.parent / f"{rep.result_path.name}_{method.name}")
                    rep.name = rep.name + " (generated dataset from " + method.name + ")"
                    self.state.report_results['data_reports'].append(rep.generate_report())
            elif isinstance(report, MLReport):
                for method in self.methods:
                    rep = copy.deepcopy(report)
                    rep.method = method
                    rep.name = rep.name + " (from " + method.name + ")"
                    self.state.report_results['ml_reports'].append(rep.generate_report())

    def _print_report_summary_log(self):
        if len(self.reports) > 0:
            gen_rep_count = len(self.state.report_results['ml_reports']) + len(
                self.state.report_results['data_reports']) + len(self.state.report_results['gen_ml_reports'])
            print_log(f"{self.state.name}: generated {gen_rep_count} reports.", True)

    def _get_reports_path(self) -> Path:
        return PathBuilder.build(self.state.result_path / 'reports')

    def _set_path(self, result_path):
        self.state.result_path = PathBuilder.build(result_path / self.state.name)
