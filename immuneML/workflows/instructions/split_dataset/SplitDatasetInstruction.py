from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from immuneML.IO.dataset_export.AIRRExporter import AIRRExporter
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.hyperparameter_optimization.config.SplitConfig import SplitConfig
from immuneML.hyperparameter_optimization.config.SplitType import SplitType
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.Instruction import Instruction
from immuneML.workflows.steps.data_splitter.DataSplitter import DataSplitter
from immuneML.workflows.steps.data_splitter.DataSplitterParams import DataSplitterParams


@dataclass
class SplitDatasetState:
    dataset: Dataset
    split_config: SplitConfig
    name: str = None
    result_path: Path = None
    train_data_paths: List[Path] = field(default_factory=list)
    test_data_paths: List[Path] = field(default_factory=list)


class SplitDatasetInstruction(Instruction):
    """
    This instruction splits the dataset into two as defined by the instruction parameters. It can be used as a first
    step in clustering to obtain discovery and validation datasets, or to leave out the test dataset for classification.

    For classification, :ref:`TrainMLModel` instruction can be used for more complex data splitting (e.g.,
    nested cross-validation with different data splitting strategies).

    **Specification arguments:**

    - dataset (str): name of the dataset to split, as defined previously in the analysis specification

    - split_config (SplitConfig): the split configuration; the split strategy can be random, manual or group_k_fold.
      For random and manual, split_count has to be 1 and the training and test datasets are exported to `train/` and
      `test/` under the instruction's result folder. For group_k_fold, split_count is the number of folds (at least 2),
      and each split is exported to `split_<i>/train/` and `split_<i>/test/`, together with `group_assignment.csv`
      listing for each group whether it is in the test set, the training set, or always in the training set
      (train_only_groups). See SplitConfig for the details of each split strategy.


    **YAML specification:**

    .. code-block:: yaml

        instructions:
            split_dataset1:
                type: SplitDataset
                dataset: d1
                split_config:
                    split_count: 1
                    split_strategy: random
                    training_percentage: 0.5

            split_dataset_by_study: # each study is in the test set of exactly one of the 3 splits
                type: SplitDataset
                dataset: d1
                split_config:
                    split_count: 3
                    split_strategy: group_k_fold
                    group_k_fold_config:
                        group_label: study_id
                        delimiter: ';' # optional, if an example can belong to several studies
                        train_only_groups: [study1] # optional, groups that are never held out

    """

    def __init__(self, state: SplitDatasetState):
        assert state.split_config.split_count == 1 or state.split_config.split_strategy == SplitType.GROUP_K_FOLD, \
            (f"SplitDataset instruction can only be used with 1 split count for split strategy "
             f"{state.split_config.split_strategy.name}, got {state.split_config.split_count} instead.")
        self.state = state

    def run(self, result_path: Path) -> SplitDatasetState:
        self.state.result_path = PathBuilder.build(result_path / self.state.name)
        split_count = self.state.split_config.split_count
        # a single split is written directly under the result path, as train/ and test/
        paths = [self.state.result_path] if split_count == 1 \
            else [PathBuilder.build(self.state.result_path / f"split_{i + 1}") for i in range(split_count)]

        splitter_params = DataSplitterParams(dataset=self.state.dataset, split_strategy=self.state.split_config.split_strategy,
                                             split_count=split_count,
                                             training_percentage=self.state.split_config.training_percentage,
                                             paths=paths, split_config=self.state.split_config)
        train_datasets, test_datasets = DataSplitter.run(splitter_params)

        for path, train_dataset, test_dataset in zip(paths, train_datasets, test_datasets):
            self.state.train_data_paths.append(path / 'train')
            self.state.test_data_paths.append(path / 'test')

            AIRRExporter.export(train_dataset, path / 'train')
            AIRRExporter.export(test_dataset, path / 'test')

        return self.state
