from dataclasses import dataclass
from pathlib import Path

from immuneML.IO.dataset_export.AIRRExporter import AIRRExporter
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.hyperparameter_optimization.config.SplitConfig import SplitConfig
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
    train_data_path: Path = None
    test_data_path: Path = None


class SplitDatasetInstruction(Instruction):
    """
    This instruction splits the dataset into two as defined by the instruction parameters. It can be used as a first
    step in clustering to obtain discovery and validation datasets, or to leave out the test dataset for classification.

    For classification, :ref:`TrainMLModel` instruction can be used for more complex data splitting (e.g.,
    nested cross-validation with different data splitting strategies).

    **Specification arguments:**

    - dataset (str): name of the dataset to split, as defined previously in the analysis specification

    - split_config (SplitConfig): the split configuration; split_count has to be 1


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

    """

    def __init__(self, state: SplitDatasetState):
        assert state.split_config.split_count == 1, \
            f"SplitDataset instruction can only be used with 1 split count, got {state.split_config.split_count} instead."
        self.state = state

    def run(self, result_path: Path) -> SplitDatasetState:
        self.state.result_path = PathBuilder.build(result_path / self.state.name)
        paths = [self.state.result_path]

        splitter_params = DataSplitterParams(dataset=self.state.dataset, split_strategy=self.state.split_config.split_strategy,
                                             split_count=self.state.split_config.split_count,
                                             training_percentage=self.state.split_config.training_percentage,
                                             paths=paths, split_config=self.state.split_config)
        train_dataset, test_dataset = DataSplitter.run(splitter_params)
        train_dataset, test_dataset = train_dataset[0], test_dataset[0]

        self.state.train_data_path = self.state.result_path / 'train'
        self.state.test_data_path = self.state.result_path / 'test'

        AIRRExporter.export(train_dataset, self.state.train_data_path)
        AIRRExporter.export(test_dataset, self.state.test_data_path)

        return self.state
