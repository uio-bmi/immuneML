import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.hyperparameter_optimization.config.SampleConfig import SampleConfig
from immuneML.workflows.steps.Step import Step
from immuneML.workflows.steps.StepParams import StepParams

@dataclass
class DataSamplerParams(StepParams):
    """Parameters for DataSampler step."""
    dataset: Dataset
    split_config: SampleConfig
    paths: List[Path] = None

class DataSampler(Step):
    """DataSampler step that samples data from datasets based on specified strategies. Currently, only random
    subsampling without replacement is supported."""

    @staticmethod
    def run(input_params: DataSamplerParams = None):
        subsampled_datasets = []

        if input_params.split_config.random_seed:
            random.seed(input_params.split_config.random_seed)

        for i in range(input_params.split_config.split_count):
            indices = list(range(input_params.dataset.get_example_count()))
            sampled_indices = random.sample(indices, k=round(len(indices) * input_params.split_config.percentage))
            sampled_dataset = input_params.dataset.make_subset(sampled_indices, input_params.paths[i], Dataset.SUBSAMPLED)
            subsampled_datasets.append(sampled_dataset)

        return subsampled_datasets
