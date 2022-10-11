from dataclasses import dataclass

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.example_weighting.ExampleWeightingParams import ExampleWeightingParams
from immuneML.example_weighting.ExampleWeightingStrategy import ExampleWeightingStrategy
from immuneML.workflows.steps.StepParams import StepParams


@dataclass
class DataWeighterParams(StepParams):

    dataset: Dataset
    weighting_strategy: ExampleWeightingStrategy
    weighting_params: ExampleWeightingParams
