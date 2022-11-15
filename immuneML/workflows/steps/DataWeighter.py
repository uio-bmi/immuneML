import datetime

from immuneML.workflows.steps.DataWeighterParams import DataWeighterParams
from immuneML.workflows.steps.Step import Step
from immuneML.workflows.steps.StepParams import StepParams


class DataWeighter(Step):

    @staticmethod
    def run(input_params: StepParams = None):
        assert isinstance(input_params, DataWeighterParams), \
            "DataWeighter step: input_params have to be an instance of DataWeighterParams class."

        dataset = input_params.dataset.clone()
        weighting_strategy = input_params.weighting_strategy
        weighting_params = input_params.weighting_params

        if weighting_strategy is None:
            return dataset

        print(f"{datetime.datetime.now()}: Computing example weights...")

        example_weights = weighting_strategy.compute_weights(dataset, weighting_params)
        dataset.set_example_weights(example_weights)

        print(f"{datetime.datetime.now()}: Example weights computed.")

        return dataset
