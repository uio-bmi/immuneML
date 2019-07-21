from source.workflows.steps.DataEncoderParams import DataEncoderParams
from source.workflows.steps.Step import Step
from source.workflows.steps.StepParams import StepParams


class DataEncoder(Step):

    @staticmethod
    def run(input_params: StepParams = None):
        assert isinstance(input_params, DataEncoderParams), \
            "DataEncoder step: input_params have to be an instance of DataEncoderParams class."

        dataset = input_params.dataset
        encoder = input_params.encoder
        encoder_params = input_params.encoder_params

        encoded_dataset = encoder.encode(dataset, encoder_params)

        return encoded_dataset
