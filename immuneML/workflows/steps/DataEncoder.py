from immuneML.util.Logger import print_log
from immuneML.workflows.steps.DataEncoderParams import DataEncoderParams
from immuneML.workflows.steps.Step import Step
from immuneML.workflows.steps.StepParams import StepParams


class DataEncoder(Step):

    @staticmethod
    def run(input_params: StepParams = None):
        assert isinstance(input_params, DataEncoderParams), \
            "DataEncoder step: input_params have to be an instance of DataEncoderParams class."

        dataset = input_params.dataset
        encoder = input_params.encoder
        encoder_params = input_params.encoder_params

        print_log(f"Encoding started...", include_datetime=True)

        encoded_dataset = encoder.encode(dataset, encoder_params)

        print_log(f"Encoding finished.", include_datetime=True)

        return encoded_dataset
