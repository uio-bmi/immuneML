import datetime

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

        print(f"{datetime.datetime.now()}: Encoding started...")

        encoded_dataset = encoder.encode(dataset, encoder_params)

        if input_params.store_encoded_data:
            print(f"{datetime.datetime.now()}: Saving encoded dataset to disk.")
            encoder.store(encoded_dataset, encoder_params)

        print(f"{datetime.datetime.now()}: Encoding finished.")

        return encoded_dataset
