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

        if encoded_dataset.encoded_data.info is None:
            encoded_dataset.encoded_data.info = {}

        if 'sequence_type' not in encoded_dataset.encoded_data.info:
            encoded_dataset.encoded_data.info['sequence_type'] = encoder_params.sequence_type
        if 'region_type' not in encoded_dataset.encoded_data.info:
            encoded_dataset.encoded_data.info['region_type'] = encoder_params.region_type

        print_log(f"Encoding finished.", include_datetime=True)

        return encoded_dataset
