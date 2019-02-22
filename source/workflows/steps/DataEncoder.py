from source.workflows.steps.Step import Step


class DataEncoder(Step):

    @staticmethod
    def run(input_params: dict = None):
        return DataEncoder.perform_step(input_params)

    @staticmethod
    def perform_step(input_params: dict = None):

        dataset = input_params["dataset"]
        encoder = input_params["encoder"]
        encoder_params = input_params["encoder_params"]

        encoded_dataset = encoder.encode(dataset, encoder_params)

        return encoded_dataset
