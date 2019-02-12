from source.encodings.DatasetEncoder import DatasetEncoder
from source.workflows.steps.Step import Step


class DataEncoder(Step):

    @staticmethod
    def run(input_params: dict = None):
        DataEncoder.check_prerequisites(input_params)
        return DataEncoder.perform_step(input_params)

    @staticmethod
    def check_prerequisites(input_params: dict = None):
        assert input_params is not None, "DataEncoderStep: input parameters cannot be None."
        assert "encoder" in input_params and issubclass(input_params["encoder"].__class__, DatasetEncoder), "DataEncoderStep: encoder parameter has to be set to an instance of DatasetEncoder class or to an instance of a subclass of DatasetEncoder."
        assert "encoder_params" in input_params, "DataEncoderStep: encoder_params have to be set with specific parameter for the encoder passed in."
        assert "result_path" in input_params, "DataEncoderStep: result_path for the encoded data has to be set."
        assert "model_path" in input_params, "DataEncoderStep: model_path for the encoded data has to be set."
        assert "learn_model" in input_params, "DataEncoderStep: learn_model for the encoded data has to be set: True for training data and False for test data."
        assert "vectorizer_path" in input_params, "DataEncoderStep: vectorizer_path for the encoded data has to be set."
        assert "pipeline_path" in input_params, "DataEncoderStep: pipeline_path for the encoded data has to be set."
        assert "batch_size" in input_params, "DataEncoderStep: batch_size has to be set."

    @staticmethod
    def perform_step(input_params: dict = None):

        dataset = input_params["dataset"]
        encoder = input_params["encoder"]
        encoder_params = input_params["encoder_params"]
        encoder_params["result_path"] = input_params["result_path"]
        encoder_params["learn_model"] = input_params["learn_model"]
        encoder_params["scaler_path"] = input_params["scaler_path"]
        encoder_params["model_path"] = input_params["model_path"]
        encoder_params["vectorizer_path"] = input_params["vectorizer_path"]
        encoder_params["pipeline_path"] = input_params["pipeline_path"]
        encoder_params["label_configuration"] = input_params["label_configuration"]
        encoder_params["batch_size"] = input_params["batch_size"]

        encoded_dataset = encoder.encode(dataset, encoder_params)

        return encoded_dataset
