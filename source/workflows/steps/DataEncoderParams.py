from source.data_model.dataset.Dataset import Dataset
from source.encodings.DatasetEncoder import DatasetEncoder
from source.encodings.EncoderParams import EncoderParams
from source.workflows.steps.StepParams import StepParams


class DataEncoderParams(StepParams):

    def __init__(self, dataset: Dataset, encoder: DatasetEncoder, encoder_params: EncoderParams):
        self.dataset = dataset
        self.encoder = encoder
        self.encoder_params = encoder_params
