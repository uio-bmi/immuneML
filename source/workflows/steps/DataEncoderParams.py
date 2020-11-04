from dataclasses import dataclass

from source.data_model.dataset.Dataset import Dataset
from source.encodings.DatasetEncoder import DatasetEncoder
from source.encodings.EncoderParams import EncoderParams
from source.workflows.steps.StepParams import StepParams


@dataclass
class DataEncoderParams(StepParams):

    dataset: Dataset
    encoder: DatasetEncoder
    encoder_params: EncoderParams
    store_encoded_data: bool
