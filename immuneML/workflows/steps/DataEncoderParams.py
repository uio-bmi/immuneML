from dataclasses import dataclass

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.workflows.steps.StepParams import StepParams


@dataclass
class DataEncoderParams(StepParams):

    dataset: Dataset
    encoder: DatasetEncoder
    encoder_params: EncoderParams
    store_encoded_data: bool
