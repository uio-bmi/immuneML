from source.encodings.DatasetEncoder import DatasetEncoder
from source.ml_methods.MLMethod import MLMethod


class HPSetting:

    def __init__(self, encoder: DatasetEncoder, encoder_params: dict,
                 ml_method: MLMethod, ml_params: dict,
                 preproc_sequence: list):

        self.encoder = encoder
        self.encoder_params = encoder_params
        self.ml_method = ml_method
        self.ml_params = ml_params
        self.preproc_sequence = preproc_sequence

    def __str__(self):
        raise NotImplementedError
