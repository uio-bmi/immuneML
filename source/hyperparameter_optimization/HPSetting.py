import hashlib

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

    def get_key(self):
        return "{}_{}_{}".format(self.encoder.__class__.__name__ if isinstance(self.encoder, DatasetEncoder) else self.encoder.__name__,
                                 self.ml_method.__class__.__name__,
                                 hashlib.md5((str(self.encoder_params) +
                                              str(self.ml_params) +
                                              str(self.preproc_sequence)).encode()).hexdigest())

    def __str__(self):
        return self.get_key()
