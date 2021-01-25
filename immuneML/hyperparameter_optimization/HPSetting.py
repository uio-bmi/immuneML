from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.ml_methods.MLMethod import MLMethod


class HPSetting:

    def __init__(self, encoder: DatasetEncoder, encoder_params: dict,
                 ml_method: MLMethod, ml_params: dict,
                 preproc_sequence: list, encoder_name: str = None,
                 ml_method_name: str = None, preproc_sequence_name: str = None):

        self.encoder = encoder
        self.encoder_params = encoder_params
        self.ml_method = ml_method
        self.ml_params = ml_params
        self.preproc_sequence = preproc_sequence
        self.encoder_name = encoder_name
        self.ml_method_name = ml_method_name
        self.preproc_sequence_name = preproc_sequence_name

    def get_key(self):
        key = f"{self.encoder_name}_{self.ml_method_name}"
        if self.preproc_sequence is not None and self.preproc_sequence_name is not None:
            key += f"_{self.preproc_sequence_name}"
        return key

    def __str__(self):
        return self.get_key()
