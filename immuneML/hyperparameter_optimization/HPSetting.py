from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.ml_methods.classifiers.MLMethod import MLMethod
from immuneML.ml_methods.dim_reduction.DimRedMethod import DimRedMethod


class HPSetting:

    def __init__(self, encoder: DatasetEncoder, encoder_params: dict,
                 ml_method: MLMethod, ml_params: dict,
                 preproc_sequence: list, encoder_name: str = None,
                 ml_method_name: str = None, preproc_sequence_name: str = None,
                 dim_reduction_method: DimRedMethod = None,
                 dim_red_params: dict = None,
                 dim_red_name: str = None):

        self.encoder = encoder
        self.encoder_params = encoder_params
        self.ml_method = ml_method
        self.ml_params = ml_params
        self.preproc_sequence = preproc_sequence
        self.encoder_name = encoder_name
        self.ml_method_name = ml_method_name
        self.preproc_sequence_name = preproc_sequence_name
        self.dim_reduction_method = dim_reduction_method
        self.dim_red_params = dim_red_params
        self.dim_red_name = dim_red_name

    def get_key(self):
        key = f"{self.encoder_name}_{self.dim_red_name}_{self.ml_method_name}_{self.preproc_sequence_name}"
        key = key.replace("_None", "")
        return key

    def __str__(self):
        return self.get_key()
