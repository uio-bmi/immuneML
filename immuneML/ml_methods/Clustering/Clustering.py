from abc import ABC

from immuneML.ml_methods.UnsupervisedSklearnMethod import UnsupervisedSklearnMethod


class Clustering(UnsupervisedSklearnMethod, ABC):
    def __init__(self, parameter_grid: dict = None, parameters: dict = None):
        _parameters = parameters if parameters is not None else {}
        _parameter_grid = parameter_grid if parameter_grid is not None else {}

        super().__init__(parameter_grid=_parameter_grid, parameters=_parameters)

    def get_params(self):
        params = self.model.get_params()
        return params

    def get_compatible_encoders(self):
        from immuneML.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder
        from immuneML.encodings.onehot.OneHotEncoder import OneHotEncoder
        from immuneML.encodings.word2vec.W2VSequenceEncoder import W2VSequenceEncoder
        from immuneML.encodings.distance_encoding.TCRdistEncoder import TCRdistEncoder

        return [KmerFrequencyEncoder, OneHotEncoder, W2VSequenceEncoder, TCRdistEncoder]
