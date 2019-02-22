import abc

from source.data_model.dataset.Dataset import Dataset
from source.encodings.EncoderParams import EncoderParams


class ModelCreatorStrategy(metaclass=abc.ABCMeta):
    """
    Defines how word2vec model can be created by defining different contexts for k-mers
    """
    @abc.abstractmethod
    def create_model(self, dataset: Dataset, params: EncoderParams, model_path):
        pass
