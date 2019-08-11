import abc

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.encodings.EncoderParams import EncoderParams


class ModelCreatorStrategy(metaclass=abc.ABCMeta):
    """
    Defines how word2vec model can be created by defining different contexts for k-mers
    """
    @abc.abstractmethod
    def create_model(self, dataset: RepertoireDataset, params: EncoderParams, model_path):
        pass
