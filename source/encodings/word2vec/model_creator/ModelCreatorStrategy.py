import abc

from source.data_model.dataset.Dataset import Dataset


class ModelCreatorStrategy(metaclass=abc.ABCMeta):
    """
    Defines how word2vec model can be created by defining different contexts for k-mers
    """
    @abc.abstractmethod
    def create_model(self, dataset: Dataset, params: dict, model_path):
        pass
