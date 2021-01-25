import abc
from pathlib import Path

from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset


class ModelCreatorStrategy(metaclass=abc.ABCMeta):
    """
    Defines how word2vec model can be created by defining different contexts for k-mers
    """
    @abc.abstractmethod
    def create_model(self, dataset: RepertoireDataset, k: int, vector_size: int, batch_size: int, model_path: Path):
        pass
