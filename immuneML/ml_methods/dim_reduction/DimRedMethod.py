import abc

from immuneML.data_model.dataset.Dataset import Dataset


class DimRedMethod:

    def __init__(self, name: str = None):
        self.method = None
        self.name = name

    @abc.abstractmethod
    def fit(self, dataset: Dataset):
        pass

    @abc.abstractmethod
    def transform(self, dataset: Dataset):
        pass

    @abc.abstractmethod
    def fit_transform(self, dataset: Dataset):
        pass

    @classmethod
    def get_title(cls):
        return "Dimensionality Reduction"
