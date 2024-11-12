import abc

from immuneML.data_model.datasets.Dataset import Dataset


class DimRedMethod:
    """

    Dimensionality reduction methods are algorithms which can be used to reduce the dimensionality
    of encoded datasets, in order to uncover and analyze patterns present in the data.

    These methods can be used in the :ref:`ExploratoryAnalysis` and :ref:`Clustering` instructions.
    """

    DOCS_TITLE = "Dimensionality reduction methods"

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
