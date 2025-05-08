import abc
from typing import List

import numpy as np

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
    def fit(self, dataset: Dataset = None, design_matrix: np.ndarray = None):
        if dataset is None:
            self.method.fit(design_matrix)
        else:
            self.method.fit(dataset.encoded_data.get_examples_as_np_matrix())

    @abc.abstractmethod
    def transform(self, dataset: Dataset = None, design_matrix: np.ndarray = None):
        if dataset is None:
            return self.method.transform(design_matrix)
        else:
            return self.method.transform(dataset.encoded_data.get_examples_as_np_matrix())

    def fit_transform(self, dataset: Dataset = None, design_matrix: np.ndarray = None):
        if dataset is None:
            return self.method.fit_transform(design_matrix)
        else:
            return self.method.fit_transform(dataset.encoded_data.get_examples_as_np_matrix())

    @abc.abstractmethod
    def get_dimension_names(self) -> List[str]:
        pass

    @classmethod
    def get_title(cls):
        return "Dimensionality Reduction"
