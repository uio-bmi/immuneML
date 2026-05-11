import abc
import logging
from abc import ABC
from typing import List

import numpy as np

from immuneML.data_model.datasets.Dataset import Dataset


class DimRedMethod(ABC):
    """
    Dimensionality reduction methods are algorithms which can be used to reduce the dimensionality
    of encoded datasets, in order to uncover and analyze patterns present in the data.

    These methods can be used in the :ref:`ExploratoryAnalysis` and :ref:`Clustering` instructions.
    """

    DOCS_TITLE = "Dimensionality reduction methods"

    def __init__(self, name: str = None):
        self.method = None
        self.name = name
        self.components = None

    def _validate_components(self, n_components=None):
        if self.components is None:
            return
        assert isinstance(self.components, list) and len(self.components) == 2 \
               and all(isinstance(c, int) and c >= 1 for c in self.components), \
            (f"{self.__class__.__name__}: 'components' must be a list of exactly 2 positive integers "
             f"(1-indexed), e.g. [3, 4]. Got: {self.components}.")
        if n_components is not None and isinstance(n_components, int):
            assert max(self.components) <= n_components, \
                (f"{self.__class__.__name__}: 'components' {self.components} requires n_components >= "
                 f"{max(self.components)}, but n_components is set to {n_components}.")

    def fit(self, dataset: Dataset = None, design_matrix: np.ndarray = None):
        if dataset is None:
            self.method = self.method.fit(design_matrix)
        else:
            self.method = self.method.fit(dataset.encoded_data.get_examples_as_np_matrix())

    def transform(self, dataset: Dataset = None, design_matrix: np.ndarray = None):
        if dataset is None:
            return self.method.transform(design_matrix)
        else:
            return self.method.transform(dataset.encoded_data.get_examples_as_np_matrix())

    def fit_transform(self, dataset: Dataset = None, design_matrix: np.ndarray = None):
        self.fit(dataset, design_matrix)
        return self.transform(dataset, design_matrix)

    def inverse_transform(self, transformed_data):
        try:
            return self.method.inverse_transform(transformed_data)
        except Exception:
            logging.warning(f"{self.__class__.__name__}: inverse transformation is not supported by this method.")
            return None

    @abc.abstractmethod
    def get_dimension_names(self) -> List[str]:
        pass

    def get_explained_variance_ratio(self):
        return None

    @classmethod
    def get_title(cls):
        return "Dimensionality Reduction"