import logging
from typing import List

import numpy as np
from sklearn.manifold import TSNE as SklearnTSNE

from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.ml_methods.dim_reduction.DimRedMethod import DimRedMethod


class TSNE(DimRedMethod):
    """
    t-distributed Stochastic Neighbor Embedding (t-SNE) method which wraps scikit-learn's TSNE. It can be useful for
    visualizing high-dimensional data. Input arguments for the method are the
    same as supported by scikit-learn (see `TSNE scikit-learn documentation
    <https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html#sklearn.manifold.TSNE>`_ for details).


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            ml_methods:
                my_tsne:
                    TSNE:
                        # arguments as defined by scikit-learn
                        n_components: 2
                        init: pca

    """
    def __init__(self, name: str = None, **kwargs):
        super().__init__(name)
        self.method_kwargs = kwargs
        self.method = SklearnTSNE(**self.method_kwargs)

    def transform(self, dataset: Dataset = None, design_matrix: np.ndarray = None):
        logging.warning(f"{TSNE.__name__}: calling transform method of TSNE, but it only supports fit_transform. "
                        f"Fitting the model and returning the transformed data...")
        return super().transform(dataset, design_matrix)

    def get_dimension_names(self) -> List[str]:
        return [f"tSNE_dimension_{i+1}" for i in range(self.method.n_components)]