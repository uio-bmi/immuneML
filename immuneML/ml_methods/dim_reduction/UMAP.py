import umap

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.ml_methods.dim_reduction.DimRedMethod import DimRedMethod


class UMAP(DimRedMethod):
    """
    Uniform manifold approximation and projection (UMAP) method which wraps umap-learn's UMAP. Input arguments for the method are the
    same as supported by umap-learn (see `UMAP in the umap-learn documentation
    <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA>`_ for details).

    Note that when providing the arguments for UMAP in the immuneML's specification, it is not possible to set
    functions as input values (e.g., for the metric parameter, it has to be one of the predefined metrics available
    in umap-learn).

    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            ml_methods:
                my_umap:
                    UMAP:
                        # arguments as defined by scikit-learn
                        n_components: 2
                        n_neighbors: 15
                        metric: euclidean

    """

    def __init__(self, name: str = None, **kwargs):
        super().__init__(name)
        self.method_kwargs = kwargs
        self.method = umap.UMAP(**kwargs)

    def fit(self, dataset: Dataset):
        self.method.fit(dataset.encoded_data.examples)

    def transform(self, dataset: Dataset):
        return self.method.transform(dataset.encoded_data.examples)

    def fit_transform(self, dataset: Dataset):
        return self.method.fit_transform(dataset.encoded_data.examples)
