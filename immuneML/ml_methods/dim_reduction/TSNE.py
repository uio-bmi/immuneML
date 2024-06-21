import logging

from sklearn.manifold import TSNE as SklearnTSNE

from immuneML.data_model.dataset.Dataset import Dataset
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

    def fit(self, dataset: Dataset):
        self.method.fit(dataset.encoded_data.examples)

    def transform(self, dataset: Dataset):
        logging.warning(f"{TSNE.__name__}: calling transform method of TSNE, but it only supports fit_transform. "
                        f"Fitting the model and returning the transformed data...")
        return self.method.fit_transform(dataset.encoded_data.examples)

    def fit_transform(self, dataset: Dataset):
        return self.method.fit_transform(dataset.encoded_data.examples)