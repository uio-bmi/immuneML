import scipy
from sklearn.decomposition import PCA as SklearnPCA

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.ml_methods.dim_reduction.DimRedMethod import DimRedMethod


class PCA(DimRedMethod):
    """
    Principal component analysis (PCA) method which wraps scikit-learn's PCA. Input arguments for the method are the
    same as supported by scikit-learn (see `PCA scikit-learn documentation
    <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA>`_ for details).

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_pca: # user-defined name of the dimensionality reduction method
            PCA: # name of the class
                # arguments as defined by scikit-learn
                n_components: 2

    """

    def __init__(self, name: str = None, **kwargs):
        super().__init__(name)
        self.method_kwargs = kwargs
        self.method = SklearnPCA(**self.method_kwargs)

    def fit(self, dataset: Dataset):
        self.method.fit(dataset.encoded_data.examples)

    def transform(self, dataset: Dataset):
        return self.method.transform(dataset.encoded_data.examples)

    def fit_transform(self, dataset: Dataset):
        data = dataset.encoded_data.examples.toarray() if scipy.sparse.issparse(dataset.encoded_data.examples) \
            else dataset.encoded_data.examples

        return self.method.fit_transform(data)
