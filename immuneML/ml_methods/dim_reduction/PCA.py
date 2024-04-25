from sklearn.decomposition import PCA as SklearnPCA

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.ml_methods.dim_reduction.DimRedMethod import DimRedMethod


class PCA(DimRedMethod):
    """
    Principal component analysis (PCA) method which wraps scikit-learn's PCA. Input arguments for the method are the
    same as supported by scikit-learn (see `PCA scikit-learn documentation
    <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA>`_ for details).

    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            ml_methods:
                my_pca:
                    PCA:
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
        return self.method.fit_transform(dataset.encoded_data.get_examples_as_np_matrix())
