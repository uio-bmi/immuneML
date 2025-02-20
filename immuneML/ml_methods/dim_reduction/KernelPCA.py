from typing import List

from sklearn.decomposition import KernelPCA as SklearnPCA

from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.ml_methods.dim_reduction.DimRedMethod import DimRedMethod


class KernelPCA(DimRedMethod):
    """
    Principal component analysis (PCA) method which wraps scikit-learn's KernelPCA, allowing for non-linear dimensionality
    reduction. Input arguments for the method are the
    same as supported by scikit-learn (see `KernelPCA scikit-learn documentation
    <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html>`_ for details).

    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            ml_methods:
                my_kernel_pca:
                    KernelPCA:
                        # arguments as defined by scikit-learn
                        n_components: 5
                        kernel: rbf

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

    def get_dimension_names(self) -> List[str]:
        return [f"PC{i+1}" for i in range(self.method.n_components)]
