from typing import List

from sklearn.decomposition import PCA as SklearnPCA

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

    def get_dimension_names(self) -> List[str]:
        return [f"PC{i+1}" for i in range(self.method.n_components)]
