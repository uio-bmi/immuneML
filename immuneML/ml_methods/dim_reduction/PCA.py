from typing import List

from sklearn.decomposition import PCA as SklearnPCA

from immuneML.ml_methods.dim_reduction.DimRedMethod import DimRedMethod


class PCA(DimRedMethod):
    """
    Principal component analysis (PCA) method which wraps scikit-learn's PCA. Input arguments for the method are the
    same as supported by scikit-learn (see `PCA scikit-learn documentation
    <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA>`_ for details),
    plus one additional immuneML argument:

    - components (list): which two components (1-indexed) to use for visualization in the
      :ref:`DimensionalityReduction` report. Default: [1, 2].

    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            ml_methods:
                my_pca:
                    PCA:
                        n_components: 5
                        components: [3, 4]

    """

    def __init__(self, name: str = None, **kwargs):
        super().__init__(name)
        self.components = kwargs.pop('components', None)
        self.method_kwargs = kwargs
        self.method = SklearnPCA(**self.method_kwargs)
        self._validate_components(self.method.n_components)

    def get_dimension_names(self) -> List[str]:
        n = getattr(self.method, 'n_components_', None) or self.method.n_components
        return [f"PC{i+1}" for i in range(n)]

    def get_explained_variance_ratio(self):
        if hasattr(self.method, 'explained_variance_ratio_'):
            return self.method.explained_variance_ratio_
        return None