from typing import List

import numpy as np
from sklearn.decomposition import KernelPCA as SklearnKernelPCA
from sklearn.metrics.pairwise import pairwise_kernels

from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.ml_methods.dim_reduction.DimRedMethod import DimRedMethod


class KernelPCA(DimRedMethod):
    """
    Kernel principal component analysis which wraps scikit-learn's KernelPCA, allowing for non-linear dimensionality
    reduction. Input arguments for the method are the same as supported by scikit-learn (see `KernelPCA scikit-learn
    documentation <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html>`_ for
    details), plus two additional immuneML arguments:

    - components (list): which two components (1-indexed) to use for visualization in the
      :ref:`DimensionalityReduction` report. Default: [1, 2].

    - compute_total_variance (bool): if True, computes the total variance in kernel feature space during fit by
      building the full n_samples × n_samples kernel matrix, so that explained variance ratios are expressed as a
      fraction of total kernel-space variance rather than relative to the retained components only. This roughly
      doubles the fit computation time. Default: false.

    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            ml_methods:
                my_kernel_pca:
                    KernelPCA:
                        n_components: 5
                        kernel: rbf
                        components: [3, 4]
                        compute_total_variance: false

    """

    def __init__(self, name: str = None, **kwargs):
        super().__init__(name)
        self.components = kwargs.pop('components', None)
        self._compute_total_variance = kwargs.pop('compute_total_variance', False)
        self._total_kernel_variance = None
        self.method_kwargs = kwargs
        self.method = SklearnKernelPCA(**self.method_kwargs)
        self._validate_components(self.method.n_components)

    def fit(self, dataset: Dataset = None, design_matrix: np.ndarray = None):
        X = dataset.encoded_data.get_examples_as_np_matrix() if dataset is not None else design_matrix
        if self._compute_total_variance:
            kernel = self.method_kwargs.get('kernel', 'linear')
            K = pairwise_kernels(X, metric=kernel, filter_params=True, **self.method_kwargs)
            n = K.shape[0]
            self._total_kernel_variance = float(K.trace() - n * K.mean())
        self.method = self.method.fit(X)

    def get_dimension_names(self) -> List[str]:
        n = getattr(self.method, 'n_components_', None) or self.method.n_components
        return [f"PC{i+1}" for i in range(n)]

    def get_explained_variance_ratio(self):
        if not hasattr(self.method, 'eigenvalues_'):
            return None
        if self._compute_total_variance and self._total_kernel_variance and self._total_kernel_variance > 0:
            return self.method.eigenvalues_ / self._total_kernel_variance
        return None