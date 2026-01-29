import pytest
import numpy as np

from immuneML.ml_methods.dim_reduction.PCA import PCA
from immuneML.ml_methods.dim_reduction.KernelPCA import KernelPCA
from immuneML.ml_methods.dim_reduction.UMAP import UMAP
from immuneML.ml_methods.dim_reduction.TSNE import TSNE


@pytest.fixture
def sample_data():
    np.random.seed(42)
    return np.random.rand(50, 10)


class TestPCA:
    def test_fit_transform_fits_method(self, sample_data):
        pca = PCA(n_components=2)
        result = pca.fit_transform(design_matrix=sample_data)

        assert result.shape == (50, 2)
        assert hasattr(pca.method, 'components_')
        assert pca.method.components_ is not None

    def test_transform_after_fit_transform(self, sample_data):
        pca = PCA(n_components=2)
        pca.fit_transform(design_matrix=sample_data)

        new_data = np.random.rand(10, 10)
        result = pca.transform(design_matrix=new_data)

        assert result.shape == (10, 2)

    def test_get_dimension_names(self):
        pca = PCA(n_components=3)
        assert pca.get_dimension_names() == ['PC1', 'PC2', 'PC3']


class TestKernelPCA:
    def test_fit_transform_fits_method(self, sample_data):
        kpca = KernelPCA(n_components=2, kernel='rbf')
        result = kpca.fit_transform(design_matrix=sample_data)

        assert result.shape == (50, 2)
        assert hasattr(kpca.method, 'eigenvalues_')
        assert kpca.method.eigenvalues_ is not None

    def test_get_dimension_names(self):
        kpca = KernelPCA(n_components=2)
        assert kpca.get_dimension_names() == ['PC1', 'PC2']


class TestUMAP:
    def test_fit_transform_fits_method(self, sample_data):
        umap_method = UMAP(n_components=2, n_neighbors=5, min_dist=0.1)
        result = umap_method.fit_transform(design_matrix=sample_data)

        assert result.shape == (50, 2)
        assert hasattr(umap_method.method, 'embedding_')
        assert umap_method.method.embedding_ is not None

    def test_get_dimension_names(self):
        umap_method = UMAP(n_components=2)
        assert umap_method.get_dimension_names() == ['UMAP_dimension_1', 'UMAP_dimension_2']


class TestTSNE:
    def test_fit_transform_fits_method(self, sample_data):
        tsne = TSNE(n_components=2, perplexity=5)
        result = tsne.fit_transform(design_matrix=sample_data)

        assert result.shape == (50, 2)
        assert hasattr(tsne.method, 'embedding_')
        assert tsne.method.embedding_ is not None

    def test_get_dimension_names(self):
        tsne = TSNE(n_components=2)
        assert tsne.get_dimension_names() == ['tSNE_1', 'tSNE_2']