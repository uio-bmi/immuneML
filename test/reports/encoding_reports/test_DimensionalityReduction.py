import shutil
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from immuneML.data_model.EncodedData import EncodedData
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.ml_methods.dim_reduction.KernelPCA import KernelPCA
from immuneML.ml_methods.dim_reduction.PCA import PCA
from immuneML.ml_methods.dim_reduction.TSNE import TSNE
from immuneML.ml_methods.dim_reduction.UMAP import UMAP
from immuneML.reports.encoding_reports.DimensionalityReduction import DimensionalityReduction

N_SAMPLES, N_FEATURES = 40, 20


def make_dataset(n_precomputed_components=None):
    X = np.random.randn(N_SAMPLES, N_FEATURES).astype(np.float32)
    ids = [f"s{i}" for i in range(N_SAMPLES)]
    dim_reduced = np.random.randn(N_SAMPLES, n_precomputed_components) if n_precomputed_components else None

    dataset = MagicMock()
    dataset.encoded_data = EncodedData(examples=X, example_ids=ids, encoding="test_encoding",
                                       dimensionality_reduced_data=dim_reduced)
    dataset.get_example_ids.return_value = ids
    dataset.get_metadata_fields.return_value = []
    return dataset


def test_pca_arbitrary_components_csv_and_variance(tmp_path):
    report = DimensionalityReduction(
        dataset=make_dataset(), result_path=tmp_path, name="test",
        dim_red_method=PCA(n_components=5, components=[3, 4]),
    )
    result = report._generate()

    csv = pd.read_csv(tmp_path / "dimensionality_reduced_data.csv")
    for col in ["PC1", "PC2", "PC3", "PC4", "PC5"]:
        assert col in csv.columns, f"missing {col} from CSV"
    assert report._dimension_names == [["PC3", "PC4"]]

    ev = pd.read_csv(tmp_path / "explained_variance.csv")
    assert len(ev) == 5
    assert ev["cumulative_explained_variance_ratio"].iloc[-1] <= 1.0 + 1e-6


def test_kernel_pca_explained_variance_flag(tmp_path):
    for compute_var, should_exist in [(False, False), (True, True)]:
        out = tmp_path / str(compute_var)
        report = DimensionalityReduction(
            dataset=make_dataset(), result_path=out, name="test",
            dim_red_method=KernelPCA(n_components=3, kernel="rbf", components=[1, 2],
                                     compute_total_variance=compute_var),
        )
        report._generate()
        assert (out / "explained_variance.csv").exists() == should_exist
        if should_exist:
            ev = pd.read_csv(out / "explained_variance.csv")
            assert len(ev) == 3 and (ev["explained_variance_ratio"] >= 0).all()


def test_umap_tsne_no_explained_variance(tmp_path):
    for name, method in [("umap", UMAP(n_components=2, components=[1, 2])),
                          ("tsne", TSNE(n_components=2, components=[1, 2]))]:
        out = tmp_path / name
        DimensionalityReduction(dataset=make_dataset(), result_path=out, name=name,
                                 dim_red_method=method)._generate()
        assert not (out / "explained_variance.csv").exists()


def test_precomputed_data_all_components_in_csv(tmp_path):
    report = DimensionalityReduction(
        dataset=make_dataset(n_precomputed_components=4), result_path=tmp_path, name="test",
        components=[3, 4],
    )
    assert report.check_prerequisites()
    report._generate()

    csv = pd.read_csv(tmp_path / "dimensionality_reduced_data.csv")
    for col in ["dimension_1", "dimension_2", "dimension_3", "dimension_4"]:
        assert col in csv.columns, f"missing {col} from CSV"
    assert report._dimension_names == [["dimension_3", "dimension_4"]]


def test_invalid_components_raise():
    with pytest.raises(AssertionError, match="n_components"):
        PCA(n_components=2, components=[3, 4])
    with pytest.raises(AssertionError):
        PCA(n_components=5, components=[0, 1])