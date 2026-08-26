import os

import numpy as np
import pytest

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.EncodedData import EncodedData
from immuneML.environment.Constants import Constants
from immuneML.environment.Label import Label
from immuneML.ml_methods.classifiers.LogRegressionCustomPenalty import LogRegressionCustomPenalty
from immuneML.ml_methods.util.Util import Util

LABEL = Label("disease", [0, 1])
FEATURE_NAMES = [f"feature_{i}" for i in range(10)]


@pytest.fixture(autouse=True)
def set_cache_type():
    os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name


def _encoded_data(n_examples=60, n_features=10, seed=1):
    rng = np.random.RandomState(seed)
    X = rng.rand(n_examples, n_features).astype(np.float32)
    y = np.array([i % 2 for i in range(n_examples)])
    return EncodedData(examples=X, labels={"disease": y}, feature_names=FEATURE_NAMES[:n_features])


def _fitted_model(backend, **kwargs):
    # n_splits must be >= 3: glmnet's LogitNet only sets lambda_best_ (needed for predict/predict_proba)
    # when cross-validation actually runs, which it only does for n_splits >= 3.
    device = "cpu" if backend == "torch" else None
    model = LogRegressionCustomPenalty(backend=backend, alpha=1, n_lambda=5, n_splits=3, max_iter=50,
                                       device=device, random_state=1, **kwargs)
    model.fit(_encoded_data(), LABEL)
    return model


def test_fit_glmnet():
    model = _fitted_model("glmnet")
    assert model.model is not None


def test_fit_torch():
    model = _fitted_model("torch")
    assert model.model is not None
    assert hasattr(model.model, "linear_")


@pytest.mark.parametrize("backend", ["glmnet", "torch"])
def test_predict(backend):
    model = _fitted_model(backend)
    result = model.predict(EncodedData(_encoded_data(n_examples=4).examples), LABEL)

    assert "disease" in result
    assert len(result["disease"]) == 4
    assert all(p in [0, 1] for p in result["disease"])


@pytest.mark.parametrize("backend", ["glmnet", "torch"])
def test_predict_proba(backend):
    model = _fitted_model(backend)
    result = model.predict_proba(EncodedData(_encoded_data(n_examples=4).examples), LABEL)

    assert "disease" in result
    assert set(result["disease"].keys()) == {0, 1}
    proba_pos = result["disease"][1]
    assert len(proba_pos) == 4
    assert all(0.0 <= p <= 1.0 for p in proba_pos)


def test_pos_weight_balanced_glmnet():
    model = LogRegressionCustomPenalty(backend="glmnet", alpha=1, n_lambda=5, n_splits=3, max_iter=50,
                                       random_state=1, pos_weight="balanced")
    model.fit(_encoded_data(), LABEL)
    assert model.model is not None


def test_pos_weight_numeric_glmnet():
    model = LogRegressionCustomPenalty(backend="glmnet", alpha=1, n_lambda=5, n_splits=3, max_iter=50,
                                       random_state=1, pos_weight=3.5)
    model.fit(_encoded_data(), LABEL)
    assert model.model is not None


def test_pos_weight_with_torch_backend_raises():
    with pytest.raises(ValueError):
        LogRegressionCustomPenalty(backend="torch", device="cpu", pos_weight="balanced")


def test_resolve_sample_weight_balanced():
    model = LogRegressionCustomPenalty(backend="glmnet", pos_weight="balanced")
    y = np.array([0, 0, 0, 1])  # 3 negative, 1 positive -> positive rows should get weight 3
    weights = model._resolve_sample_weight(y)
    np.testing.assert_array_almost_equal(weights, [1.0, 1.0, 1.0, 3.0])


def test_resolve_sample_weight_numeric():
    model = LogRegressionCustomPenalty(backend="glmnet", pos_weight=2.0)
    y = np.array([0, 1, 0, 1])
    weights = model._resolve_sample_weight(y)
    np.testing.assert_array_almost_equal(weights, [1.0, 2.0, 1.0, 2.0])


def test_resolve_sample_weight_none_by_default():
    model = LogRegressionCustomPenalty(backend="glmnet")
    y = np.array([0, 1, 0, 1])
    assert model._resolve_sample_weight(y) is None


def test_resolve_sample_weight_balanced_missing_class_warns_and_returns_none(caplog):
    model = LogRegressionCustomPenalty(backend="glmnet", pos_weight="balanced")
    y = np.array([0, 0, 0, 0])  # only one class present
    assert model._resolve_sample_weight(y) is None


def test_non_penalized_features_excluded_from_penalty():
    model = LogRegressionCustomPenalty(backend="glmnet", alpha=1, n_lambda=5, n_splits=3,
                                       non_penalized_features=["feature_0", "feature_1"], random_state=1)
    model.fit(_encoded_data(), LABEL)
    assert model.non_penalized_features == ["feature_0", "feature_1"]


@pytest.mark.parametrize("backend", ["glmnet", "torch"])
def test_store_and_load(backend, tmp_path):
    model = _fitted_model(backend)
    model.store(tmp_path)
    assert (tmp_path / "model.yaml").is_file()
    assert (tmp_path / "model.pkl").is_file()

    loaded = LogRegressionCustomPenalty(backend=backend)
    loaded.load(tmp_path)
    assert loaded.model is not None
    assert loaded.feature_names == FEATURE_NAMES


def test_torch_fit_falls_back_to_cpu_on_gpu_oom(monkeypatch):
    """When fitting on 'cuda' raises a GPU out-of-memory RuntimeError, the model should retry on CPU
    instead of propagating the error."""
    model = LogRegressionCustomPenalty(backend="torch", alpha=1, n_lambda=3, n_splits=2, max_iter=20,
                                       device="cuda", random_state=1)

    original_fit_torch = LogRegressionCustomPenalty._fit_torch
    call_count = {"n": 0}

    def flaky_fit_torch(self, X, y, non_penalized_indices, n_jobs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB")
        return original_fit_torch(self, X, y, non_penalized_indices, n_jobs)

    monkeypatch.setattr(LogRegressionCustomPenalty, "_fit_torch", flaky_fit_torch)

    model.fit(_encoded_data(), LABEL)

    assert call_count["n"] == 2
    assert model.device == "cpu"
    assert model.model is not None


def test_is_gpu_oom_error_matches_known_messages():
    assert Util.is_gpu_oom_error(RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB"))
    assert Util.is_gpu_oom_error(RuntimeError("CUDA error: out of memory"))
    assert not Util.is_gpu_oom_error(RuntimeError("some other unrelated runtime error"))