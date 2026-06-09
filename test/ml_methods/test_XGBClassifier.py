import os

import numpy as np
import pytest
from scipy import sparse
from xgboost import XGBClassifier as _XGB

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.EncodedData import EncodedData
from immuneML.environment.Constants import Constants
from immuneML.environment.Label import Label
from immuneML.ml_methods.classifiers.XGBClassifier import XGBClassifier
from immuneML.util.PathBuilder import PathBuilder

LABEL = Label("disease", [0, 1])
X = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1],
              [1, 0, 1], [0, 0, 1], [1, 1, 0], [0, 0, 0]])
Y = {"disease": np.array([1, 0, 1, 0, 1, 0, 1, 0])}


@pytest.fixture(autouse=True)
def set_cache_type():
    os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name


def _fitted_model(**kwargs):
    model = XGBClassifier(**kwargs)
    model.fit(EncodedData(X, Y), LABEL)
    return model


def test_fit():
    model = XGBClassifier()
    model.fit(EncodedData(X, Y), LABEL)
    assert model.model is not None
    assert isinstance(model.model, _XGB)


def test_fit_with_kwargs():
    model = XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(EncodedData(X, Y), LABEL)
    assert model.model.get_params()["n_estimators"] == 10
    assert model.model.get_params()["max_depth"] == 3


def test_fit_sparse():
    model = XGBClassifier(n_estimators=5)
    model.fit(EncodedData(sparse.csr_matrix(X), Y), LABEL)
    assert model.model is not None


def test_predict():
    model = _fitted_model(n_estimators=5)
    result = model.predict(EncodedData(X[:2]), LABEL)

    assert "disease" in result
    assert len(result["disease"]) == 2
    assert all(p in [0, 1] for p in result["disease"])


def test_predict_proba():
    model = _fitted_model(n_estimators=5)
    result = model.predict_proba(EncodedData(X[:2]), LABEL)

    assert "disease" in result
    assert set(result["disease"].keys()) == {0, 1}
    proba_pos = result["disease"][1]
    assert len(proba_pos) == 2
    assert all(0.0 <= p <= 1.0 for p in proba_pos)


def test_get_params_includes_feature_importances():
    model = _fitted_model(n_estimators=5)
    params = model.get_params()

    assert "feature_importances" in params
    assert len(params["feature_importances"]) == X.shape[1]


def test_store(tmp_path):
    model = _fitted_model(n_estimators=5)
    model.store(tmp_path)

    assert (tmp_path / "xgb_classifier.pickle").is_file()
    assert (tmp_path / "xgb_classifier.yaml").is_file()


def test_store_and_load(tmp_path):
    model = _fitted_model(n_estimators=5)
    model.store(tmp_path)

    loaded = XGBClassifier()
    loaded.load(tmp_path)

    assert isinstance(loaded.model, _XGB)

    result = loaded.predict(EncodedData(X[:2]), LABEL)
    assert len(result["disease"]) == 2


def test_fit_with_example_weights():
    weights = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
    model = XGBClassifier(n_estimators=5)
    encoded = EncodedData(X, Y, example_weights=weights)
    model.fit(encoded, LABEL)
    assert model.model is not None