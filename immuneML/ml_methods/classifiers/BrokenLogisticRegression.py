from pathlib import Path

import dill
import numpy as np
import yaml
from sklearn.utils.validation import check_is_fitted

from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.environment.Label import Label
from immuneML.ml_methods.classifiers.MLMethod import MLMethod
from immuneML.ml_methods.util.Util import Util
from immuneML.util.PathBuilder import PathBuilder
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression



class BrokenLogisticRegression(MLMethod):
    """
    This is a wrapper of scikit-learn’s LogisticRegression class.
    The parameters are defined in the `scikit-learn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_


    **Specification arguments:**

    - parameters: a dictionary of parameters that will be directly passed to scikit-learn's class upon calling __init__()
      method; for detailed list see scikit-learn's documentation of the specific class inheriting SklearnMethod

    **YAML specification:**

        definitions:
            ml_methods:
                ml_methods:
                    log_reg:
                        LogisticRegression: # name of the class inheriting SklearnMethod
                            # sklearn parameters (same names as in original sklearn class)
                            max_iter: 1000 # specific parameter value
                            penalty: l1
    """

    FIT_CV = "fit_CV"
    FIT = "fit"

    default_parameters = {"max_iter": 1000, "solver": "saga"}

    def __init__(self, parameters: dict = None):
        super().__init__()
        parameters = {**self.default_parameters, **(parameters if parameters is not None else {})}
        self.model = None
        self.parameters = parameters

    def _fit_model(self, encoded_data: EncodedData, cores_for_training: int = 2):
        mapped_y = Util.map_to_new_class_values(encoded_data.labels[self.label.name], self.class_mapping)

        params = self.parameters.copy()
        params["n_jobs"] = cores_for_training

        self.model = SklearnLogisticRegression(**params)
        self.model.fit(X=encoded_data.examples, y=mapped_y)

        return self.model

    def _predict(self, encoded_data: EncodedData):
        self.check_is_fitted(self.label.name)

        predictions = self.model.predict(X=encoded_data.examples)

        return {self.label.name: Util.map_to_old_class_values(np.array(predictions), self.class_mapping)}

    def _predict_proba(self, encoded_data: EncodedData):
        probabilities = self.model.predict_proba(X=encoded_data.examples)
        class_names = Util.map_to_old_class_values(self.model.classes_, self.class_mapping)

        return {class_name: probabilities[:, i] for i, class_name in enumerate(class_names)}

    def get_compatible_encoders(self):
        from immuneML.encodings.evenness_profile.EvennessProfileEncoder import EvennessProfileEncoder
        from immuneML.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder
        from immuneML.encodings.onehot.OneHotEncoder import OneHotEncoder
        from immuneML.encodings.word2vec.Word2VecEncoder import Word2VecEncoder
        from immuneML.encodings.reference_encoding.MatchedSequencesEncoder import MatchedSequencesEncoder
        from immuneML.encodings.reference_encoding.MatchedReceptorsEncoder import MatchedReceptorsEncoder
        from immuneML.encodings.reference_encoding.MatchedRegexEncoder import MatchedRegexEncoder
        from immuneML.encodings.motif_encoding.MotifEncoder import MotifEncoder

        return [KmerFrequencyEncoder, OneHotEncoder, Word2VecEncoder, EvennessProfileEncoder,
                MatchedSequencesEncoder, MatchedReceptorsEncoder, MatchedRegexEncoder, MotifEncoder]

    def can_predict_proba(self) -> bool:
        return True

    def can_fit_with_example_weights(self) -> bool:
        return False

    def check_is_fitted(self, label_name: str):
        if self.label.name == label_name or label_name is None:
            return check_is_fitted(self.model,
                                   ["estimators_", "coef_", "estimator", "_fit_X", "dual_coef_", "classes_"],
                                   all_or_any=any)

    def store(self, path: Path):
        PathBuilder.build(path)
        file_path = path / f"{self._get_model_filename()}.pickle"
        with file_path.open("wb") as file:
            dill.dump(self.model, file)

        params_path = path / f"{self._get_model_filename()}.yaml"

        with params_path.open("w") as file:
            desc = {
                **(self.get_params()),
                "feature_names": self.get_feature_names(),
                "classes": self.model.classes_.tolist(),
                "class_mapping": self.class_mapping,
            }

            if self.label is not None:
                desc["label"] = self.label.get_desc_for_storage()

            yaml.dump(desc, file)

    def get_params(self):
        params = self.model.get_params()
        params["coefficients"] = self.model.coef_[0].tolist()
        params["intercept"] = self.model.intercept_.tolist()

    def _get_model_filename(self):
        return "broken_logistic_regression"

    def load(self, path: Path):
        name = f"{self._get_model_filename()}.pickle"
        file_path = path / name
        if file_path.is_file():
            with file_path.open("rb") as file:
                self.model = dill.load(file)
        else:
            raise FileNotFoundError(f"{self.__class__.__name__} model could not be loaded from {file_path}"
                                    f". Check if the path to the {name} file is properly set.")

        params_path = path / f"{self._get_model_filename()}.yaml"

        if params_path.is_file():
            with params_path.open("r") as file:
                desc = yaml.safe_load(file)
                if "label" in desc:
                    setattr(self, "label", Label(**desc["label"]))
                for param in ["feature_names", "classes", "class_mapping"]:
                    if param in desc:
                        setattr(self, param, desc[param])

